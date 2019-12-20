import math
import os
import pdb
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as F
from loader.loader import LoaderConfig
from loader.meta_dataset import (MetaDataset, MetaMultiDataset,
                                 PseudoMetaDataset)
from loader.metadata import Metadata
from nn2.environment import Environment
from nn2.model import Model
from nn2.sampler2 import Sampler
from setproctitle import setproctitle as ptitle
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from utils import utils
from utils.color import Color
from utils.helpers import Logger, LoopMananger, OptimGetter
from utils.result import ResultDict, ResultFrame

# torch.multiprocessing.set_sharing_strategy('file_system')

C = utils.getCudaManager('default')
# sig_1 = utils.getSignalCatcher('SIGINT')
# sig_2 = utils.getSignalCatcher('SIGTSTP')
logger = Logger()


def train(cfg, status, metadata, shared_sampler, shared_optim):
  ##############################################################################
  def debug():
    if rank == 0:
      return utils.ForkablePdb().set_trace()
  ##############################################################################
  assert status.mode in ['train', 'valid']
  train = True if status.mode == 'train' else False
  device = utils.get_device(
    status.rank, cfg.args.gpu_ids, cfg.ctrl.module_per_gpu)
  # gpu_id = cfg.args.gpu_ids[status.rank % len(cfg.args.gpu_ids)]
  # device = 'cpu' if gpu_id == -1 else f'cuda:{gpu_id}'
  title = f'Env|Mode:{status.mode}|Rank:{status.rank}'
  print(title)
  ptitle(title)

  # C.set_cuda(gpu_id >= 0)
  # torch.cuda.set_device(gpu_id)
  utils.random_seed(cfg.args.seed + status.rank)
  ##############################################################################
  sampler = shared_sampler.new().to(device.sampler, non_blocking=True)
  model, env = initialize(cfg, metadata, device)
  state = env.reset(status=status)
  ##############################################################################
  # TF writer
  if not train:
    print(f'Save tfrecords: {cfg.dirs.save.tfrecord}')
    writer = SummaryWriter(cfg.dirs.save.tfrecord, flush_secs=1)
    reward_best = -9999

  # ready for agents standing at different phase of episode
  if not train:
    print('\nWaiting for all the agents to get ready..')
    while not all(status.ready):
      time.sleep(1)  # idling
      ready = ['O' if r is True else 'X' for r in status.ready]
      print(f'Ready: [ {" | ".join(ready)} ]', end='\r')
    print('\nReady to go. Start training..\n')
    print(f'[Class] S: {len(env.meta_sup)} | Q: {len(env.meta_que)} | '
          f'class_balanced: {cfg.loader.class_balanced}')
    print(f'[Loader] split_method: {cfg.loader.split_method} | '
          f'batch_size: {cfg.loader.batch_size}\n')
  torch.set_grad_enabled(train)
  # loop manager
  m = LoopMananger(
      status=status,
      outer_steps=cfg.steps.outer.max,
      inner_steps=cfg.steps.inner.max,
      log_steps=cfg.steps.inner.log,
      unroll_steps=cfg.steps.inner.unroll,
      query_steps=cfg.steps.inner.query,
      anneal_steps=cfg.steps.outer.anneal,
  )
  ##############################################################################
  ##############################################################################
  for outer_step, inner_step in m:
    if m.start_of_episode() or m.start_of_unroll():
      # if m.start_of_episode():
        # print(f'start epi: {status.rank}')
      # if train or (not train and m.start_of_episode()):
      # copy from shared memory
      if not train:
        print('Load parameters from shared memory.')
      sampler.copy_state_from(shared_sampler, non_blocking=True)
      actions, values, rewards = [], [], []
      loss_encoder = 0
      if not train:
        reward_total = 0
        loss_policy_total = 0
        loss_value_total = 0
        loss_encoder_total = 0

    ############################################################################
    # RL step
    eps = cfg.sampler.eps if train else 0
    action, value, embed_sampler = sampler(state, eps, debug=not train)
    state, reward, info, terminal = env(action, loop_manager=m, status=status)
    if not cfg.sampler.encoder.reuse_model:
      # sampler encoder loss
      embed_model = info.base.s.embed
      loss_encoder += F.mse_loss(embed_sampler, embed_model.detach().to(device.sampler))
    # record results
    actions.append(action)
    values.append(value)
    rewards.append(reward)
    ############################################################################
    if not cfg.ctrl.no_log and m.log_step() and (not train or
       (cfg.ctrl.no_valid and status.rank == 0)):
    # if rank == 0 and inner_step % 10 == 0:
      print(f'\n[Dir] {cfg.dirs.save.top}')
      print(f'[Step] outer: {outer_step:4d} | inner: {inner_step:4d}')
      # Note that loss is actual loss that was used for training
      #           acc is metric acc and does not care about fc layer.
      print(f'[Model(S)] loss(act): {info.ours.s.loss:6.3f} | '
            f'loss: {info.ours.s.cls_loss.mean():6.3f} | '
            f'acc: {info.ours.s.cls_acc.float().mean():6.3f} | '
            f'sparsity: {info.ours.mask_sp:6.3f}')
      print(f'[Base (S)] loss(act): {info.base.s.loss:6.3f} | '
            f'loss: {info.base.s.cls_loss.mean():6.3f} | '
            f'acc: {info.base.s.cls_acc.float().mean():6.3f} | '
            f'sparsity: {info.base.mask_sp:6.3f}')
      if cfg.ctrl.q_track.train:
        print(f'[Model(Qt)] loss(act): {info.ours.qt.loss:6.3f} | '
              f'loss: {info.ours.qt.cls_loss.mean():6.3f} | '
              f'acc: {info.ours.qt.cls_acc.float().mean():6.3f}')
        print(f'[Base (Qt)] loss(act): {info.base.qt.loss:6.3f} | '
              f'loss: {info.ours.qt.cls_loss.mean():6.3f} | '
              f'acc: {info.base.qt.cls_acc.float().mean():6.3f}')
      probs = [f'{p:6.3f}' for p in action.probs[:, 1].tolist()[:10]]
      mask = ['   O  ' if m == 1 else '   X  '
              for m in action.mask.squeeze().tolist()[:10]]
      # probs = ' | '.join([f'{p:6.3f}' for p in action.mask.tolist()])
      print(f'[RL] action(p[:{len(probs)}]): {" | ".join(probs)}')
      print(f'[RL] action(m[:{len(mask)}]): {" | ".join(mask)}')
      print(f'[RL] reward: {reward:6.3f} | value: {value:6.3f}\n')


    if not (terminal or m.end_of_episode() or m.end_of_unroll()):
      continue  # keep stacking action/value/reward

    if terminal or m.end_of_episode():
      R = torch.tensor(0).to(device.sampler)
    else:
      R = sampler(state)[1]
    values.append(R.detach())  # For GAE

    loss_policy = 0
    loss_value = 0
    if cfg.rl.gae:
      gae = torch.tensor(0.).to(device)

    ############################################################################
    # Backtracing
    for i in reversed(range(len(rewards))):
      # utils.ForkablePdb().set_trace()
      R = rewards[i] + cfg.rl.gamma * R
      td = R - values[i]  # Temporal Difference
      # if not train:
      #   utils.forkable_pdb().set_trace()
      loss_value += (0.5 * td.pow(2)).squeeze()
      log_probs = actions[i].log_probs.mean()
      entropy = actions[i].entropy.mean()
      if not cfg.rl.gae:
        # Default actor-critic policy loss
        # utils.ForkablePdb().set_trace()
        loss_policy -= log_probs * td.detach().squeeze() #+ entropy
      else:
        # GAE(Generalized Advantage Estimation)
        delta_t = rewards[i] + cfg.rl.gamma * \
            values[i + 1].data - values[i].data
        gae *= cfg.rl.gamma * cfg.rl.tau + delta_t
        # utils.ForkablePdb().set_trace()
        entropy = actions[i].entropy.sum()
        loss_policy = loss_policy - gae.data * log_probs - 0.000001 * entropy
    ############################################################################
    # loss averaged by step number
    loss_value = loss_value / len(rewards)
    loss_policy = loss_policy / len(rewards)
    loss_encoder = loss_encoder / len(rewards)

    if train and all(status.ready):
      # update global sampler
      sampler.zero_grad()
      # utils.forkable_pdb().set_trace()
      loss_total = loss_value + loss_policy
      if not cfg.sampler.encoder.reuse_model:
        loss_total += loss_encoder
      # if loss_encoder < 1.0:
        # loss_total += loss_policy
      loss_total.backward()
      clip_grad_norm_(sampler.parameters(), cfg.sampler.grad_norm)
      sampler.copy_grad_to(shared_sampler)
      shared_optim.step()
    # detach
    sampler.detach_states()
    # log
    if not train:
      # losses
      try:
        loss_policy_total += loss_policy.tolist()
        loss_value_total += loss_value.tolist()
        loss_encoder_total += loss_encoder.tolist()
      except:
        utils.ForkablePdb().set_trace()
      # reward
      reward_trunc = sum(rewards).tolist()
      # utils.ForkablePdb().set_trace()
      reward_total += reward_trunc
      reward_trunc_avg = reward_trunc / m.unroll_steps
      reward_total_avg = reward_total / inner_step
      if not cfg.ctrl.no_log:
        print(f'[Step] outer: {outer_step:4d} | inner: {inner_step:4d}')
        print(f'[RL_Loss] policy: {loss_policy:6.3f} | '
              f'value: {loss_value:6.3f} | '
              f'encoder: {loss_encoder:6.3f} | '
              f'entropy: {entropy:6.3f}')
        print(f'[Info] self_gain: {info.r.self_gain:6.3f} | '
              f'rel_gain: {info.r.rel_gain:6.3f} | '
              f'sp: {info.r.sparsity:6.3f}')
          # print(info.r.self_gain, info.r.rel_gain, info.r.sparsity)
        e_s = ' | '.join([f'{e:6.3f}' for e in embed_sampler[0, :10].tolist()])
        e_m = ' | '.join([f'{e:6.3f}' for e in embed_model[0, :10].tolist()])
        print(f'[Encoder] sampler: {e_s}')
        print(f'[Encoder] model:   {e_m}')
        print(f'[Model_Acc(Q)] ours_prev: {info.ours.q.acc_prev:6.3f} | '
              f'ours: {info.ours.q.acc:6.3f} | base: {info.base.q.acc:6.3f}')
        print(f'[Reward] avg_truc: {reward_trunc_avg:6.3f} | '
              f'avg_total: {reward_total_avg:6.3f}')
        try:
          print(f'[1st TD] R: {R:6.3f} | V: {values[i]:6.3f} | TD: {td:6.3f}\n')
        except Exception as e:
          if not train:
            print(e)
            utils.forkable_pdb().set_trace()

    if terminal or m.end_of_episode():
      if not train:
        # TODO: when truncation lengths vary, this does not make sense
        loss_mean = dict(policy=loss_policy / m.n_trunc,
                         value=loss_value / m.n_trunc,
                         encoder=loss_encoder / m.n_trunc)
        writer.add_scalars('loss_mean', loss_mean, outer_step)
        reward_mean = reward_total / m.n_trunc
        writer.add_scalar('reward', reward_mean, outer_step)
        if reward_best < reward_total:
          reward_best = reward_total
          sampler.save(cfg.dirs.save.params)
          # torch.save(sampler.state_dict(),
          #            f'{outer_step}_{inner_step}.sampler')
          print(f'[!] Best reward({reward_best:6.3f})! Sampler saved.')
        print('End of episode.\n')
        # if terminal:
        #   print('[!] Terminal state.')
      if terminal:
        print(f'[!] Terminal state: {rank}.')
        m.next_episode()
      state = env.reset(status)
      sampler.zero_states()

  ##############################################################################
  ##############################################################################
  return None


def initialize(cfg, metadata, device):
  # loader configuration
  if cfg.loader.class_balanced:
    # class balanced sampling
    loader_cfg = LoaderConfig(
        class_size=cfg.loader.class_size,
        sample_size=cfg.loader.sample_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
    ).to(device.model, non_blocking=True)
    batch_size = cfg.loader.class_size * cfg.loader.sample_size
  else:
    # typical uniform sampling
    #   (at least 2 samples per class)
    loader_cfg = LoaderConfig(
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
    ).to(device.model, non_blocking=True)
    batch_size = cfg.loader.batch_size
  # model (belongs to enviroment)
  model = Model(
      input_dim=cfg.loader.input_dim,
      embed_dim=cfg.model.embed_dim,
      channels=cfg.model.channels,
      kernels=cfg.model.kernels,
      distance_type= cfg.model.distance_type,
      fc_pulling=cfg.model.fc_pulling,
      optim_getter=OptimGetter(cfg.model.optim, lr=cfg.model.lr),
      n_classes=len(metadata),
  )#.to(device.model, non_blocking=True)
  # environment
  env = Environment(
      model=model,
      metadata=metadata,
      loader_cfg=loader_cfg,
      data_split_method=cfg.loader.split_method,
      mask_unit=cfg.sampler.mask_unit,
      async_stream=cfg.ctrl.async_stream,
      sync_baseline=cfg.ctrl.sync_baseline,
      query_track=cfg.ctrl.q_track.train,
      max_action_collapsed=cfg.rl.max_action_collapsed,
  ).to(device.model, device.model_base, non_blocking=True)

  return model, env

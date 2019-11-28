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
from nn.output import ModelOutput
from nn.sampler2 import MaskDist, MaskMode, MaskUnit
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

def loop(mode, cfg, rank, ready, done, metadata, shared_sampler=None,
         shared_optim=None, ready_step=None):
  ##############################################################################
  def debug():
    if rank == 0:
      return utils.ForkablePdb().set_trace()
  ##############################################################################
  assert mode in ['train', 'valid', 'test']
  train = True if mode == 'train' else False
  gpu_id = cfg.args.gpu_ids[rank % len(cfg.args.gpu_ids)]
  device = 'cpu' if gpu_id == -1 else f'cuda:{gpu_id}'
  ptitle(f'RL-Evironment|MODE:{mode}|RANK:{rank}|GPU_ID:{device}')
  C.set_cuda(gpu_id >= 0)
  torch.cuda.set_device(gpu_id)
  utils.set_random_seed(cfg.args.seed + rank)
  ##############################################################################
  # loader configuration
  if cfg.loader.class_balanced:
    # class balanced sampling
    loader_cfg = LoaderConfig(
        class_size=cfg.loader.class_size,
        sample_size=cfg.loader.sample_size,
        num_workers=2,
    ).to(device, non_blocking=True)
  else:
    # typical uniform sampling
    #   (at least 2 samples per class)
    loader_cfg = LoaderConfig(
        batch_size=cfg.loader.batch_size,
        num_workers=2,
    ).to(device, non_blocking=True)

  # agent
  # utils.ForkablePdb().set_trace()
  sampler = shared_sampler.new().to(device)
  # model (belongs to enviroment)
  model = Model(
      input_dim=cfg.loader.input_dim,
      embed_dim=cfg.model.embed_dim,
      channels=cfg.model.channels,
      kernels=cfg.model.kernels,
      distance_type=cfg.model.distance_type,
      last_layer=cfg.model.last_layer,
      optim_getter=OptimGetter(cfg.model.optim, lr=cfg.model.lr),
      n_classes=len(metadata),
  ).to(device, non_blocking=True)
  # environment
  env = Environment(
      model=model,
      metadata=metadata,
      loader_cfg=loader_cfg,
      data_split_method=cfg.loader.split_method,
      mask_unit=cfg.sampler.mask_unit,
  ).to(device, non_blocking=True)

  # loop manager
  m = LoopMananger(train, rank, ready, ready_step, done, cfg.steps)
  state = env.reset()
  ##############################################################################

  # TF writer
  if not train:
    print(f'Save tfrecords: {cfg.dirs.save.tfrecord}')
    writer = SummaryWriter(cfg.dirs.save.tfrecord, flush_secs=1)
    reward_best = -9999

  # ready for agents standing at different phase of episode
  if not train:
    print('\nWaiting for all the agents to get ready..')
    while not all(ready):
      time.sleep(1)  # idling
      status = ['O' if r is True else 'X' for r in ready]
      print(f'Ready: [ {" | ".join(status)} ]', end='\r')
    print('\nReady to go. Start training..\n')

  torch.set_grad_enabled(train)
  ##############################################################################
  ##############################################################################
  for outer_step, inner_step in m:
    if m.start_of_episode() or m.start_of_unroll():
      # if train or (not train and m.start_of_episode()):
      # copy from shared memory
      if not train:
        print('Update parameters from shared memory.')
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
    action, value, embed_sampler = sampler(state)
    # action = action_.sample()
    state, reward, info, embed_model = env(action, loop_manager=m)
    loss_encoder += F.mse_loss(embed_sampler, embed_model.detach())
    actions.append(action)
    values.append(value)
    rewards.append(reward)
    ############################################################################

    if m.log_step():
      print(f'\n[Step] outer: {outer_step:4d} | inner: {inner_step:4d}')
      print(f'[Model] loss: {state.loss.mean():6.3f} |'
            f'acc: {state.acc.float().mean():6.3f} | '
            f'sparsity: {state.sparsity:6.3f}')
      print(f'[Base] loss: {info.base.loss.mean():6.3f} |'
            f'acc: {info.base.acc.float().mean():6.3f} | '
            f'sparsity: {info.base.sparsity:6.3f}\n')

    if not (m.end_of_episode() or m.end_of_unroll()):
      continue  # keep stacking action/value/reward

    if m.end_of_episode():
      R = torch.zeros(1, 1).to(device)
    else:
      R = sampler(state)[1]
    values.append(R.detach())  # For GAE

    loss_policy = 0
    loss_value = 0
    if cfg.rl.gae:
      gae = torch.zeros(1, 1).to(device)

    ############################################################################
    # Backtracing
    for i in reversed(range(len(rewards))):
      # utils.ForkablePdb().set_trace()
      R = rewards[i] + cfg.rl.gamma * R
      td = R - values[i]  # Temporal Difference
      loss_value += (0.5 * td.pow(2)).squeeze()
      if not cfg.rl.gae:
        # Default actor-critic policy loss
        # utils.ForkablePdb().set_trace()
        loss_policy -= actions[i].log_probs.sum() * \
            td.detach().squeeze() * 0.01
      else:
        # GAE(Generalized Advantage Estimation)
        with torch.no_grad():
          delta_t = rewards[i] + cfg.rl.gamma * values[i + 1] - values[i]
          gae *= cfg.rl.gamma * cfg.rl.tau + delta_t
        utils.ForkablePdb().set_trace()
        loss_policy -= actions[i].log_probs.sum() * gae \
            - 0.01 * actions[i].entropy.mean()
    ############################################################################

    if train and all(ready):
      # update global sampler
      sampler.zero_grad()
      loss_total = loss_policy + loss_value + loss_encoder
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
      print(f'[Step] outer: {outer_step:4d} | inner: {inner_step:4d}')
      print(f'[Info] acc_gain: {info.acc_gain:6.3f}')
      print(f'[Loss] policy: {loss_policy:6.3f} | '
            f'value: {loss_value:6.3f} | '
            f'encoder: {loss_value:6.3f}')
      print(f'[Reward] avg_truc: {reward_trunc_avg:6.3f} | '
            f'avg_total: {reward_total_avg:6.3f}\n')

    if m.end_of_episode():
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
          torch.save(sampler.state_dict(),
                     f'{outer_step}_{inner_step}.sampler')
          print(f'[!] Best reward({reward_best:6.3f})! Sampler saved.')
        print('End of episode.\n')
      state = env.reset()
      sampler.zero_states()
  ##############################################################################
  ##############################################################################
  return None

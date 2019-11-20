import os
import pdb
import time
from collections import OrderedDict

import gin
import numpy as np
import torch
import torch.multiprocessing
from loader.loader import LoaderConfig
from loader.meta_dataset import (MetaDataset, MetaMultiDataset,
                                 PseudoMetaDataset)
from loader.metadata import Metadata
from nn2.environment import Environment
from nn2.sampler2 import Sampler
from nn2.model import Model
from nn.output import ModelOutput
#####################################################################
from nn.sampler2 import MaskDist, MaskMode, MaskUnit
from setproctitle import setproctitle as ptitle
from torch.utils.tensorboard import SummaryWriter
from utils import utils
from utils.color import Color
from utils.helpers import LoopMananger, Logger, OptimGetter
from utils.result import ResultDict, ResultFrame

#####################################################################

# torch.multiprocessing.set_sharing_strategy('file_system')

C = utils.getCudaManager('default')
# sig_1 = utils.getSignalCatcher('SIGINT')
# sig_2 = utils.getSignalCatcher('SIGTSTP')
logger = Logger()

#####################################################################


def loop(mode, cfg, rank, done, metadata, shared_sampler=None,
         shared_optim=None):
  def debug():
    if rank == 0:
      return utils.ForkablePdb().set_trace()
  #####################################################################
  assert mode in ['train', 'valid', 'test']
  train = True if mode == 'train' else False
  gpu_id = cfg.args.gpu_ids[rank % len(cfg.args.gpu_ids)]
  device = 'cpu' if gpu_id != -1 else f'cuda:{gpu_id}'
  ptitle(f'RL-Evironment|MODE:{mode}|RANK:{rank}|GPU_ID:{device}')
  C.set_cuda(gpu_id >= 0)
  torch.cuda.set_device(gpu_id)
  utils.set_random_seed(cfg.args.seed + rank)
  # loader configuration
  if cfg.loader.class_balanced:
    # class balanced sampling
    loader_cfg = LoaderConfig(
        class_size=cfg.loader.class_size,
        sample_size=cfg.loader.sample_size,
        num_workers=0,
    ).to(device)
  else:
    # typical uniform sampling
    #   (at least 2 samples per class)
    loader_cfg = LoaderConfig(
        batch_size=cfg.loader.batch_size,
        num_workers=0,
    ).to(device)

  # agent
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
      auto_reset=False,
  ).to(device)
  # environment
  env = Environment(
      model=model,
      metadata=metadata,
      loader_cfg=loader_cfg,
      data_split_method=cfg.loader.split_method,
      mask_unit=cfg.sampler.mask_unit,
  ).to(device)

  # loop manager
  m = LoopMananger(done, cfg.steps)
  state = env.reset()

  for outer_step, inner_step in m:
    if m.start_of_episode() or m.start_of_unroll():
      # copy from shared memory
      sampler.copy_state_from(shared_sampler)
      actions, values, rewards = [], [], []

    ############################################################################
    action, value = sampler(state)
    state, reward = env(action.sample())
    actions.append(action)
    values.append(value)
    rewards.append(reward)
    ############################################################################

    if not (m.end_of_unroll() or m.end_of_episode()):
      continue  # keep stacking action/value/reward

    if m.end_of_episode():
        R = torch.zeros(1, 1).to(device)
    else:
        R = sampler(state)[1]

    policy_loss = 0
    value_loss = 0
    if cfg.rl.gae:
      gae = torch.zeros(1, 1).to(device)

    for i in reversed(range(len(rewards))):
      R = rewards[i] + cfg.rl.gamma * R
      td = R - values[i]
      value_loss += 0.5 * td.pow(2)
      if not cfg.rl.gae:
        # Default actor-critic policy loss
        policy_loss -= action[i].log_prob * td.detach().squeeze()
      else:
        # Generalized Advantage Estimation
        with torch.no_grad():
          delta_t = rewards[i] + cfg.rl.gamma * values[i + 1] - values[i]
          gae *= cfg.rl.gamma * cfg.rl.tau + delta_t
        policy_loss -= actions[i].log_probs * gae - 0.01 * actions[i].entopy

    # update global sampler
    sampler.zero_grad()
    (policy_loss + value_loss).backward()
    sampler.copy_grad_to(shared_sampler)
    shared_optim.step()
    # detach

    if m.end_of_episode():
      state = env.reset()
  return None

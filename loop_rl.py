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
from nn.model import Model
from nn.output import ModelOutput
#####################################################################
from nn.sampler2 import MaskDist, MaskMode, MaskUnit
from setproctitle import setproctitle as ptitle
from torch.utils.tensorboard import SummaryWriter
from utils import utils
from utils.color import Color
from utils.helpers import InnerStepScheduler, Logger
from utils.result import ResultDict, ResultFrame

#####################################################################

# torch.multiprocessing.set_sharing_strategy('file_system')

C = utils.getCudaManager('default')
# sig_1 = utils.getSignalCatcher('SIGINT')
# sig_2 = utils.getSignalCatcher('SIGTSTP')
logger = Logger()

#####################################################################


def loop(mode, cfg, rank, done, metadata, shared_sampler=None, shared_optim=None):
  def debug():
    if rank == 0:
      return utils.ForkablePdb().set_trace()
  #####################################################################
  assert mode in ['train', 'valid', 'test']
  train = True if mode == 'train' else False
  gpu_id = cfg.args.gpu_ids[rank % len(cfg.args.gpu_ids)]
  ptitle(f'RL-Evironment|MODE:{mode}|RANK:{rank}|GPU_ID:{gpu_id}')
  torch.cuda.set_device(gpu_id)
  C.set_cuda(gpu_id >= 0)
  utils.set_random_seed(cfg.args.seed + rank)

  # temporary args
  class_balanced = False
  data_split_method = {1: 'inclusive', 2: 'exclusive'}[2]
  model_mode = 'metric'
  anneal_outer_steps = 50

  if class_balanced:
    loader_cfg = LoaderConfig(
        class_size=10, sample_size=3, num_workers=1)
  else:
    loader_cfg = LoaderConfig(batch_size=128, num_workers=1)
  # inner_step_scheduler = InnerStepScheduler(
  #     outer_steps, inner_steps, anneal_outer_steps)
  env = Environment(
      metadata=metadata,
      data_split_method=data_split_method,
      loader_cfg=loader_cfg,
      # inner_step_scheduler=inner_step_scheduler,
      model_mode=model_mode,
      gpu_id=gpu_id,
  )

  outer_step = -1
  while not done.value:
    outer_step += 1
    time.sleep(1)
    aaa = C(torch.randn([10, 10]))
    if not train:
      print(outer_step, done.value, gpu_id)
      if outer_step == 60:
        done.value = True
    else:
      if outer_step == 10:
        debug()

  return None

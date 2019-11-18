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
  C.set_cuda(gpu_id >= 0)
  torch.cuda.set_device(gpu_id)
  utils.set_random_seed(cfg.args.seed + rank)

  if cfg.loader.class_balanced:
    # loader config: class balanced sampling
    loader_cfg = LoaderConfig(
        class_size=cfg.loader.class_size,
        sample_size=cfg.loader.sample_size,
        num_workers=0,
    )
  else:
    # loader config: typical uniform sampling
    #   (at least 2 samples per class)
    loader_cfg = LoaderConfig(
        batch_size=cfg.loader.batch_size,
        num_workers=0,
    )
  model = Model(
      last_layer=cfg.model.last_layer,
      distance_type=cfg.model.distance_type,
      optim=cfg.model.optim,
      lr=cfg.model.lr,
      n_classes=len(metadata),
  )
  env = Environment(
      model=model,
      metadata=metadata,
      loader_cfg=loader_cfg,
      data_split_method=cfg.loader.split_method,
      mask_unit=cfg.sampler.mask_unit,
  )
  sampler = Sampler(
      embed_dim=cfg.sampler.embed_dim,
      rnn_dim=cfg.sampler.rnn_dim,
      mask_unit=cfg.sampler.mask_unit,
  )

  # initialize local variables
  outer_step = -1
  state = env.reset()

  while not done.value:
    outer_step += 1
    # copy from shared memory
    sampler.copy_state_from(shared_sampler)

    for step in range(cfg.train.unroll_steps):
      pass

    time.sleep(1)
    aaa = C(torch.randn([10, 10]))
    if not train:
      print(outer_step, done.value, gpu_id)
      if outer_step == 60:
        done.value = True
    # else:
    #   if outer_step == 10:
    #     debug()

  return None

import argparse
import os
import pdb
import sys
import time
from ctypes import c_bool

import gin
import torch
import torch.multiprocessing as mp
from loader.metadata import ImagenetMetadata
from loop_rl import loop
from nn2.model import Model
from nn2.sampler2 import Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import utils
from utils.helpers import Resize, OptimGetter

IMAGENET_DIR = '/st1/dataset/learn2sample/imagenet_l2s_84_84'
DEVKIT_DIR = '/v9/whshin/imagenet/ILSVRC2012_devkit_t12'
C = utils.getCudaManager('default')

parser = argparse.ArgumentParser(description='Learning to sample')
parser.add_argument('--cpu', action='store_true', help='disable CUDA')
parser.add_argument('--volatile', action='store_true', help='no saved files.')
parser.add_argument('--config_dir', type=str, default='config',
                    help='directory name that contains YAML files.')
parser.add_argument('--config', type=str, default='default',
                    help='YAML filename to load configuration.')
parser.add_argument('--parallel', action='store_true',
                    help='use torh.nn.DataParallel')
parser.add_argument('--visible_devices', nargs='+', type=int, default=None,
                    help='for the environment variable: CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=1, help='set random seed.')
parser.add_argument('--workers', type=int, default=32,
                    help='number of traing precesses.')
parser.add_argument('--gpu_ids', type=int, default=[-1], nargs='+',
                    help='GPU to use. (use CPU if -1)')
parser.add_argument('--gpu_all', action='store_true',
                    help='Use all the GPUs currently available.')
parser.add_argument('--debug', action='store_true', help='debug mode on')


def meta_train(cfg):
  # [ImageNet 1K] meta-train:100 / meta-valid:450 / meta-test:450 (classes)
  meta_data = ImagenetMetadata.load_or_make(
      data_dir=cfg.dirs.imagenet,
      devkit_dir=cfg.dirs.imagenet_devkit,
      remake=False
  )
  meta_train, meta_valid, meta_test = meta_data.split_classes((2, 4, 4))

  print('Loading a shared sampler..')
  # encoder (feature extractor)
  if cfg.sampler.encoder.reuse_model:
    encoder = None
  else:
    encoder = Model(
          input_dim=cfg.loader.input_dim,
          embed_dim=cfg.model.embed_dim,
          channels=cfg.sampler.encoder.channels,
          kernels=cfg.sampler.encoder.kernels,
          preprocess=Resize(size=32, mode='area'),
          distance_type=cfg.model.distance_type,
      )
  # sampler
  shared_sampler = Sampler(
      embed_dim=cfg.model.embed_dim,
      rnn_dim=cfg.sampler.rnn_dim,
      mask_unit=cfg.sampler.mask_unit,
      encoder=encoder,
  )
  shared_sampler.share_memory()

  print('Loading a shared optimizer..')
  getter = OptimGetter(cfg.sampler.optim, lr=cfg.sampler.lr, shared=True)
  if cfg.sampler.encoder.reuse_model:
    params = [{'params': shared_sampler.encoder_params()},
              {'params': shared_sampler.non_encoder_params(),
               'lr': cfg.sampler.encoder.lr}]
  else:
    params = shared_sampler.parameters()
  shared_optim = getter(params)
  shared_optim.share_memory()
  #####################################################################

  # if args.save_path:
  #   writer = SummaryWriter(os.path.join(save_path, 'tfevent'))

  if not (len(cfg.args.gpu_ids) == 1 and cfg.args.gpu_ids[0] == -1):
    mp.set_start_method('spawn')  # just for GPUs
  ready = mp.Array(c_bool, [False] * (cfg.args.workers - 1))
  done = mp.Value(c_bool, False)
  print(f'Starting {cfg.args.workers} processes '
        f'with GPU ids={cfg.args.gpu_ids}')
  processes = []
  if cfg.args.workers == 1:
    trunc_size = cfg.steps.inner.max
  else:
    trunc_size = cfg.steps.inner.max // (cfg.args.workers - 1)
  # Fix later: consider scheduler!
  ready_step = range(0, cfg.steps.inner.max, trunc_size)
  for rank in tqdm(range(0, cfg.args.workers)):
    # import pdb; pdb.set_trace()
    if rank < cfg.args.workers - 1:
      # training processes (multiple)
      p = mp.Process(
          target=loop,
          kwargs=dict(
              mode='train',
              cfg=cfg,
              rank=rank,
              ready=ready,
              done=done,
              metadata=meta_train,
              shared_sampler=shared_sampler,
              shared_optim=shared_optim,
              ready_step=ready_step[rank] + 1,
          ))
    else:
      # validation process (single)
      p = mp.Process(
          target=loop,
          kwargs=dict(
              mode='valid',
              cfg=cfg,
              rank=rank,
              ready=ready,
              done=done,
              metadata=meta_valid,
              shared_sampler=shared_sampler,
              shared_optim=shared_optim,
          ))
    # import pdb; pdb.set_trace()
    p.start()
    processes.append(p)
    time.sleep(0.1)

  print(f'Forked/spawned all the processes.')
  for p in processes:
    p.join()
    time.sleep(0.1)


if __name__ == '__main__':
  print('Start_of_program.')
  cfg = utils.prepare_config_and_dirs(parser.parse_args())
  torch.manual_seed(cfg.args.seed)
  meta_train(cfg)
  print('End_of_program.')

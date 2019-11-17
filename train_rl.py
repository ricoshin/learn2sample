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
from nn2.sampler2 import Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import shared_optim as optim
from utils import utils

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


def meta_train(cfg):
  # [ImageNet 1K] meta-train:100 / meta-valid:450 / meta-test:450 (classes)
  meta_data = ImagenetMetadata.load_or_make(
      data_dir=IMAGENET_DIR, devkit_dir=DEVKIT_DIR, remake=False)
  meta_train, meta_valid, meta_test = meta_data.split_classes((2, 4, 4))

  print('Loading a shared sampler..')
  shared_sampler = Sampler(
      embed_dim=cfg.sampler.embed_dim,
      rnn_dim=cfg.sampler.rnn_dim,
      mask_mode=cfg.sampler.mask_mode,
  )
  shared_sampler.share_memory()

  print('Loading a shared optimizer..')
  shared_optim = {'rmsprop': 'SharedRMSprop', 'adam': 'SharedAdam'}[
      cfg.train.optim.lower()]
  shared_optim = getattr(optim, shared_optim)(
      shared_sampler.parameters(), lr=cfg.train.lr)
  shared_optim.share_memory()
  #####################################################################

  # if args.save_path:
  #   writer = SummaryWriter(os.path.join(save_path, 'tfevent'))

  if not (len(cfg.args.gpu_ids) == 1 and cfg.args.gpu_ids[0] == -1):
    mp.set_start_method('spawn')  # just for GPUs
  done = mp.Value(c_bool, False)
  print(f'Starting {cfg.args.workers} processes '
        f'with GPU ids={cfg.args.gpu_ids}')
  processes = []
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
              done=done,
              metadata=meta_train,
              shared_sampler=shared_sampler,
              shared_optim=shared_optim,
          ))
    else:
      # validation process (single)
      p = mp.Process(
          target=loop,
          kwargs=dict(
              mode='valid',
              cfg=cfg,
              rank=rank,
              done=done,
              metadata=meta_valid,
              shared_sampler=shared_sampler,
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
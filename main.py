import argparse
import os
import pdb
import sys
import time
from ctypes import c_bool
from dotmap import DotMap

import gin
import torch
import torch.multiprocessing as mp
from loader.metadata import ImagenetMetadata
from trainer.train import train
from trainer.test import test
from nn2.model import Model
from nn2.sampler2 import Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import utils
from utils.helpers import OptimGetter, Resize

IMAGENET_DIR = '/st1/dataset/learn2sample/imagenet_l2s_84_84'
DEVKIT_DIR = '/v9/whshin/imagenet/ILSVRC2012_devkit_t12'
C = utils.getCudaManager('default')

parser = argparse.ArgumentParser(description='Learning to sample')
parser.add_argument('--cpu', action='store_true', 
                    help='disable CUDA')
parser.add_argument('--volatile', action='store_true', 
                    help='no saved files.')
parser.add_argument('--config_dir', type=str, default='config',
                    help='directory name that contains YAML files.')
parser.add_argument('--config', type=str, default='default',
                    help='YAML filename to load configuration.')
parser.add_argument('--parallel', action='store_true',
                    help='use torh.nn.DataParallel')
parser.add_argument('--visible_devices', nargs='+', type=int, 
                    default=None,
                    help='for the environment variable: CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=1, 
                    help='set random seed.')
parser.add_argument('--workers', type=int, default=32,
                    help='number of traing precesses.')
parser.add_argument('--gpu_ids', type=int, default=[-1], nargs='+',
                    help='GPU to use. (use CPU if -1)')
parser.add_argument('--gpu_all', action='store_true',
                    help='Use all the GPUs currently available.')
parser.add_argument('--debug', action='store_true', 
                    help='debug mode on.')
parser.add_argument('--eval_dir', type=str, default='',
                    help='path to learned parameter for evaluation.')
parser.add_argument('--no_valid', action='store_true',
                    help='run training agent only.')

def main(cfg):
  # [ImageNet 1K] meta-train:100 / meta-valid:450 / meta-test:450 (classes)
  meta_data = ImagenetMetadata.load_or_make(
      data_dir=cfg.dirs.imagenet,
      devkit_dir=cfg.dirs.imagenet_devkit,
      remake=False
  )
  # meta_data, _ = meta_data.split_classes((5, 5))
  # meta_train, meta_valid, meta_test = meta_data.split_classes((1, 2, 2))
  meta_data, _ = meta_data.split_classes((60, 940))
  meta_train, meta_valid, meta_test = meta_data.split_classes((1, 1, 1))

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
        auto_reset=True,
    )
  # sampler
  sampler = Sampler(
      embed_dim=cfg.model.embed_dim,
      rnn_dim=cfg.sampler.rnn_dim,
      mask_unit=cfg.sampler.mask_unit,
      prob_clamp=cfg.sampler.prob_clamp,
      encoder=encoder,
  )
  if cfg.args.eval_dir:
    # WARNING: meta_test
    return test(cfg=cfg, metadata=meta_valid, sampler=sampler)
  ##############################################################################
  print('Loading a shared optimizer..')
  getter = OptimGetter(cfg.sampler.optim, lr=cfg.sampler.lr, shared=True)
  if cfg.sampler.encoder.reuse_model:
    params = sampler.parameters()
  else:
    params = [{'params': sampler.encoder_params()},
              {'params': sampler.non_encoder_params(),
               'lr': cfg.sampler.encoder.lr}]
  optim = getter(params)

  # allows data to go into a state where any process can use it directly
  # so passing that data as argument to different processes won't make
  # copy of that data
  sampler.share_memory()
  optim.share_memory()
  ##############################################################################
  if not (len(cfg.args.gpu_ids) == 1 and cfg.args.gpu_ids[0] == -1):
    mp.set_start_method('spawn')  # just for GPUs

  # Divide one thread to 3 modules (env, sampler, model)
  # Assign each module to each gpu
  if cfg.ctrl.module_per_gpu: 
    cfg.args.workers = len(cfg.args.gpu_ids) // 3
    cfg.args.gpu_ids = cfg.args.gpu_ids[:cfg.args.workers * 3]
    print(f'Module-based deploy. workers={cfg.args.workers}.')

  if cfg.ctrl.no_valid:
    n_train_workers = cfg.args.workers
  else:
    n_train_workers = cfg.args.workers - 1

  if cfg.args.workers == 1:
    trunc_size = cfg.steps.inner.max
  else:
    trunc_size = cfg.steps.inner.max // (n_train_workers)
  # Fix later: consider scheduler!

  # status
  ready_step = list(reversed(range(0, cfg.steps.inner.max, trunc_size)))
  ready = [False if cfg.ctrl.diverse_start else True]

  ready = mp.Array(c_bool, ready * n_train_workers)
  done = mp.Value(c_bool, False)
  ##############################################################################
  processes = []
  print(f'Starting {cfg.args.workers} processes '
        f'with GPU ids={cfg.args.gpu_ids}')
  for rank in tqdm(range(0, cfg.args.workers)):
    # import pdb; pdb.set_trace()
    if rank < n_train_workers:
      # training processes (multiple)
      p = mp.Process(
          target=train,
          kwargs=dict(
              cfg=cfg,
              metadata=meta_train,
              shared_sampler=sampler,
              shared_optim=optim,
              status=DotMap(dict(
                mode='train',
                rank=rank,
                ready=ready,
                done=done,
                ready_step=ready_step[rank] + 1,
              )),
          ))
    else:
      # validation process (single)
      p = mp.Process(
          target=train,
          kwargs=dict(
              cfg=cfg,
              metadata=meta_valid,
              shared_sampler=sampler,
              shared_optim=optim,
              status=DotMap(dict(
                mode='valid',
                rank=rank,
                ready=ready,
                done=done,
              )),
          ))
    ##############################################################################
    p.start()
    processes.append(p)
    time.sleep(0.5)

  print(f'Forked/spawned all the processes.')
  for p in processes:
    p.join()
    time.sleep(0.1)


if __name__ == '__main__':
  print('Start_of_program.')
  cfg = utils.prepare_config_and_dirs(parser.parse_args())
  utils.random_seed(cfg.args.seed)
  main(cfg)
  print('End_of_program.')

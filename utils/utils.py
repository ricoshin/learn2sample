import argparse
import importlib
import logging
import os
import pdb
import random
import shutil
import signal
import sys
import time
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
#import mock
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from dotmap import DotMap
from tensorboardX import SummaryWriter
from utils import shared_optim

_tensor_managers = {}
_cuda_managers = {}
_debuggers = {}


def getSignalCatcher(name):
  if name not in _debuggers:
    _debuggers.update({name: SignalCatcher(name)})
  return _debuggers[name]


def set_random_seed(seed, deterministic_cudnn=False):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if deterministic_cudnn:
    # performance drop might be resulted in
    torch.backends.cudnn.deterministic = True


def getCudaManager(name):
  if name not in _cuda_managers:
    _cuda_managers.update({name: CUDAManager(name)})
  return _cuda_managers[name]


def isnan(*args):
  assert all([isinstance(arg, torch.Tensor) for arg in args])
  for arg in args:
    if torch.isnan(arg):
      import pdb
      pdb.set_trace()
  return args


def get_shared_optim(optim_name, *args, **kwargs):
  optim_name = {'rmsprop': 'SharedRMSprop', 'adam': 'SharedAdam'}[
      optim_name.lower()]
  return getattr(shared_optim, optim_name)(*args, **kwargs)


def get_optim(optim_name, *args, **kwargs):
  optim_name = {'sgd': 'SGD', 'rmsprop': 'RMSprop', 'adam': 'Adam'}[
      optim_name.lower()]
  return getattr(torch.optim, optim_name)(*args, **kwargs)


def set_logger(save_path):
  # log_fmt = '%(asctime)s %(levelname)s %(message)s'
  # date_fmt = '%d/%m/%Y %H:%M:%S'
  # formatter = logging.Formatter(log_fmt, datefmt=date_fmt)
  formatter = MyFormatter()

  # set log level
  levels = dict(debug=logging.DEBUG,
                info=logging.INFO,
                warning=logging.WARNING,
                error=logging.ERROR,
                critical=logging.CRITICAL)

  log_level = levels['debug']

  # setup file handler
  file_handler = logging.FileHandler(os.path.join(save_path + 'log'))
  file_handler.setFormatter(formatter)
  file_handler.setLevel(log_level)

  # setup stdio handler
  stream_handler = logging.StreamHandler()
  stream_handler.setFormatter(formatter)
  stream_handler.setLevel(log_level)

  # get logger
  logger = logging.getLogger('main')
  logger.setLevel(log_level)

  # add file & stdio handler to logger
  logger.addHandler(file_handler)
  logger.addHandler(stream_handler)


def prepare_config_and_dirs(args: argparse.Namespace):
  assert isinstance(args, argparse.Namespace)
  # load yaml config file
  yaml_path = os.path.join(args.config_dir, args.config + '.yaml')
  with open(yaml_path, "r") as f:
    cfg = DotMap(yaml.safe_load(f))
  cfg.args = DotMap(vars(args))
  # postprocess configs
  if cfg.args.workers < 2:
    print('Warning: You need at least 2 workers for a train-valid pair.')
  if cfg.args.gpu_all and not cfg.args.debug:
    assert len(cfg.args.gpu_ids) <= torch.cuda.device_count()
    cfg.args.gpu_ids = list(range(torch.cuda.device_count()))
  if cfg.args.debug:
    print('Debugging mode: single worker / volatile mode on.')
    cfg.args.workers = 1
    cfg.args.volatile = True
  # prepare directories if needed
  if not cfg.args.volatile:
    save_dir = os.path.join(cfg.dirs.result, cfg.args.config)
    if os.path.exists(save_dir):
      print(f'Directory for saving already exists: {save_dir}')
      ans = input("Do you want to remove existing dirs and files? [Y/N]")
      if ans.lower() != 'n':
        shutil.rmtree(save_dir, ignore_errors=True)
        print(f'Removed previous dirs and files.')
    print(f'Made new directory for saving: {save_dir}')
    # yaml file backup
    os.makedirs(save_dir)
    shutil.copy(yaml_path, save_dir)
    # source code backup
    code_path = os.path.join(save_dir, 'src')
    shutil.copytree(os.path.abspath(os.path.curdir), code_path,
                    ignore=lambda src, names: {'.git', '__pycahe__', 'result'})
    # add more dirs
    cfg.dirs.save = DotMap(dict(
        tfrecord=os.path.join(save_dir, 'tfrecord')),
        image=os.path.join(save_dir, 'image'),
    )
  return cfg


class SignalCatcher(object):
  """Signal catcher for debugging."""

  def __init__(self, name):
    self.name = name
    self._signal = getattr(signal, name)
    self._cond_func = None
    self.signal_on = False
    self._set_signal()

  def _set_signal(self):
    def _toggle_func(signal, frame):
      if self.signal_on:
        self.signal_on = False
        print(f'Signal {self.name} Off!')
      else:
        self.signal_on = True
        print(f'Signal {self.name} On!')
    signal.signal(self._signal, _toggle_func)

  def is_active(self, cond=True):
    assert isinstance(cond, bool)
    if self.signal_on and cond:
      return True
    else:
      return False


class MyDataParallel(torch.nn.DataParallel):
  """To access the attributes after warpping a module with DataParallel."""

  def __getattr__(self, name):
    if name == 'module':
      # to avoid recursion
      return self.__dict__['_modules']['module']
    return getattr(self.module, name)


class ParallelizableModule(torch.nn.Module):
  """To perform submodule level parallelization by wrapping them with
  MyDataParallel recursively only if the one is subclass of itself."""

  def data_parallel_recursive_(self, is_parallel=True, recursive=False):
    if is_parallel:
      for name, module in self.named_children():
        if recursive and issubclass(module.__class__, ParallelizableModule):
          module.data_parallel_(is_parallel)
        self.__dict__['_modules'][name] = MyDataParallel(module)


class ForkablePdb(pdb.Pdb):

  _original_stdin_fd = sys.stdin.fileno()
  _original_stdin = None

  def __init__(self):
    pdb.Pdb.__init__(self, nosigint=True)

  def _cmdloop(self):
    current_stdin = sys.stdin
    try:
      if not self._original_stdin:
        # TODO: fix Bad file descriptor error
        self._original_stdin = os.fdopen(self._original_stdin_fd)
      sys.stdin = self._original_stdin
      self.cmdloop()
    finally:
      sys.stdin = current_stdin


class CUDAManager(object):
  """Global CUDA manager. Ease the pain of carrying cuda config around all over
     the modules.
  """

  def __init__(self, cuda=None, name='default'):
    self.cuda = cuda
    self.name = name
    self.parallel = False
    self.visible_devices = None

  def set_cuda(self, cuda, manual_seed=None):
    assert isinstance(cuda, bool)
    self.cuda = cuda
    # print(f"Global cuda manager '{self.name}' is set to {self.cuda}.")
    if cuda and manual_seed:
      torch.cuda.manual_seed(manual_seed)
      torch.backends.cudnn.deterministic = True
    return self

  def set_visible_devices(self, visible_devices=None):
    """no effect. remove later."""
    self.visible_devices = visible_devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # it cannot restore full visiblity when the global python process way
    #   was started from restrained visiblity in the first palce.
    if visible_devices is None:
      if 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ.pop('CUDA_VISIBLE_DEVICES')
    else:
      assert isinstance(visible_devices, (list, tuple))
      assert isinstance(visible_devices[0], int)
      devices = [str(device) for device in visible_devices]
      os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices)

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
      print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
      print('CUDA_VISIBLE_DEVICES is not specified.')
    importlib.reload(torch)
    print('Number of GPUs that are currently available: '
          f'{torch.cuda.device_count()}')

  def set_parallel(self, manual_seed=5555):
    if torch.cuda.device_count() < 3:
      if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        print('Number of GPUs that are currently available: '
              f'{torch.cuda.device_count()}')
        raise Exception('Visible number of GPU has to be larger than 3!')
    if not self.cuda:
      raise Exception('Cannnot set pararllel mode when CUDAManager.cuda=False')
    self.parallel = True
    print('Set data and model parallelization.')
    torch.cuda.manual_seed_all(manual_seed)

  def __call__(self, obj, device=None, parallel=True):
    if isinstance(obj, (list, tuple)):
      return [self.__call__(obj_, device, parallel) for obj_ in obj]
    if not hasattr(obj, 'cuda'):
      return obj  # to skip non-tensors
    if self.cuda is None:
      raise Exception("cuda configuration has to be set "
                      "by calling CUDAManager.set_cuda(boolean)")
    return obj.cuda() if self.cuda else obj
    # if (isinstance(obj, torch.nn.Module) and self.parallel and parallel
    #     and not obj.__class__ is MyDataParallel):
    #   obj = MyDataParallel(obj)
    #   print('a')


class MyFormatter(logging.Formatter):
  info_fmt = "%(message)s"
  else_fmt = '[%(levelname)s] %(message)s'

  def __init__(self, fmt="%(message)s"):
    logging.Formatter.__init__(self, fmt)

  def format(self, record):
    # Save the original format configured by the user
    # when the logger formatter was instantiated
    format_orig = self._fmt
    # Replace the original format with one customized by logging level
    if record.levelno == logging.INFO:
      self._fmt = MyFormatter.info_fmt
    else:
      self._fmt = MyFormatter.else_fmt
    # Call the original formatter class to do the grunt work
    result = logging.Formatter.format(self, record)
    # Restore the original format configured by the user
    self._fmt = format_orig

    return result

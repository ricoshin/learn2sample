import logging
import os
import shutil
import signal
import time
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
#import mock
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tensorboardX import SummaryWriter

_tensor_managers = {}
_cuda_managers = {}


# def to_gpu(cuda, obj):
#   return obj.cuda() if cuda else obj

_debuggers = {}


def getSignalCatcher(name):
  if name not in _debuggers:
    _debuggers.update({name: SignalCatcher(name)})
  return _debuggers[name]
  # global _debugger
  # if _debugger is None:
  #   _debugger = Debugger()
  # return _debugger


class SignalCatcher(object):
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


class CUDAManager(object):
  def __init__(self, cuda=None, name='default'):
    self.cuda = cuda
    self.name = name

  def set_cuda(self, cuda):
    self.cuda = cuda
    print(f"Global cuda manager '{self.name}' is set to {self.cuda}.")
    return self

  def __call__(self, obj):
    if self.cuda is None:
      raise Exception("cuda configuration has to be set "
                      "by calling CUDAManager.set_cuda(boolean)")
    return obj.cuda() if self.cuda else obj


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


def prepare_dir(gin_path):
  # make directory
  save_path = os.path.join('result', *gin_path[:-4].split('/')[1:])
  if os.path.exists(save_path):
    print(f'Directory for saving already exists: {save_path}')
    ans = input("Do you want to remove existing dirs and files? [Y/N]")
    if ans.lower() != 'n':
      shutil.rmtree(save_path, ignore_errors=True)
      print(f'Removed previous dirs and files.')
  print(f'Made new directory for saving: {save_path}')
  # gin file backup
  os.makedirs(save_path)
  shutil.copy(gin_path, save_path)
  # source code backup
  code_path = os.path.join(save_path, 'src')
  shutil.copytree(os.path.abspath(os.path.curdir), code_path,
                  ignore=lambda src, names: {'.git', '__pycahe__', 'result'})
  return save_path


def print_colorized_mask(mask):
  out_str = []
  reset = "\033[0m"
  pallete = ['160', '166', '172', '178', '184', '190']
  for m in mask.squeeze().tolist():
    m *= 100
    offset = 100 / len(pallete)
    color = pallete[int(m // offset)]
    out_str.append(f"\033[38;5;{color}m" + f'{int(m):3d}' + reset)
    # import pdb; pdb.set_trace()
  return "|".join(out_str)


def print_confidence(conf, fmt='2d'):
  ddd = fmt.join(['cf: {:', '}({:', '}/{:', '})']).format(
    *[int(c.tolist() * 100) for c in conf])
  return ddd

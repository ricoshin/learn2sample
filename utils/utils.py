import importlib
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
from loader.episode import Episode
from nn.output import ModelOutput
from tensorboardX import SummaryWriter

_tensor_managers = {}
_cuda_managers = {}
_debuggers = {}


def getSignalCatcher(name):
  if name not in _debuggers:
    _debuggers.update({name: SignalCatcher(name)})
  return _debuggers[name]


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


class CUDAManager(object):
  """Global CUDA manager. Ease the pain of carrying cuda config around all over
     the modules.
  """

  def __init__(self, cuda=None, name='default'):
    self.cuda = cuda
    self.name = name
    self.parallel = False
    self.visible_devices = None

  def set_cuda(self, cuda, manual_seed=999):
    assert isinstance(cuda, bool)
    self.cuda = cuda
    print(f"Global cuda manager '{self.name}' is set to {self.cuda}.")
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
    return obj.cuda(device if self.parallel else 0) if self.cuda else obj
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


class Printer():
  """Collection of printing functions."""
  @staticmethod
  def step_info(epoch, mode, out_step_cur, out_step_max, in_step_cur,
                in_step_max, lr):
    return (f'[{mode}|epoch:{epoch:2d}]'
            f'[out:{out_step_cur:3d}/{out_step_max}|'
            f'in:{in_step_cur:4d}/{in_step_max}][{lr:4.3f}]')

  @staticmethod
  def way_shot_query(episode):
    assert isinstance(episode, Episode)
    return (f'W/S/Q:{episode.n_classes:2d}/{episode.s.n_samples:2d}/'
            f'{episode.q.n_samples:2d}|')

  @staticmethod
  def colorized_mask(mask, min=0.0, max=1.0, multi=100, fmt='3d', vis_num=20,
                     colors=[160, 166, 172, 178, 184, 190]):
    out_str = []
    reset = "\033[0m"
    masks = mask.squeeze().tolist()
    if len(masks) > vis_num:
      id_offset = len(masks) / vis_num
    else:
      id_offset = 1
      vis_num = len(masks)
    for i in range(vis_num):
      m = masks[int(i * id_offset)] * multi
      color_offset = (max - min) * multi / len(colors)
      color = colors[int(m // color_offset) if int(m // color_offset)
                     < len(colors) else -1]
      out_str.append(f"\033[38;5;{str(color)}m" +
                     "%%%s" % fmt % int(m) + reset)
      # import pdb; pdb.set_trace()
    return f'[{"|".join(out_str)}]'

  @staticmethod
  def outputs(outputs, print_conf):
    assert isinstance(outputs, (list, tuple))
    assert(all([isinstance(out, ModelOutput) for out in outputs]))
    return "".join([out.to_text(print_conf) for out in outputs])

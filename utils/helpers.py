import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from loader.episode import Episode
from loader.metadata import Metadata
# from nn.output import ModelOutput
from utils import shared_optim, utils


def norm_col_init(weights, std=1.0):
  x = torch.randn(weights.size())
  x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
  return x


def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    weight_shape = list(m.weight.data.size())
    fan_in = np.prod(weight_shape[1:4])
    fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
    w_bound = np.sqrt(6. / (fan_in + fan_out))
    m.weight.data.uniform_(-w_bound, w_bound)
    m.bias.data.fill_(0)
  elif classname.find('Linear') != -1:
    weight_shape = list(m.weight.data.size())
    fan_in = weight_shape[1]
    fan_out = weight_shape[0]
    w_bound = np.sqrt(6. / (fan_in + fan_out))
    m.weight.data.uniform_(-w_bound, w_bound)
    m.bias.data.fill_(0)


class Resize():
  def __init__(self, size=32, mode='area'):
    self.size = size
    self.mode = mode

  def __call__(self, x):
    return F.interpolate(x, size=self.size, mode=self.mode)


class Flatten(torch.nn.Module):
  def forward(self, input):
    return input.view(input.size(0), -1)


class BaseModule(torch.nn.Module):
  def __init__(self, *args, **kwargs):
    self.device = 'cpu'
    super(BaseModule, self).__init__(*args, **kwargs)

  def to(self, device, non_blocking=False):
    self.device = device if device is not None else self.device
    return super(BaseModule, self).to(device=device, non_blocking=non_blocking)

  def cuda(self, device):
    raise NotImplemented('Use .to() instead of .cuda()')

  def cpu(self, device):
    raise NotImplemented('Use .to() instead of .cpu()')


class OptimGetter():
  """This class helps the user to renew registered parameters in optimizer
  while retaining all the other arguments such as learning rate.
  """

  def __init__(self, optim_name, shared=False, **kwargs):
    self.optim_name = optim_name
    self.shared = shared
    self.kwargs = kwargs

  def __call__(self, params):
    if self.shared:
      optim_name = {'rmsprop': 'SharedRMSprop', 'adam': 'SharedAdam'}[
          self.optim_name.lower()]
      return getattr(shared_optim, optim_name)(params, **self.kwargs)
    else:
      optim_name = {'sgd': 'SGD', 'rmsprop': 'RMSprop', 'adam': 'Adam'}[
          self.optim_name.lower()]
      return getattr(torch.optim, optim_name)(params, **self.kwargs)


class InnerStepScheduler():
  def __init__(self, outer_steps, inner_steps, anneal_outer_steps):
    assert 0 <= anneal_outer_steps <= outer_steps
    self.outer_steps = outer_steps
    self.inner_steps = inner_steps
    self.anneal_outer_steps = anneal_outer_steps

  def __call__(self, cur_outer_step, verbose=False):
    """linearly increase cur_inner_steps from 0 to 1.
      when cur_outer_step == anneal_outer_steps,
      cur_outer_step reaches to inner_steps.
    """
    if cur_outer_step < self.anneal_outer_steps:
      r = cur_outer_step / self.anneal_outer_steps
      cur_inner_steps = int(self.inner_steps * r)
    else:
      cur_inner_steps = self.inner_steps
    if verbose and not self.anneal_outer_steps == 0:
      print('Inner step scheduled : '
            f'{cur_inner_steps}/{self.inner_steps} ({r*100:5.2f}%)')
    return cur_inner_steps


class LoopMananger():
  def __init__(
    self, status, outer_steps, inner_steps, log_steps, unroll_steps=None,
    query_steps=None, anneal_steps=None):
    # TODO: too many None..
    self.train = True if status.mode == 'train' else False  # boolean
    self.outer_steps = outer_steps
    self.inner_steps = inner_steps
    self.log_steps = log_steps
    self.unroll_steps = unroll_steps
    self.query_steps = query_steps
    self.anneal_steps = anneal_steps
    self.rank = status.rank
    self.ready = status.ready
    self.ready_step = status.ready_step
    self.done = status.done
    if self.unroll_steps is not None:
      self.n_trunc = math.ceil(self.inner_steps / self.unroll_steps)
    if self.train:
      self.inner_scheduler = InnerStepScheduler(
          outer_steps=self.outer_steps,
          inner_steps=self.inner_steps,
          anneal_outer_steps=self.anneal_steps,
      )
    self.inner_step = 0
    self.outer_step = 0
    self._next_episode = False

  def __iter__(self):
    for i in range(1, self.outer_steps + 1):
      # print(f'new_epi: {self.rank} / {self.train}')
      if self.train:
        inner_steps = self.inner_scheduler(i)
      else:
        inner_steps = self.inner_steps
      for j in range(1, inner_steps + 1):
        if self._next_episode:
          self._next_episode = False
          utils.forkable_pdb().set_trace()
          break
        if (self.train and self.ready[self.rank] is False and
            j == self.ready_step):
          self.ready[self.rank] = True  # get ready!
          while not all(self.ready):
            time.sleep(1)  # wait for other agents to stand-by
            continue
        if self.train and self.done.value:
          break  #  stop all the agents when valid agent says so
        self.outer_step = i
        self.inner_step = j
        yield i, j

  def start_of_unroll(self):
    if self.unroll_steps is None:
      return False
    return (self.inner_step - 1) % self.unroll_steps == 0

  def end_of_unroll(self):
    if self.unroll_steps is None:
      return False
    return self.inner_step % self.unroll_steps == 0

  def start_of_episode(self):
    return self.inner_step == 1

  def end_of_episode(self):
    return self.inner_step == self.inner_steps

  def log_step(self):
    return self.inner_step % self.log_steps == 0

  def next_episode(self):
    self._next_episode = True


class Logger():
  """Collection of printing functions."""

  def __init__(self):
    self._log = ""

  def flush(self):
    print(self._log)
    self._log = ""

  def step_info(self, epoch, mode, out_step_cur, out_step_max, in_step_cur,
                in_step_max, lr):
    lr = lr.tolist()[0] if isinstance(lr, torch.Tensor) else lr
    self._log += (f'[{mode}|epoch:{epoch:2d}]'
                  f'[out:{out_step_cur:3d}/{out_step_max}|'
                  f'in:{in_step_cur:4d}/{in_step_max}][{lr:5.4f}]')

  # def split_info(self, meta_support, meta_query, meta_episode_iterator):
  #   assert isinstance(meta_support, Metadata)
  #   assert isinstance(meta_query, Metadata)
  #   # assert isinstance(meta_episode_iterator, MetaEpisodeIterator)
  #   # entire dataset
  #   s = meta_support
  #   q = meta_query
  #   # sampled inner dataset
  #   epi_s = meta_episode_iterator.support
  #   epi_q = meta_episode_iterator.query
  #   n_samples = meta_episode_iterator.samples_per_class
  #   self._log += (
  #       f'S({len(s)}):{len(epi_s)}w-{n_samples}s|'
  #       f'Q({len(q)}):{len(epi_q)}w-{n_samples}s|'
  #   )

  def colorized_mask(self, mask, min=0.0, max=1.0, multi=100, fmt='3d',
                     vis_num=20, colors=[160, 166, 172, 178, 184, 190],
                     cond=True):
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
    self._log += f'[{"|".join(out_str)}]'

  # def outputs(self, outputs, print_conf):
  #   assert isinstance(outputs, (list, tuple))
  #   ###########################################################
  #   from nn.reinforcement import Policy
  #   assert(all([isinstance(out, (ModelOutput, Policy)) for out in outputs]))
  #   ###################################################################
  #   self._log += "".join([out.to_text(print_conf) for out in outputs])

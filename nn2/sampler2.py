import copy
import math
import os
import sys
from collections import namedtuple
from enum import Enum, auto

import gin
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.nn import functional as F
from utils import utils
from utils.helpers import BaseModule, weights_init, norm_col_init
from utils.utils import MyDataParallel, ParallelizableModule

C = utils.getCudaManager('default')


class MaskUnit(Enum):
  """Masking unit."""
  SAMPLE = auto()    # samplewise mask
  CLASS = auto()     # classwise mask


class MaskDist(Enum):
  """Mask distribution."""
  SOFT = auto()      # simple sigmoid, attention-like mask
  DISCRETE = auto()  # hard, non-differentiable mask
  CONCRETE = auto()  # different mask from CONCRETE distribution
  RL = auto()        # simple softmax delivered to policy network in RL


class MaskMode(object):
  def __init__(self, unit, dist):
    assert isinstance(unit, MaskUnit) and isinstance(dist, MaskDist)
    self.unit = unit
    self.dist = dist

  def __repr__(self):
    return f"MaskMode({self.unit}, {self.dist})"


class Action(object):
  def __init__(self, logits):
    self.logits = logits
    self.prob = F.softmax(logits, dim=1)
    self.log_prob = F.log_softmax(logits, dim=1)
    self.entropy = -(self.log_prob * self.prob).sum(1)

  def sample(self):
    return self.prob.multinomial(1).data


class Sampler(BaseModule):
  """A Sampler module incorporating all other submodules."""
  _save_name = 'meta.params'

  def __init__(self, embed_dim, rnn_dim, mask_unit, encoder=None):
    super(Sampler, self).__init__()
    self.embed_dim = embed_dim
    self.rnn_dim = rnn_dim
    self.mask_unit = mask_unit
    self.encoder = encoder
    # rnn + linears
    input_dim = 3 * 5  # [loss, acc, dist] x [value, mean, var, skew, kurt]
    self.input_linear = nn.Linear(input_dim, rnn_dim)
    self.lstm = nn.LSTMCell(rnn_dim, rnn_dim)
    self.actor_linear = nn.Linear(rnn_dim, 2)
    self.critic_linear = nn.Linear(rnn_dim, 1)
    # initialization
    self.apply(weights_init)
    self.actor_linear.weight.data = norm_col_init(
        self.actor_linear.weight.data, 0.01)
    self.actor_linear.bias.data.fill_(0)
    self.critic_linear.weight.data = norm_col_init(
        self.critic_linear.weight.data, 1.0)
    self.critic_linear.bias.data.fill_(0)
    self.lstm.bias_ih.data.fill_(0)
    self.lstm.bias_hh.data.fill_(0)
    # lstm states
    self.reset_states()

  def mean_by_labels(self, tensor, labels):
    mean = []
    for i in set(labels.tolist()):
      mean.append(tensor[labels == i].mean(0, keepdim=True))
    return torch.cat(mean, dim=0)

  def moments_statistics(self, x, dim, eps=1e-8, detach=True):
    """returns: [mean, variance, skewness, kurtosis]"""
    mean = x.mean(dim=dim, keepdim=True)
    var = x.var(dim=dim, keepdim=True)
    std = var.sqrt()
    skew = ((x - mean)**3 / (std**3 + eps)).mean(dim=dim, keepdim=True) + eps
    kurt = ((x - mean)**4 / (std**4 + eps)).mean(dim=dim, keepdim=True) + eps
    out = [mean, var, skew, kurt]
    if detach:
      out = list(map(lambda x: getattr(x, 'detach')(), out))
    return torch.cat(out, dim=-1)

  def preprocess(self, x, labels):
    x_mean = self.mean_by_labels(x, labels)
    x_stat = self.moments_statistics(x_mean, dim=0)
    x_mean = x_mean.unsqueeze(1)
    x_stat = x_stat.repeat(x_mean.size(0), 1)
    return torch.cat([x_mean, x_stat], dim=1)

  def zero_states(self, batch_size):
    self.hx = torch.zeros(batch_size, self.rnn_dim)
    self.cx = torch.zeros(batch_size, self.rnn_dim)

  def reset_states(self):
    self.hx, self.cx = None, None

  def detach_states(self):
    if not hasattr(self, 'hx'):
      raise RuntimeError('Sampler.hx does NOT exist!')
    self.hx = self.hx.detach()
    self.cx = self.cx.detach()

  def forward(self, state):
    # preprocess [value, mean, skew, kurt]
    loss = self.preprocess(state.loss, state.labels)
    acc = self.preprocess(state.acc.float(), state.labels)
    dist = self.preprocess(state.dist.mean(1), state.labels)
    # concatenat preprocessed tensors [loss, acc, dist]
    x = torch.cat([loss, acc, dist], dim=1).detach()
    #   [n_class, input_dim]
    # create states at the frist step
    if self.hx is None:
      assert self.cx is None
      self.zero_states(x.size(0))
    # unroll one step
    x = self.input_linear(x)
    self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
    return self.critic_linear(self.hx), self.actor_linear(self.hx)

  def save(self, save_path=None):
    if save_path:
      # sys.setrecursionlimit(10000)  # workaround for recursion error
      file_path = os.path.join(save_path, Sampler._save_name)
      with open(file_path, 'wb') as f:
        torch.save(self.state_dict(), f)
      print(f'Saved meta-learned parameters as: {file_path}')
    return self

  @classmethod
  def load(cls, load_path):
    file_path = os.path.join(load_path,  Sampler._save_name)
    with open(file_path, 'rb') as f:
      state_dict = torch.load(f)
    print(f'Loaded meta-learned params from: {file_path}')
    model = cls()
    model.load_state_dict(state_dict)
    return model

  def new(self):
    return Sampler(self.embed_dim, self.rnn_dim, self.mask_unit, self.encoder)

  def copy_state_from(self, sampler_src):
    self.load_state_dict(sampler_src.state_dict())
    self.to(self.device)

  def copy_grad_to(self, sampler_tar):
    cpu = self.device == torch.device('cpu')
    for tar, src in zip(sampler_tar.parameters(), self.paramters()):
      if tar is not None and cpu:
        return
      elif cpu:
        tar._grad = src.grad
      else:
        tar._grad = src.grad.cpu()

  def cuda_parallel_(self, dict_, parallel):
    if parallel:
      for name, module in self.named_modules():
        if name in dict_.keys():
          module.cuda(dict_[name])
    else:
      self.cuda()

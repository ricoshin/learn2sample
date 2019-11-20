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
    self.lstm = nn.LSTMCell(embed_dim, rnn_dim)
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

  def preprocess(x, labels):
    loss = self.mean_by_labels(state.loss, state.labels)
    acc = self.mean_by_labels(state.acc.float(), state.labels)
    dist = self.mean_by_labels(state.dist.mean(1), state.labels)
    loss = self.moments_statistics(loss, dim=0)



  def mean_by_labels(self, tensor, labels):
    mean = []
    for i in set(labels.tolist()):
      mean.append(tensor[labels == i].mean(0, keepdim=True))
    return torch.cat(mean, dim=0)

  def forward(self, state):
    loss = self.preprocess(state.loss, state.labels)
    acc = self.preprocess(state.acc.float(), state.labels)
    dist = self.preprocess(state.dist.mean(1), state.labels)
    utils.ForkablePdb().set_trace()
    print('a')



    return mask, lr

  def initialize(self):
    self.t = 0  # timestamp
    self.loss_mean = self.acc_mean = None  # running mean

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

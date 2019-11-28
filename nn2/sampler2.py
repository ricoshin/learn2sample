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
from dotmap import DotMap
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from utils import utils
from utils.helpers import BaseModule, norm_col_init, weights_init
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
    import pdb; pdb.set_trace()
    prob = F.softmax(logits, dim=1)
    log_prob = F.log_softmax(logits, dim=1)
    self.m = Categorical(probs)

  @property
  def entropy(self):
    return self.m.entropy()

  @property
  def probs(self):
    return self.m.probs()

  def sample(self):
    max_resample = 1000
    n_resample = 0
    while True:
      instance = self.m.sample()
      probs = self.m.probs
      log_probs = self.m.log_prob(instance)
      entropy = self.m.entropy()
      sparsity = (instance == 1.).sum().float() / len(instance)
      if sparsity == 0:
        n_resample += 1
        if n_resample >= max_resample:
          print('Max resampling number exceeded! Make random mask.')
        continue
      else:
        break
    return DotMap(dict(
        instance=instance,
        probs=probs,
        log_probs=log_probs,
        entropy=entropy,
        sparsity=sparsity,
    ))


class Sampler(BaseModule):
  """A Sampler module incorporating all other submodules."""
  _save_name = 'meta.params'

  def __init__(self, embed_dim, rnn_dim, mask_unit, encoder=None):
    super(Sampler, self).__init__()
    self.embed_dim = embed_dim
    self.rnn_dim = rnn_dim
    self.mask_unit = mask_unit
    self.encoder = encoder
    # [loss, acc, dist] x [value, mean, var, skew, kurt] (batch_sz: n_class)
    classwise_input_dim = 3 * 5
    # [loss, acc, dist] x [mean, var, skew, kurt] (batch_sz: 1)
    global_input_dim = 3 * 4
    # rnn + linears
    self.classwise_linear = nn.Linear(classwise_input_dim, rnn_dim)
    self.global_linear = nn.Linear(global_input_dim, rnn_dim)
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
    self.zero_states()

  def to(self, *args, **kwargs):
    self.hx, self.cx = self.hx.to(self.device), self.cx.to(self.device)

    return super(Sampler, self).to(*args, **kwargs)

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
    x_mean_ = self.mean_by_labels(x, labels)
    x_stat_ = self.moments_statistics(x_mean_, dim=0)
    x_mean = x_mean_.unsqueeze(1)
    x_stat = x_stat_.repeat(x_mean.size(0), 1)
    return torch.cat([x_mean, x_stat], dim=1), x_stat_

  def zero_states(self):
    self.hx = torch.zeros(1, self.rnn_dim).to(self.device)
    self.cx = torch.zeros(1, self.rnn_dim).to(self.device)

  def detach_states(self):
    if not hasattr(self, 'hx'):
      raise RuntimeError('Sampler.hx does NOT exist!')
    self.hx = self.hx.detach()
    self.cx = self.cx.detach()

  def forward(self, state):
    # encoder
    if self.encoder is not None:
      encoded = self.encoder(state.meta_s)
      embed = encoded.embed
      # encoder_loss = F.mse_loss(encoded.embed, state.embed.detach())
    else:
      embed = None
      # TODO: encoded=? when self.encoder is None

    # preprocess [value, mean, skew, kurt]
    loss, loss_ = self.preprocess(encoded.loss, encoded.labels)
    acc, acc_ = self.preprocess(encoded.acc.float(), encoded.labels)
    dist, dist_ = self.preprocess(encoded.dist.mean(1), encoded.labels)
    # concatenat preprocessed tensors [loss, acc, dist]
    x_classwise = torch.cat([loss, acc, dist], dim=1).detach()
    x_global = torch.cat([loss_, acc_, dist_], dim=0).detach()
    # classwise: [n_class, input_dim] / global: [input_dim]

    # classwise linear
    x_classwise = self.classwise_linear(x_classwise)
    # global linear + rnn
    x_global = self.global_linear(x_global.unsqueeze(0))
    self.hx, self.cx = self.lstm(x_global, (self.hx, self.cx))
    x_global = self.hx
    # merge classwise + global
    x_merged = x_classwise + x_global

    # action
    logits = self.actor_linear(x_merged)
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy= -(log_probs * probs).sum(1)
    action = DotMap(dict(probs=probs, log_probs=log_probs, entropy=entropy))

    # value
    value = self.critic_linear(x_global).squeeze()
    return action, value, embed

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
    return Sampler(
      self.embed_dim, self.rnn_dim, self.mask_unit, self.encoder.new())

  def copy_state_from(self, sampler_src, non_blocking=False):
    self.load_state_dict(sampler_src.state_dict())
    self.to(self.device, non_blocking=non_blocking)

  def copy_grad_to(self, sampler_tar):
    for tar, src in zip(sampler_tar.parameters(), self.parameters()):
      tar._grad = src.grad.to(tar.device)

  def encoder_params(self):
    if self.encoder is None:
      raise RuntimeError('Sampler does NOT have encoder!')
    return self.encoder.parameters()

  def non_encoder_params(self):
    return iter([p for n, p in self.named_parameters() if 'encoder' in n])

  def cuda_parallel_(self, dict_, parallel):
    if parallel:
      for name, module in self.named_modules():
        if name in dict_.keys():
          module.cuda(dict_[name])
    else:
      self.cuda()

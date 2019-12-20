import copy
import math
import os
import random
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
    import pdb
    pdb.set_trace()
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
    max_resample = 10000
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

  def __init__(self, embed_dim, rnn_dim, mask_unit, prob_clamp, encoder=None):
    super(Sampler, self).__init__()
    self.embed_dim = embed_dim
    self.rnn_dim = rnn_dim
    self.mask_unit = mask_unit
    self.prob_clamp = prob_clamp
    self.encoder = encoder
    # [loss, acc, dist] x [value, mean, var, skew, kurt] (batch_sz: n_class)
    classwise_input_dim = 3 * 5
    # [loss, acc, dist] x [mean, var, skew, kurt] (batch_sz: 1)
    global_input_dim = 3 * 4
    # rnn + linears
    self.classwise_linear = nn.Linear(classwise_input_dim, rnn_dim)
    self.classwise_linear2 = nn.Linear(classwise_input_dim, rnn_dim)
    self.group_linear = nn.Linear(rnn_dim, rnn_dim)
    self.tanh = nn.Tanh()
    self.global_linear = nn.Linear(global_input_dim, rnn_dim)
    self.actor_lstm = nn.LSTMCell(rnn_dim, rnn_dim)
    self.critic_lstm = nn.LSTMCell(rnn_dim, rnn_dim)
    self.actor_linear = nn.Linear(rnn_dim, 2)
    self.critic_linear = nn.Linear(rnn_dim, 1)
    self.batch_norm = nn.BatchNorm1d(
      rnn_dim, affine=False, track_running_stats=False)
    # initialization
    self.apply(weights_init)
    self.actor_linear.weight.data = norm_col_init(
        self.actor_linear.weight.data, 0.01)
    self.actor_linear.bias.data.fill_(0)
    self.critic_linear.weight.data = norm_col_init(
        self.critic_linear.weight.data, 1.0)
    self.critic_linear.bias.data.fill_(0)
    self.actor_lstm.bias_ih.data.fill_(0)
    self.actor_lstm.bias_hh.data.fill_(0)
    self.critic_lstm.bias_ih.data.fill_(0)
    self.critic_lstm.bias_hh.data.fill_(0)
    # lstm states
    self.zero_states()

  def to(self, device, non_blocking=False):
    self.actor_hx = self.actor_hx.to(device, non_blocking=non_blocking)
    self.actor_cx = self.actor_cx.to(device, non_blocking=non_blocking)
    self.critic_hx = self.critic_hx.to(device, non_blocking=non_blocking)
    self.critic_cx = self.critic_cx.to(device, non_blocking=non_blocking)
    return super(Sampler, self).to(device, non_blocking)

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
    out = [mean, std, skew, kurt]
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
    self.actor_hx = torch.zeros(1, self.rnn_dim).to(self.device)
    self.actor_cx = torch.zeros(1, self.rnn_dim).to(self.device)
    self.critic_hx = torch.zeros(1, self.rnn_dim).to(self.device)
    self.critic_cx = torch.zeros(1, self.rnn_dim).to(self.device)

  def detach_states(self):
    if not hasattr(self, 'actor_hx'):
      raise RuntimeError('LSTM states have NOT been initialized!')
    self.actor_hx = self.actor_hx.detach()
    self.actor_cx = self.actor_cx.detach()
    self.critic_hx = self.critic_hx.detach()
    self.critic_cx = self.critic_cx.detach()

  def forward(self, state, eps=0.0, debug=False):
    # encoder
    if self.encoder is not None:
      encoded = self.encoder(state.meta_s.to(self.device))
      embed = encoded.embed
      # encoder_loss = F.mse_loss(encoded.embed, state.embed.detach())
    else:
      encoded = state
      embed = None
      # TODO: encoded=? when self.encoder is None

    # preprocess [value, mean, skew, kurt]
    cls_loss = encoded.cls_loss / np.log(encoded.n_classes)
    cls_dist = encoded.cls_dist.mean(1)
    cls_dist = (cls_dist - cls_dist.min()) / cls_dist.max()
    # utils.forkable_pdb().set_trace()
    loss, loss_ = self.preprocess(cls_loss, encoded.labels)
    acc, acc_ = self.preprocess(encoded.cls_acc.float(), encoded.labels)
    dist, dist_ = self.preprocess(cls_dist, encoded.labels)
    # concatenat preprocessed tensors [loss, acc, dist]
    x_classwise_ = torch.cat([loss, acc, dist], dim=1).detach()
    x_global = torch.cat([loss_, acc_, dist_], dim=0).detach()
    # classwise: [n_class, input_dim] / global: [input_dim]

    # classwise linear
    x_classwise = self.classwise_linear(x_classwise_)
    # x_classwise = self.batch_norm(x_classwise)
    # x_classwise = (x_classwise - x_classwise.mean(0)).div(x_classwise.std(0))
    # utils.forkable_pdb().set_trace()

    # global linear + rnn
    x_global = self.global_linear(x_global.unsqueeze(0))
    self.actor_hx, self.actor_cx = self.actor_lstm(
      x_global, (self.actor_hx, self.actor_cx))
    x_global = self.actor_hx

    # action
    x_merged = x_classwise + x_global
    logits = self.actor_linear(x_merged)
    probs = F.softmax(logits, dim=1).clamp(
      0 + self.prob_clamp, 1 - self.prob_clamp)
    mask, collapsed = self.sample_mask(probs, eps, debug)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -(log_probs * probs).sum(1)
    log_probs = log_probs.gather(1, mask)
    action = DotMap(dict(
        mask=mask,
        probs=probs,
        log_probs=log_probs,
        entropy=entropy,
        collapsed=collapsed,
    ))

    if torch.isnan(probs).any():
      utils.forkable_pdb().set_trace()

    # value
    # utils.forkable_pdb().set_trace()
    x_classwise = self.classwise_linear2(x_classwise_) + x_global

    x_selected = x_classwise[mask.bool().squeeze()].mean(0, keepdim=True)
    x_populated = x_classwise.mean(0, keepdim=True)
    x_selected = self.tanh(self.group_linear(x_selected))
    x_populated = self.tanh(self.group_linear(x_populated))
    x_relative = x_populated + x_selected
    self.critic_hx, self.critic_cx = self.critic_lstm(
      x_relative, (self.critic_hx, self.critic_cx))
    value = self.critic_linear(self.critic_hx).squeeze()
    return action, value, embed

  def random_mask(self, size):
    return (torch.ones(size) * 0.5).multinomial(1).data.to(self.device)

  def sample_mask(self, probs, eps_greedy=0.1, debug=False):
    eps = 1e-8
    collapsed = False
    n_iter, max_iter = 0, 1000
    while True:
      n_iter += 1
      if ((probs == 0).all() or (probs < 0).any() or
          torch.isnan(probs).any() or torch.isinf(probs).any()):
        if (probs == 0).all():
          print('Zero Probs! Random_mask will be applied!')
        elif (probs < 0).any().all():
          print('Too low Probs! Random_mask will be applied!')
        elif torch.isnan(probs).any():
          print('Nan Probs! Random_mask will be applied!')
        elif torch.isinf(probs).any():
          print('Inf Probs! Random_mask will be applied!')
        collapsed = True
        mask = self.random_mask(probs.size())
      elif random.uniform(0, 1) < eps_greedy:
        mask = self.random_mask(probs.size())
      elif (probs[:, 1] < eps).all():
        print('Too small or large Mask! Random_mask will be applied!')
        collapsed = True
        mask = self.random_mask(probs.size())
      else:
        mask = probs.clamp(eps, 1 - eps).multinomial(1).data
      if n_iter >= max_iter:
        print('Resampling number exceeded maximum iteration! '
              'Random mask will be applied!')
        mask = self.random_mask(probs.size())
        collapsed = True
      if (mask > 0).any():
        break
    return mask, collapsed

  def save(self, save_path=None, file_name=None):
    if save_path:
      if file_name is None:
        file_name = Sampler._save_name
      # sys.setrecursionlimit(10000)  # workaround for recursion error
      file_path = os.path.join(save_path, file_name)
      with open(file_path, 'wb') as f:
        torch.save(self.state_dict(), f)
      print(f'Saved meta-learned parameters as: {file_path}')
    return self

  def load(self, load_path, device=None, file_name=None):
    if file_name is None:
      file_name = Sampler._save_name
    file_path = os.path.join(load_path,  file_name)
    device = self.device if device is None else device
    with open(file_path, 'rb') as f:
      state_dict = torch.load(f, map_location=device)
    print(f'Loaded meta-learned params from: {file_path}')
    self.load_state_dict(state_dict)
    return self

  def new(self):
    if self.encoder is not None:
      encoder = self.encoder.new()
    else:
      encoder = None
    return Sampler(
        embed_dim=self.embed_dim,
        rnn_dim=self.rnn_dim,
        mask_unit=self.mask_unit,
        prob_clamp=self.prob_clamp,
        encoder=encoder)

  def copy_state_from(self, sampler_src, non_blocking=False):
    self.load_state_dict(sampler_src.state_dict())
    self.to(self.device, non_blocking=non_blocking)

  def copy_grad_to(self, sampler_tar):
    for tar, (name, src) in zip(
      sampler_tar.parameters(), self.named_parameters()):
      if src.grad is None:
        # print(f'[!] {name} is not being used.')
        pass
      else:
        tar._grad = src.grad.to(tar.device)

  def encoder_params(self):
    if self.encoder is None:
      raise RuntimeError('Sampler does NOT have encoder!')
    return self.encoder.parameters()

  def non_encoder_params(self):
    return iter([p for n, p in self.named_parameters() if 'encoder' not in n])

  def cuda_parallel_(self, dict_, parallel):
    if parallel:
      for name, module in self.named_modules():
        if name in dict_.keys():
          module.cuda(dict_[name])
    else:
      self.cuda()

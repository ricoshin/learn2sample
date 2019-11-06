import math
import os
import sys
from enum import Enum, auto

import gin
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.nn import functional as F
from utils import utils
from utils.utils import MyDataParallel, ParallelizableModule
from collections import namedtuple

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


# def get_sampler(sampler_type):
#   assert sampler_type in ['pre-sampler', 'post-sampler']
#   return Sampler

# class MaskGenerator(object):
#   def __init__(self, unit, dist):
#     assert isinstance(unit, MaskUnit) and isinstance(dist, MaskDist)
#     self.unit = unit
#     self.dist = dist
#
#   def generate(self, value):
#     Mask.new()
#
# class Mask(object):
#   def __init__(self, value, mode, labels):
#     assert isinstance(mode, MaskMode)
#     self.value = value
#     self.mode = mode
#
#
#   def __call__(self, target):
#     pass



@gin.configurable
class Sampler(ParallelizableModule):
  """A Sampler module incorporating all other submodules."""
  _save_name = 'meta.params'

  def __init__(self, embed_dim, rnn_dim, mask_mode):
    super(Sampler, self).__init__()
    # arguments
    h_for_each = 3 * 3 * 64
    n_timecode = 3
    moment = [.0, .5, .9, .99, .999]
    n_shared_features = len(moment) * 2 + n_timecode
    n_relative_features = 2
    # module attributes
    self.pairwse_attention = nn.Linear(h_for_each, 1)  # TODO: automatic
    self.shared_encoder = nn.Linear(n_shared_features, h_for_each)
    self.relative_encoder = nn.Linear(n_relative_features, h_for_each)
    self.mask_generator = nn.Linear(h_for_each * 2, 2)
    self.lr_generator = nn.Linear(h_for_each * 2, 1)
    # functionals
    self.softmax = nn.Softmax(dim=1)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    # misc
    self.loss_mean = self.acc_mean = None  # for running mean tracking
    self.new_momentum = lambda: C(torch.tensor(moment))
    self.m = self.new_momentum()  # running mean momentum
    self.t = 0
    self.t_scales = np.linspace(1, np.log(1000) / np.log(10), n_timecode)
    self.t_encoder = lambda t: [
        np.tanh(3 * t / 10**s - 1) for s in self.t_scales]

  def forward(self, mask_mode, feature_extraction_fn=None):
    assert isinstance(mask_mode, MaskMode)

    if feature_extraction_fn is not None:
      assert callable(feature_extraction_fn)
      output = feature_extraction_fn()
      pairwise_dist = output.pairwise_dist.detach()
      classwise_loss = output.classwise_loss.detach()
      classwise_acc = output.classwise_acc.detach()
      n_classes = output.n_classes

    # attention-based pairwse distance information
    atten = self.softmax(self.pairwse_attention(pairwise_dist))
    # [n_classes, hidden_dim]
    p = self.tanh((pairwise_dist * atten).sum(dim=1))

    # loss
    loss_class = classwise_loss / np.log(n_classes)
    loss_mean = loss_class.mean()
    loss_rel = (loss_class - loss_mean) / loss_class.std()  # [n_classes]

    # acc
    acc_class = classwise_acc
    acc_mean = classwise_acc.mean()
    acc_rel = (acc_class - acc_mean) / acc_class.std()  # [n_classes]

    # running mean of loss & acc
    loss_mean = loss_mean.repeat(len(self.m))
    acc_mean = acc_mean.repeat(len(self.m))

    if self.loss_mean is None:
      self.loss_mean = loss_mean  # [n_momentum]
      self.acc_mean = acc_mean  # [n_momentum]
    else:
      self.loss_mean = self.m * self.loss_mean + (1 - self.m) * loss_mean
      self.acc_mean = self.m * self.acc_mean + (1 - self.m) * acc_mean

    # time encoding
    self.t += 1
    time = C(torch.tensor(self.t_encoder(self.t)))

    # shared feature encoding
    #   log_loss_mean: take log to suppress too large losses
    #                  then add 1 to avoid negative infinity
    log_loss_mean = (self.loss_mean + 1).log()
    s = torch.cat([log_loss_mean, self.acc_mean, time], dim=0).detach()
    s = self.tanh(self.shared_encoder(s)).repeat(n_classes, 1)
    # s: [n_classes, hidden_dim]

    # relative feature encoding
    r = torch.stack([loss_rel, acc_rel], dim=1).detach()
    r = self.tanh(self.relative_encoder(r))
    # r: [n_classes, hidden_dim]

    # mask generation (binary classification)
    h = torch.cat([p, s + r], dim=1)
    mask_logits = self.mask_generator(h)

    # learning rate generation
    h_mean = self.tanh(h.mean(dim=0))
    lr_logits = self.lr_generator(h_mean)
    lr = F.softplus(lr_logits) * 0.1

    #####################################################################
    # on_off_style = ['softmax', 'sigmoid'][1]  # TODO: global argument

    # soft mask
    if mask_mode.dist is MaskDist.SOFT:
      # if on_off_style == 'sigmoid':
      mask = self.sigmoid(mask_logits[:, 0])  # TODO: logit[:, 1] is redundant
    elif mask_mode.dist is MaskDist.RL:
      # elif on_off_style == 'softmax':
      mask = self.softmax(mask_logits)  # This guy is for RL case
    # discrete mask
    #####################################################################
    elif mask_mode.dist is MaskDist.DISCRETE:
        mask = lr_logits[:, 0].max(dim=1)[1]  # TODO: logit[:, 1] is redundant
      # concrete mask
    elif mask_mode.dist is MaskDist.CONCRETE:
        # infer Bernoulli parameter
        mean = self.sigmoid(mask_logits[:, 0])
        sigma = F.softplus(mask_logits[:, 1]) * 0.1
        eps = torch.randn(mean.size()).to(mean.device)
        # continously relaxed Bernoulli
        probs = mean + (sigma * eps)
        temp = torch.tensor([0.1]).to(mean.device)
        mask = RelaxedBernoulli(temp, probs=probs)
        # mask = mask.rsample()
        if torch.isnan(mask.rsample()).sum() > 0:
          import pdb; pdb.set_trace()

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

  def cuda_parallel_(self, dict_, parallel):
    if parallel:
      for name, module in self.named_modules():
        if name in dict_.keys():
          module.cuda(dict_[name])
    else:
      self.cuda()

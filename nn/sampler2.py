import math
import os
import sys
import gin
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.nn import functional as F
from utils import utils
from utils.utils import MyDataParallel, ParallelizableModule

C = utils.getCudaManager('default')


@gin.configurable
class Sampler(ParallelizableModule):
  """A Sampler module incorporating all other submodules."""
  _save_name = 'meta.params'

  def __init__(self, embed_dim, rnn_dim, mask_mode):
    super(Sampler, self).__init__()
    h_for_each = 3 * 3 * 64
    moment = [.0, .5, .9, .99, 0.999]
    n_timecode = 3
    n_shared_features = len(moment) * 2 + n_timecode
    n_relative_features = 2

    self.pairwse_attention = nn.Linear(h_for_each, 1)  # TODO: automatic
    self.shared_encoder = nn.Linear(n_shared_features, h_for_each)
    self.relative_encoder = nn.Linear(n_relative_features, h_for_each)
    self.binary_classifier = nn.Linear(h_for_each * 2, 2)
    self.softmax = nn.Softmax(dim=1)
    self.loss_mean = self.acc_mean = None  # for running mean tracking
    self.m = C(torch.tensor(moment))  # running mean momentum
    self.t = 0
    self.t_scales = np.linspace(1, np.log(10000) / np.log(10), n_timecode)
    self.t_encoder = lambda t: [np.tanh(3*t / 10**s - 1) for s in self.t_scales]

  def forward(self, pairwise_dist, classwise_loss, classwise_acc, n_classes,
              hard_mask=False):
    """pairwise_dist: query(n_classes), support(n_classes), hidden_dim
       classwise_loss:
    """
    # attention-based pairwise representation
    atten = self.softmax(self.pairwse_attention(pairwise_dist))
    p = (pairwise_dist * atten).sum(dim=1)  # [n_classes, hidden_dim]

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

    # shared features encoding
    s = torch.cat([self.loss_mean, self.acc_mean, time], dim=0).detach()
    s = self.shared_encoder(s).repeat(n_classes, 1)  # [n_classes, hidden_dim]

    # relative features encoding
    r = torch.stack([loss_rel, acc_rel], dim=1).detach()
    r = self.relative_encoder(r)  # [n_classes, hidden_dim]

    # mask generation (binary classification)
    mask = self.softmax(self.binary_classifier(torch.cat([p, s + r], dim=1)))

    if hard_mask:
      mask = mask.max(dim=1)[1]  # hard mask
    else:
      mask = mask[:, 1]  # soft mask
    return mask, None  # lr is not implemented

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

  @property
  def dec_ins(self):
    return DecoderInstance()

  def cuda_parallel_(self, dict_, parallel):
    if parallel:
      for name, module in self.named_modules():
        if name in dict_.keys():
          module.cuda(dict_[name])
    else:
      self.cuda()

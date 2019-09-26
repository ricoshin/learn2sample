import math
import os

import gin
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.nn import functional as F


class Preprocessor(nn.Module):
  def __init__(self, p=10.0, eps=1e-6):
    super(Preprocessor, self).__init__()
    self.eps = eps
    self.p = p

  def forward(self, x):
    indicator = (x.abs() > math.exp(-self.p)).float()
    x1 = (x.abs() + self.eps).log() / self.p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(self.p) * x * (1 - indicator)

    return torch.cat((x1, x2), 1)


@gin.configurable
class EncoderInstance(nn.Module):
  def __init__(self, embed_dim):
    super(EncoderInstance, self).__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(3, embed_dim//2, 5, 3),  # [n_cls*n_ins, 16, 10, 10]
        # nn.BatchNorm2d(16),
        nn.ReLU(True),
        nn.Conv2d(embed_dim//2, embed_dim, 3, 3),  # [n_cls*n_ins, 32, 3, 3]
        # nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.MaxPool2d(3),  # torch.Size([n_cls*n_ins, 32])
    )

  def forward(self, x):
    """
    Args:
      x (torch.FloatTensor):
        support set(images) from the current episode.
        torch.Size([n_cls*n_ins, 3, 84, 84])
    Returns:
      x (torch.FloatTensor):
        instance-wise representation.
        torch.Size([n_cls*n_ins, feature_dim])
    """
    x = F.interpolate(x, size=32, mode='area')
    self.resized = x
    # torch.Size([n_cls*n_ins, 3, 32, 32])
    for layer in self.layers:
      x = layer(x)
    return x.squeeze()


@gin.configurable
class DecoderInstance(nn.Module):
  def __init__(self, embed_dim):
    super(DecoderInstance, self).__init__()
    self.layers = nn.Sequential(
      nn.ConvTranspose2d(embed_dim, embed_dim, 3, 1),
      nn.ReLU(True),
      nn.ConvTranspose2d(embed_dim, embed_dim//2, 4, 3),
      nn.ReLU(True),
      nn.ConvTranspose2d(embed_dim//2, 3, 5, 3),
    )

  def forward(self, x):
    # torch.Size([n_cls*n_ins, 3, 32, 32])
    x = x.view(x.size(0), x.size(1), 1, 1)
    for layer in self.layers:
      x = layer(x)
    return x


@gin.configurable
class EncoderClass(nn.Module):
  def __init__(self, embed_dim):
    super(EncoderClass, self).__init__()
    self.bn = nn.BatchNorm1d(embed_dim, affine=False)
    self.rho = nn.Linear(embed_dim, embed_dim)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU(True)
    self.lamb = nn.Parameter(torch.tensor([.9]))
    self.gamm = nn.Parameter(torch.tensor([.1]))
    self.max_pool = nn.MaxPool1d(embed_dim)

  def forward(self, x):
    """
    Args:
      x (torch.FloatTensor):
        instance-wise representation(class dim added).
        torch.Size([n_cls, n_ins, feature_dim])
    Returns:
      x (torch.FloatTensor):
        class-wise representation.
        torch.Size([n_cls, feature_dim])
    """
    # x = x.view(sup.n_classes, sup.n_samples, -1)
    # Permutation invariant set encoding
    x = x.mean(dim=1)  # [n_cls , feature_dim]
    # Permutation equivariant set encoding
    # max_pool = F.max_pool1d(x.permute(1, 0).unsqueeze(0), x.size(0))
    # x = self.relu((self.lamb * x) + (self.gamm * max_pool.squeeze()))
    # x = self.relu((self.lamb * x) + (self.gamm * x.mean(dim=0) / (x.std(dim=0) + 1e-8)))
    # import pdb; pdb.set_trace()
    x = self.bn(x)
    return x

  def save(self, save_path):
    file_path = os.path.join(save_path, Sampler._save_name)
    with open(file_path, 'wb') as f:
      torch.save(self.state_dict(), f)
    print(f'Saved meta-learned parameters as: {file_path}')

  @classmethod
  def load(cls, load_path):
    file_path = os.path.join(load_path,  Sampler._save_name)
    with open(file_path, 'rb') as f:
      state_dict = torch.load(f)
    print(f'Loaded meta-learned params from: {file_path}')
    model = cls()
    model.load_state_dict(state_dict)
    return model



@gin.configurable
class MaskGenerator(nn.Module):
  def __init__(self, embed_dim, rnn_dim, sample_mode, input_more, output_more,
               temp):
    super(MaskGenerator, self).__init__()
    self.sample_mode = sample_mode
    self.input_more = input_more
    self.output_more = output_more

    input_dim = embed_dim  # feature dim
    output_dim = 1  # mask only
    input_dim = input_dim + 3 if input_more else input_dim  # mask and loss
    output_dim = output_dim + 1 if output_more else output_dim  # lr

    self.preprocess = Preprocessor()
    self.state_linear = nn.Linear(embed_dim, embed_dim)
    self.gru = nn.GRUCell(input_dim, rnn_dim)
    self.out_linear = nn.Linear(rnn_dim, output_dim)
    self.relu = nn.ReLU(inplace=True)

    if not sample_mode:
      self.sigmoid = nn.Sigmoid()
    self.temp = temp
    self.state = None

    # self._state_init = nn.Parameter(torch.randn(1, 32))

  # @property
  # def state(self):
  #   if self._state is None:
  #     raise RuntimeError("Sampler has never been initialized. "
  #       "Run Sampler.initialize() before feeding the first data.")
  #   else:
  #     return self._state

  # @state.setter
  # def state(self, value):
  #   # TO DO: check type, dimension
  #   self._state = value

  # def init_state(self, n_batches):
  #   return self._state_init.repeat([n_batches, 1])

  def detach_(self):
    self.state.detach_()

  def init_mask(self, n_batches):
    return torch.zeros(n_batches, 1)

  def init_loss(self, n_batches):
    return self.init_mask(n_batches)

  def forward(self, x, mask=None, loss=None):
    """
    Args:
      x (torch.FloatTensor):
        class-wise representation.
        torch.Size([n_cls, feature_dim])
      mask: torch.Size([n_cls, 1])
      loss: torch.Size([n_cls, 1])
    Returns:
      x (torch.FloatTensor):
        class-wise mask layout.
        torch.Size([n_cls, 1])

    """
    # generate states from the features at the first loop.
    if self.state is None:
      state = self.state_linear(x.detach())
    else:
      state = self.state

    if self.input_more:
      mask = mask.detach()
      loss = self.preprocess(loss / np.log(loss.size(0))).detach()  # scaling by log
      x = torch.cat([x, mask, loss], dim=1)  # [n_cls , feature_dim + 2]

    self.state = self.gru(x, state)  # [n_cls , rnn_h_dim]
    x = self.out_linear(state)  # [n_cls , 1]

    if self.output_more:
      m = x[:, 0].unsqueeze(1)
      lr = x[:, 1].mean().exp()
    else:
      m = x

    if self.sample_mode:
      m = RelaxedBernoulli(self.temp, m).rsample()
    else:
      m = self.sigmoid(m / self.temp)  # 0.2

    if self.output_more:
      return m, lr
    else:
      return m  # [n_cls , 1]


@gin.configurable
class Sampler(nn.Module):
  _save_name = 'meta.params'

  def __init__(self, embed_dim, rnn_dim):
    super(Sampler, self).__init__()
    self.enc_ins = EncoderInstance(embed_dim)
    self.enc_cls = EncoderClass(embed_dim)
    self.mask_gen = MaskGenerator(embed_dim, rnn_dim)

  def detach_(self):
    self.mask_gen.detach_()

  def save(self, save_path=None):
    if save_path:
      file_path = os.path.join(save_path, Sampler._save_name)
      with open(file_path, 'wb') as f:
        torch.save(self.state_dict(), f)
      print(f'Saved meta-learned parameters as: {file_path}')

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

  # def sgd_step(self, loss, lr, second_order=False):
  #   # names, params = list(zip(named_params))
  #   grads = torch.autograd.grad(loss, self.parameters())
  #   for param, grad in zip(self.parameters(), grads):
  #     param.requires_grad_(False).copy_(param - lr * grad)

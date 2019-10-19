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


class Preprocessor(nn.Module):
  """To preprocess loss values that will be used as rnn inputs.
  Logarithm is supposed to suppress too large loss values."""
  def __init__(self, p=10.0, eps=1e-6):
    super(Preprocessor, self).__init__()
    self.eps = eps
    self.p = p

  def forward(self, x):
    indicator = (x.abs() > math.exp(-self.p)).float()
    x1 = (x.abs() + self.eps).log() / self.p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(self.p) * x * (1 - indicator)
    return torch.cat((x1, x2), 1)


class EncoderInstance(ParallelizableModule):
  """Instance-level encoder."""
  def __init__(self, embed_dim):
    super(EncoderInstance, self).__init__()
    # assert embed_dim % 2 == 0
    # embed_dim = int(embed_dim / 2)
    self.layers = nn.Sequential(
        nn.Conv2d(3, embed_dim//2, 5, 3),  # [n_cls*n_ins, 16, 10, 10]
        # nn.BatchNorm2d(embed_dim//2),
        nn.ReLU(True),
        nn.Conv2d(embed_dim//2, embed_dim, 3, 3),  # [n_cls*n_ins, 32, 3, 3]
        # nn.BatchNorm2d(embed_dim),
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
      # import pdb; pdb.set_trace()
    return x.squeeze()


class DecoderInstance(nn.Module):
  """Can be used for autoencoder pretraining."""
  def __init__(self, embed_dim):
    super(DecoderInstance, self).__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(3, 200, 3, 2),  # [n_cls*n_ins, 200, 30, 30]
      nn.BatchNorm2d(200),
      nn.ReLU(True),
      nn.Conv2d(200, 300, 3, 2),  # [n_cls*n_ins, 300, 14, 14]
      nn.BatchNorm2d(300),
      nn.ReLU(True),
      nn.Conv2d(300, 300, 3, 2),  # [n_cls*n_ins, 300, 12, 12]
      nn.BatchNorm2d(300),
      nn.ReLU(True),
      nn.Conv2d(300, 400, 3, 2),  # [n_cls*n_ins, 500, 5, 5]
      nn.BatchNorm2d(400),
      nn.ReLU(True),
      nn.Conv2d(400, 500, 4, 1),
      # nn.ConvTranspose2d(embed_dim, embed_dim, 3, 1),
      # nn.ReLU(True),
      # nn.ConvTranspose2d(embed_dim, embed_dim//2, 4, 3),
      # nn.ReLU(True),
      # nn.ConvTranspose2d(embed_dim//2, 3, 5, 3),
    )

  def forward(self, x):
    # torch.Size([n_cls*n_ins, 3, 32, 32])
    x = x.view(x.size(0), x.size(1), 1, 1)
    for layer in self.layers:
      x = layer(x)
    return x


class EncoderClass(nn.Module):
  """Class-level encoder using Deep set."""
  def __init__(self, embed_dim):
    super(EncoderClass, self).__init__()
    # assert embed_dim % 2 == 0
    # embed_dim = int(embed_dim / 2)
    self.bn = nn.BatchNorm1d(embed_dim, affine=False, track_running_stats=False)
    # self.rho = nn.Linear(embed_dim, embed_dim)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU(True)
    # self.lamb = nn.Parameter(torch.tensor([.9]))
    # self.gamm = nn.Parameter(torch.tensor([.1]))
    self.max_pool = nn.MaxPool1d(embed_dim)
    self.linear_class_embed = nn.Linear(embed_dim * 4, embed_dim)
    self.linear_task_embed = nn.Linear(embed_dim * 4, embed_dim)
    self.linear_final_embed = nn.Linear(embed_dim * 2, embed_dim)

  def moments_statistics(self, x, dim, eps=1e-8, detach=True):
    mean = x.mean(dim=dim, keepdim=True)
    var = x.var(dim=dim, keepdim=True)
    std = var.sqrt()
    skew = ((x - mean)**3 / (std**3 + eps)).mean(dim=dim, keepdim=True) + eps
    kurt = ((x - mean)**4 / (std**4 + eps)).mean(dim=dim, keepdim=True) + eps
    out = [mean, var.detach(), skew.detach(), kurt.detach()]
    if detach:
      out = list(map(lambda x: getattr(x, 'detach')(), out))
    return torch.cat(out, dim=-1)

  def forward(self, x, n_classes, mode='moments'):
    """
    Args:
      x (torch.FloatTensor):
        instance-wise representation(class dim added).
        torch.Size([n_cls, n_ins, feature_dim])
      n_classes (int):
        number of classes.
    Returns:
      x (torch.FloatTensor):
        class-wise representation.
        torch.Size([n_cls, feature_dim])
    """
    assert mode in ['simple', 'moments']
    n_total = x.size(0)
    embed_dim = x.size(1)
    n_samples = int(n_total / n_classes)
    # x_mean = x.mean(dim=0)#x.mean(dim=1)
    if mode == 'moments':
      x_class = x.view(n_classes, n_samples, embed_dim)
      x_class = self.moments_statistics(x_class, dim=1, detach=False)
      x_class = self.linear_class_embed(x_class)
      # x_class = x.mean(dim=1, keepdim=True)
      x_task = self.moments_statistics(x_class, dim=0, detach=False)
      x_task = self.linear_task_embed(x_task)
      x = torch.cat([x_task.repeat(n_classes, 1, 1), x_class], dim=-1).squeeze()
      # x = torch.cat([x_task.expand_as(x_class), x_class], dim=-1).squeeze()
      # import pdb; pdb.set_trace()
      x = self.linear_final_embed(x)
    elif mode =='simple':
      x_class = x.view(n_classes, n_samples, embed_dim)
      x_class = x_class.mean(dim=1, keepdim=True)
      # x = x_class - x_class.mean()
      # x = torch.cat(x_class.std())
      x = self.bn(x_class).squeeze()
      # x = x_class - x_class.mean()  # [n_cls , feature_dim]

    return x

    x = x.mean(dim=1)
    # x = x.view(sup.n_classes, sup.n_samples, -1)
    # Permutation invariant set encoding
    # import pdb; pdb.set_trace()
    # x = self.tanh(x.mean(dim=1) - x_mean)  # [n_cls , feature_dim]
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


class EncoderIntegrated(nn.Module):
  def __init__(self, embed_dim, mask_mode):
    super(EncoderIntegrated, self).__init__()
    assert mask_mode in ['class', 'sample']
    self.mask_mode = mask_mode
    self.enc_ins = EncoderInstance(embed_dim)
    self.enc_cls = EncoderClass(embed_dim)

  def forward(self, x, n_classes):
    assert isinstance(x, torch.Tensor)
    assert isinstance(n_classes, int)
    x = self.enc_ins(x)
    if self.mask_mode == 'class':
      x = self.enc_cls(x, n_classes)
    return x


@gin.configurable
class MaskGenerator(ParallelizableModule):
  """GRU-based mask generator."""
  def __init__(self, embed_dim, rnn_dim, mask_mode, sample_mode, input_more,
               output_more, temp):
    super(MaskGenerator, self).__init__()
    assert mask_mode in ['class', 'sample']
    self.sample_mode = sample_mode
    self.mask_mode = mask_mode
    self.input_more = input_more
    self.output_more = output_more
    self.rnn_dim = rnn_dim
    self._mask_gen_fn = None

    input_dim = embed_dim + 1 # feature dim, step
    output_dim = 1  # mask only
    input_dim = input_dim + 5 if input_more else input_dim  # mask, loss
    output_dim = output_dim + 1 if output_more else output_dim  # lr

    self.preprocess = Preprocessor()
    self.state_linear = nn.Linear(embed_dim, rnn_dim)
    self.gru = nn.GRUCell(input_dim, rnn_dim)
    self.out_linear = nn.Linear(rnn_dim, output_dim)
    self.c = nn.Parameter(torch.tensor(-3.0))
    self.relu = nn.ReLU(inplace=True)
    self._step = C(torch.tensor(0.0))
    self._step_max = C(torch.tensor(150.0))

    if not sample_mode:
      self.sigmoid = nn.Sigmoid()
    self.temp = temp
    self.state = None

  def detach_(self):
    self.state.detach_()

  @property
  def step(self):
    self._step += 1
    return self._step / self._step_max

  def init_mask(self, n_classes, n_samples):
    if self.mask_mode == 'class':
      n_batches = n_classes
    elif self.mask_mode == 'sample':
      n_batches = n_classes * n_samples
    else:
      raise Exception(f'Unknown mask_mode: {self.mask_mode}')
    return torch.ones(n_batches, 1)

  def init_loss(self, n_batches):
    return self.init_mask(n_batches)

  def static_init_state(self, n_batches):
    return C(torch.zeros(n_batches, self.rnn_dim), 1)

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
      self.state = self.state_linear(x.detach())
      # state = self.static_init_state(x.size(0))

    if self.input_more:
      # detach from the graph
      mask = mask.detach()
      loss = loss.detach()

      # averaged and relative loss
      loss = loss / np.log(loss.size(0))  # scaling by log
      loss_mean = loss.mean().repeat(loss.size(0), 1)  # averaged loss
      loss_rel = loss - loss_mean  # relative loss
      loss_mean = self.preprocess(loss_mean).detach()
      loss_rel = self.preprocess(loss_rel).detach()
      step = self.step.repeat(loss.size(0), 1)
      x = torch.cat([x, mask, loss_mean, loss_rel, step], dim=1)
      # [n_cls , feature_dim + 1 + 2 + 2]
    else:
      step = self.step.repeat(loss.size(0), 1)
      x = torch.cat([x, step], dim=1)

    self.state = self.gru(x, self.state)  # [n_cls , rnn_h_dim]
    x = self.out_linear(self.state)  # [n_cls , 1]

    if self.output_more:
      mask = x[:, 0].unsqueeze(1)
      lr = x[:, 1].mean().exp()# + self.c).exp()
    else:
      mask = x

    if self.sample_mode:
      mask = RelaxedBernoulli(self.temp, mask).rsample()
    else:
      mask = self.sigmoid(mask / self.temp)  # 0.2

    if self.output_more:
      return mask, lr
    else:
      return mask, None  # [n_cls , 1]


@gin.configurable
class Sampler(ParallelizableModule):
  """A Sampler module incorporating all other submodules."""
  _save_name = 'meta.params'

  def __init__(self, embed_dim, rnn_dim, mask_mode):
    super(Sampler, self).__init__()
    # self.enc_ins = EncoderInstance(embed_dim)
    # self.enc_cls = EncoderClass(embed_dim)
    self.encoder = EncoderIntegrated(embed_dim, mask_mode)
    self.mask_gen = MaskGenerator(embed_dim, rnn_dim, mask_mode)

  def detach_(self):
    self.mask_gen.detach_()

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

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


class EncoderInstance(nn.Module):
  def __init__(self):
    super(EncoderInstance, self).__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(3, 16, 5, 3),  # [n_cls*n_ins, 16, 10, 10]
      # nn.BatchNorm2d(16),
      nn.ReLU(True),
      nn.Conv2d(16, 32, 3, 3),  # [n_cls*n_ins, 32, 3, 3]
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
    # torch.Size([n_cls*n_ins, 3, 32, 32])
    for layer in self.layers:
      x = layer(x)
    return x.squeeze()


class EncoderClass(nn.Module):
  def __init__(self):
    super(EncoderClass, self).__init__()
    self.relu = nn.ReLU(True)
    self.bn = nn.BatchNorm1d(32, affine=False)

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
    x = self.relu(x.mean(dim=1))  # [n_cls , feature_dim]
    # Permutation equivariant set encoding
    x = self.bn(x)
    return x


class MaskGenerator(nn.Module):
  def __init__(self):
    super(MaskGenerator, self).__init__()
    self.state_linear = nn.Linear(32, 32)
    self.gru = nn.GRUCell(34, 32)  # 2 more dimensions for mask and loss
    self.out_linear = nn.Linear(32, 1)  # 1 more dimension for learning rate
    self.relu = nn.ReLU(inplace=True)
    self.sigmoid = nn.Sigmoid()
    # self._state_init = nn.Parameter(torch.randn(1, 32))
    self.state = None

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

  def forward(self, x, mask, loss):
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
    mask = mask.detach()
    loss = loss.mean(dim=1, keepdim=True).detach()
    if self.state is None:
      state = self.state_linear(x.detach())
    else:
      state = self.state
    x = torch.cat([x, mask, loss], dim=1)  # [n_cls , feature_dim + 2]
    self.state = self.gru(x, state)  # [n_cls , rnn_h_dim]
    x = self.out_linear(state)  # [n_cls , 1]
    # p = RelaxedBernoulli(0.5, p).rsample()
    x = self.sigmoid(5*x)
    return x  # [n_cls , 1]


class Sampler(nn.Module):
  def __init__(self):
    super(Sampler, self).__init__()
    self.enc_ins = EncoderInstance()
    self.enc_cls = EncoderClass()
    self.mask_gen = MaskGenerator()

  def detach_(self):
    self.mask_gen.detach_()


  # def sgd_step(self, loss, lr, second_order=False):
  #   # names, params = list(zip(named_params))
  #   grads = torch.autograd.grad(loss, self.parameters())
  #   for param, grad in zip(self.parameters(), grads):
  #     param.requires_grad_(False).copy_(param - lr * grad)

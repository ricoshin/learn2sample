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
    self.bn = nn.BatchNorm1d(32, affine=True)

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


class Sampler(nn.Module):
  def __init__(self):
    super(Sampler, self).__init__()
    self.gru = nn.GRUCell(32, 32)
    self.linear = nn.Linear(32, 1)
    self._state_init = nn.Parameter(torch.randn(1, 32))
    self._state = None

  @property
  def state(self):
    if self._state is None:
      raise RuntimeError("Sampler has never been initialized. "
        "Run Sampler.initialize() before feeding the first data.")
    else:
      return self._state

  @state.setter
  def state(self, value):
    # TO DO: check type, dimension
    self._state = value

  def initialze(self, n_batches):
    self._state = self._state_init.repeat([n_batches, 1])

  def forward(self, x):
    """
    Args:
      x (torch.FloatTensor):
        class-wise representation.
        torch.Size([n_cls, feature_dim])
    Returns:
      x (torch.FloatTensor):
        class-wise mask layout.
        torch.Size([n_cls, 1, 1, 1, 1])

    """
    self.state = self.gru(x, self.state)  # [n_cls , rnn_h_dim]
    x = self.linear(self.state)  # [n_cls , 1]
    x = RelaxedBernoulli(1.0, x).sample()
    x = x.view(-1, *([1]*4))  # [n_cls , 1, 1, 1, 1]
    return x

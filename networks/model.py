import torch
import torch.nn as nn
from torch.nn import functional as F

class Model(nn.Module):
  def __init__(self, n_classes=None):
    super(Model, self).__init__()
    ch = [3, 100, 150, 200, 250]
    kn = [5, 5, 3, 3]
    self.layers = self._stack_layers(ch, kn)
    self.n_classes = n_classes
    if self.n_classes is not None:
      self.classifier = nn.Conv2d(ch[-1], self.n_classes, 3)
    self.nll_loss = nn.NLLLoss(reduction='none')

  def _stack_layers(self, ch, kn):
    """
    Args:
      ch(list): number of channels.
      kn(list): size of kernels.
      n_classes(int): number of classes.
    Returns:
      layers(nn.Module)
    """
    layers = nn.ModuleList()
    for i in range(len(ch) - 1):
      layers.append(nn.Conv2d(ch[i], ch[i + 1], kn[i]))
      layers.append(nn.BatchNorm2d(ch[i + 1]))
      layers.append(nn.ReLU(True))
      layers.append(nn.MaxPool2d(2, 2))
    return layers

  @classmethod
  def get_init(cls):
    """Build a minimum volatile model(without the final layer)
    just for getting initial parameters."""
    return cls(n_classes=None).named_parameters()

  def init_with(self, params):
    """Initialize the model parameters by matching the names."""
    p_src = {name: param for name, param in params]}
    for name, p_tar in self.named_parameters():
      if name in p_src.keys():
        p_tar.data = p_src[name].clone()

  @staticmethod
  def compute_grad(loss, params, second_order=False):
    params = [for param in name, param in params]
    return torch.autograd.grad(loss, params, create_graph=second_order)

  def forward(self, dataset, mask=None, detach_params=False):
    """
    Args:
      dataset (loader.Dataset):
        Support or query set(imgs/labels/ids).
      mask (torch.FloatTensor):
        Classwise weighting overlay mask.
        Controls the effects from each classes.
        torch.Size([n_cls*n_ins, 1, 1, 1, 1])
      detach_params (bool):
        This flag can be used to prevent from computing second derivative of
        model parameters, while stil keeping those piplelines open heading to
        the sampler parameters.

    Returns:
      loss (torch.FloatTensor): cross entropy loss.
      acc (torch.FloatTensor): accuracy of classification.
    """
    if self.n_classes is None:
      raise RuntimeError(
        'forward() with Model.n_class=None is only meant to be called in '
        'get_init_parameters() just for getting initial parameters.')

    x = dataset.imgs
    y = dataset.labels
    n_samples = dataset.n_samples
    view_classwise = dataset.get_view_classwise_fn()

    for layer in self.layers:
      x = layer(x)

    x = F.log_softmax(self.classifier(x))
    loss = self.nll_loss(x.squeeze(), y)

    if detach_params:
      loss = loss.detach()

    if mask is not None:
      # to match dimension
      loss = view_classwise(loss)
      mask = mask.squeeze().unsqueeze(1)
      # weighted loss by sampler mask
      loss = (loss * mask).sum() / (mask.sum() * n_samples)
    else:
      loss = loss.sum()
    acc = (x.argmax(dim=1) == y).float().mean()

    return loss, acc

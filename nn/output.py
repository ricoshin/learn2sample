from collections import OrderedDict

import torch
from utils.color import Color


class ModelOutput(object):
  """A class for dynamically dealing with different types of output."""

  def __init__(self, params_name, dataset_name, n_classes, loss_sample, label,
               log_softmax, mask=None):

    self.params_name = params_name
    self.dataset_name = dataset_name
    self.n_classes = n_classes

    self.label = label
    self.log_softmax = log_softmax
    self._mask = mask
    self.predicted = log_softmax.argmax(dim=1)

    self.is_true = self.predicted == label
    self.is_false = self.predicted != label

    self.loss_sample = loss_sample
    self.element_conf = log_softmax.exp()
    self.acc_sample = self.is_true.float()

    self._conf = None
    self._loss_m = None
    self._acc_m = None
    self._loss_s = None
    self._alcc_scaled = None

  def __repr__(self):
    return ',\n'.join(['ModelOutput{',
      f'\tparams_name="{self.params_name}"',
      f'\tdata_name="{self.dataset_name}"',
      f'\tloss={self.loss}',
      f'\tacc={self.acc}',
    '}'])

  @property
  def mask(self):
    if self._mask is None:
      raise Exception("Mask does NOT exist in this output!")
    return self._mask

  @property
  def mask_mode(self):
    if self._mask is None:
      return 'None'
    if self.mask.size(0) == self.n_classes:
      return 'class'
    elif self.mask.size(0) > self.n_classes:
      return 'sample'
    else:
      raise Exception('mask size has to be >= n_classes!')

  @property
  def loss(self):
    return self.loss_sample.view(self.n_classes, -1).mean(dim=1)

  @property
  def acc(self):
    return self.acc_sample.view(self.n_classes, -1).mean(dim=1)

  def attach_mask(self, mask):
    """apply any mask, even when it was not passed to the Model."""
    if self._mask is not None:
      raise Exception("Mask already exists!")
    self._mask = mask
    return self

  @property
  def conf(self):
    """confidence."""
    if self._conf is None:
      pred = self.element_conf.max(dim=1)[0]  # prediction
      actual = self.element_conf.gather(1, self.label.view(-1, 1))  # label
      true = pred[self.is_true]  # correct prediction
      false = pred[self.is_false]   # wrong prediction
      self._conf = dict(pred=pred.mean(), actual=actual.mean(),
                        true=true.mean(), false=false.mean())
    return self._conf

  @property
  def loss_masked(self):
    """masked elementwise loss."""
    if self._loss_m is None:
      self._loss_m = self.overlay_mask(self.loss_sample, self.mask)
    return self._loss_m.mean(dim=1, keepdim=True)

  @property
  def loss_masked_mean(self):
    """mask weighted mean of loss."""
    return self.loss_masked.sum() / self.mask.sum()

  @property
  def acc_masked(self):
    """masked elementwise accuracy."""
    if self._acc_m is None:
      self._acc_m = self.overlay_mask(self.acc_sample, self.mask)
    return self._acc_m.mean(dim=1, keepdim=True)

  @property
  def acc_masked_mean(self):
    """mask weighted mean of accuracy."""
    return self.acc_masked.sum() / self.mask.sum()

  @property
  def loss_scaled_mean(self):
    """loss scaled by mean of mask."""
    if self._loss_s is None:
      self._loss_s = self.loss.mean() * self.mask.mean().detach()
    return self._loss_s

  @property
  def acc_scaled_mean(self):
    """accuracy scaled by mean of mask."""
    if self._acc_s is None:
      self._acc_s = self.acc.mean() * self.mask.mean().detach()
    return self._acc_s

  def overlay_mask(self, tensor, mask):
    assert all([isinstance(t, torch.Tensor) for t in [tensor, mask]])
    if self.mask_mode == 'class':  # classwise mask
      rest_dims = [tensor.size(i) for i in range(1, len(tensor.shape))]
      tensor = tensor.view(self.n_classes, -1, *rest_dims)
    elif self.mask_mode == 'sample':  # sample mask
      tensor = tensor.view(-1, 1)  # to match dim
    else:
      raise Exception('Cannot overy_mask when mask is unset.')
    return tensor * mask

  def _get_name(self, name):
    return '_'.join([self.params_name, self.dataset_name[0].lower(), name])

  def as_dict(self):
    """to update utils.utils.Result"""
    names = ['loss', 'acc']
    if self._mask is not None and self.params_name == 'ours':
      names.extend(('mask', 'loss_masked', 'acc_masked'))
    return OrderedDict({self._get_name(n): getattr(self, n) for n in names})

  def to_text(self, print_conf=False):
    out = f'[{self.params_name}]{self.dataset_name[0]}:'
    if self._mask is not None:
      out += f'm.{self.loss_masked_mean.tolist(): 5.2f}/'
      out += f'm.{self.acc_masked_mean.tolist()*100:5.1f}%|'
    # if self.dataset_name == 'Support':
    #   out += f'{self.loss.tolist(): 5.2f}/{self.acc.tolist()*100:5.1f}%|'
    if self.dataset_name == 'Query':
      out += f'{Color.GREEN}{self.loss.mean().tolist():5.2f}{Color.END}/'
      out += f'{Color.RED}{self.acc.mean().tolist()*100:5.1f}{Color.END}%|'
      if print_conf:
        out += 'cf:{:2d}/{:2d}/{:2d}/{:2d}|'.format(
               *[int(c.tolist() * 100) if not torch.isnan(c) else -1
                 for c in self.conf.values()])
    return out

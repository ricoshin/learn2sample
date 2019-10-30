from collections import OrderedDict

import torch
from utils.color import Color

import torch.nn.functional as F

# xxx = torch.tensor([1.2, 0.1, 0.9, 0.05, 0.5]).repeat(5, 1).log().cuda()
# target = torch.tensor([0, 1, 2, 3, 4]).cuda()
# mask_h = torch.tensor([1, 0, 0, 1, 0]).float().cuda()
# mask_s = torch.tensor([.99, .01, .01, .99, .01]).cuda()
# stand_logsoftmax = F.log_softmax(xxx, dim=1)
# stand_softmax = F.softmax(xxx, dim=1)
# stand_loss_h = F.cross_entropy(xxx, target, mask_h, reduction='none')
# stand_loss_s = F.cross_entropy(xxx, target, mask_s, reduction='none')


class ModelOutput(object):
  """A class for dynamically dealing with different types of output."""

  def __init__(self, params_name, dataset_name, n_classes, logits, labels,
               pairwise_dist, mask=None):

    self.params_name = params_name
    self.dataset_name = dataset_name
    self.n_classes = n_classes

    self.logits = logits
    self.labels = labels
    self.pairwise_dist = pairwise_dist

    self._mask = mask
    self.predicted = self.logits.argmax(dim=1)

    self.is_true = self.predicted == labels
    self.is_false = self.predicted != labels
    self.acc_sample = self.is_true.float()

    self._conf = None
    self._loss_m = None
    self._acc_m = None
    self._loss_s = None

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
  def loss_sample(self):
    return F.cross_entropy(self.logits, self.labels, reduction='none')

  @property
  def element_conf(self):
    return F.softmax(self.logits, dim=1)

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
      self._loss_m = self.weighted_cross_entropy(
        self.logits, self.labels, self.mask.squeeze(), dim=1, reduction='none')
    return self._loss_m.mean(dim=1, keepdim=True)

  @property
  def loss_masked_mean(self):
    """mask weighted mean of loss."""
    return self.loss_masked.sum() / self.mask.sum()

  @property
  def acc_masked(self):
    """masked elementwise accuracy."""
    if self._acc_m is None:
      weighted_log_softmax = self.weighted_log_softmax(
        self.logits, self.mask.squeeze(), dim=1, mask_numer=True)
      is_true = weighted_log_softmax.argmax(dim=1) == self.labels
      self._acc_m = is_true.float().view(self.n_classes, -1)
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

  def weighted_log_softmax(
    self, input, weight, dim, eps=1e-12, mask_numer=True):
    assert len(weight.size()) == 1
    assert weight.size(0) == input.size(dim)
    size = [1 if i != dim else weight.size(0)
            for i in range(len(input.size()))]
    weight = weight.view(size)
    x_shifted = input - input.max(dim=dim, keepdim=True)[0]
    log_numer = x_shifted + (weight + eps).log()
    log_denom = log_numer.exp().sum(dim=dim, keepdim=True).log()
    if not mask_numer:
      log_numer = x_shifted
    log_softmax = log_numer - log_denom
    return log_softmax

  def weighted_cross_entropy(
    self, input, target, weight, dim, reduction, eps=1e-12):
    log_dropmax_ = self.weighted_log_softmax(
      input, weight, dim, eps, mask_numer=False)
    xent = F.nll_loss(log_dropmax_, target, reduction='none')
    return xent.view(self.n_classes, -1) * weight.view(-1, 1)

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
    if self.dataset_name == 'Support':
      out += f'{self.loss.mean().tolist(): 5.2f}/'
      out += f'{self.acc.mean().tolist()*100:5.1f}%|'
      # if self._mask is not None:
      #   out += f'm.{self.loss_masked_mean.tolist(): 5.2f}/'
      #   out += f'm.{self.acc_masked_mean.tolist()*100:5.1f}%|'
    if self.dataset_name == 'Query':
      out += f'{Color.GREEN}{self.loss.mean().tolist():5.2f}{Color.END}/'
      out += f'{Color.RED}{self.acc.mean().tolist()*100:5.1f}{Color.END}%|'
      if print_conf:
        out += 'cf:{:2d}/{:2d}/{:2d}/{:2d}|'.format(
               *[int(c.tolist() * 100) if not torch.isnan(c) else -1
                 for c in self.conf.values()])
    return out

  @classmethod
  def as_merged_dict(cls, outputs):
    """merge outputs with nonoverlapping keys into a dictionary."""
    assert isinstance(outputs, (list, tuple))
    assert isinstance(outputs[0], cls)
    merged_dict = OrderedDict()
    for output in outputs:
      dict_ = output.as_dict()
      for key in dict_.keys():
        assert key not in merged_dict.keys()
      merged_dict.update(dict_)
    return merged_dict

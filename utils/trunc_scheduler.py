from torch._six import inf
from functools import partial


class Truncation(object):
  """Increase truncation length when the mean of outer loss has stopped
  decreasing and the variance of outer objective is lower than threshold.
  """

  def __init__(self, initial_value, mode='min', factor=1.5, patience=10,
               verbose=False, threshold=1e-4, threshold_mode='rel',
               cooldown=5, max_len=50, eps=1e-8):

    if factor <= 1.0:
      raise ValueError('Factor should be > 1.0.')
    self.len = initial_value
    self.factor = factor
    self.max_len = max_len
    self.patience = patience
    self.verbose = verbose
    self.cooldown = cooldown
    self.cooldown_counter = 0
    self.mode = mode
    self.threshold = threshold
    self.threshold_mode = threshold_mode
    self.best = None
    self.num_bad_epochs = None
    self.mode_worse = None  # the worse value for the chosen mode
    self.is_better = None
    self.eps = eps
    self.last_epoch = -1
    self._init_is_better(mode=mode, threshold=threshold,
                         threshold_mode=threshold_mode)
    self._reset()

  def _reset(self):
    """Resets num_bad_epochs counter and cooldown counter."""
    self.best = self.mode_worse
    self.cooldown_counter = 0
    self.num_bad_epochs = 0

  def step(self, metrics, epoch=None):
    # convert `metrics` to float, in case it's a zero-dim Tensor
    current = float(metrics)
    if epoch is None:
      epoch = self.last_epoch = self.last_epoch + 1
    self.last_epoch = epoch

    if self.is_better(current, self.best):
      self.best = current
      self.num_bad_epochs = 0
    else:
      self.num_bad_epochs += 1

    if self.in_cooldown:
      self.cooldown_counter -= 1
      self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

    if self.num_bad_epochs > self.patience:
      self._increase_len(epoch)
      self.cooldown_counter = self.cooldown
      self.num_bad_epochs = 0

  def _increase_len(self, epoch):
    old_len = float(self.len)
    new_len = min(int(old_len * self.factor), self.max_len)
    if new_len - old_len > self.eps:
      self.len = new_len
      if self.verbose:
        print('\n\n\nepoch {:3d}: increasing truncation length'
              ' to {:5d}.'.format(epoch, int(new_len)))

  @property
  def in_cooldown(self):
    return self.cooldown_counter > 0

  def _cmp(self, mode, threshold_mode, threshold, a, best):
    if mode == 'min' and threshold_mode == 'rel':
      rel_epsilon = 1. - threshold
      return a < best * rel_epsilon

    elif mode == 'min' and threshold_mode == 'abs':
      return a < best - threshold

    elif mode == 'max' and threshold_mode == 'rel':
      rel_epsilon = threshold + 1.
      return a > best * rel_epsilon

    else:  # mode == 'max' and epsilon_mode == 'abs':
      return a > best + threshold

  def _init_is_better(self, mode, threshold, threshold_mode):
    if mode not in {'min', 'max'}:
      raise ValueError('mode ' + mode + ' is unknown!')
    if threshold_mode not in {'rel', 'abs'}:
      raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

    if mode == 'min':
      self.mode_worse = inf
    else:  # mode == 'max':
      self.mode_worse = -inf

    self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

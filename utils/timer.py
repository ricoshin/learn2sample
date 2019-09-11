import time
from enum import Enum, unique


@unique
class WatchMode(Enum):
  NONE = 0
  GO_STOP = 1
  TOUCH = 2


class Walltime(object):
  def __init__(self):
    self._time = 0

  @property
  def time(self):
    return self._time

  @time.setter
  def time(self, new_time):
    isinstance(new_time, float)
    self._time = new_time

  def __add__(self, time):
    assert isinstance(time, float)
    self._time += time


class WalltimeShouter(object):
  def __init__(mode='interval'):
    self.watch = StopWatch(mode)

  def __enter__(self):
    self.watch.touch()

  def __exit__(self, type, value, trace_back):
    self.watch.touch(self.verbose)


class WalltimeChecker(object):
  def __init__(self, *args):
    if len(args) == 1 and args[0] is None:
      self.times = ()
      self.watch = None
    else:
      assert all([isinstance(arg, Walltime) for arg in args])
      self.times = args
      self.watch = StopWatch('interval')

  def __enter__(self):
    if self.times:
      self.watch.touch()
      return self.watch

  def __exit__(self, type, value, trace_back):
    if self.times:
      time = self.watch.touch()
      for i in range(len(self.times)):
        self.times[i].time += time


class StopWatch(object):
  def __init__(self, name='default', mode='interval'):
    assert mode in ['cumulative', 'interval']
    self.name = name
    self.initial = None
    self.cumulative = [0]
    self.interval = []
    self.mode = mode
    self._time = None

  @property
  def average(self):
    if len(interval) == 0:
      return 0
    return self.interval / len(self.interval)

  def go(self):
    if self.initial is not None:
      raise Exception(f'StopWatch[{self.name}] already started!')
    self.initial = time.time()
    return self

  def stop(self, mode=None, verbose=False):
    mode = self.mode if mode is None else mode
    assert mode in ['cumulative', 'interval']
    if self.initial is None:
      raise Exception(f'start StopWatch[{self.name}] first!')

    cumulative = time.time() - self.initial
    self.cumulative.append(cumulative)
    self.interval.append(self.cumulative[-1] - self.cumulative[-2])

    if mode == 'cumulative':
      out = self.cumulative[-1]
    elif mode == 'interval':
      out = self.interval[-1]
    else:
      out = None

    if verbose:
      print('\n' + self.format(out, mode) + '\n')

    return out

  def touch(self, mode=None, verbose=False):
    mode = self.mode if mode is None else mode
    assert mode in ['cumulative', 'interval']
    if self.initial is None:
      self.go()
      return 0.0
    return self.stop(mode, verbose)

  def format(self, sec, mode, hms=False):
    outstr = f'StopWatch[{self.name}/{mode}] {sec}'
    if hms:
      outstr += time.strftime("(%Hhrs %Mmins %Ssecs)", time.gmtime(seconds))
    return outstr

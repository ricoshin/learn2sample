import os
import numpy as np
import torch
import pickle
import pandas as pd


class ResultList(list):
  input_types = (list, tuple)
  def __init__(self, item):
    assert isinstance(item, ResultList.input_types)
    super().__init__(item)
    self._dtype = self._get_type(item)

  def __repr__(self):
    return f"ResultList({super().__repr__()})"

  @property
  def dtype(self):
    return self._dtype

  def numpy(self):
    return np.array(list(self))

  def sum(self):
    return np.sum(list(self))

  def mean(self):
    return np.mean(list(self))

  def append(self, item):
    if len(self) == 0:
      self._dtype = self._get_type(item)
    else:
      self._check_homogeneity(item)
    super().append(item)

  def _get_type(self, item):
    if isinstance(item, ResultList):
      self._dtype = item.dtype
    else:
      self._dtype = self._get_innermost_type(item)

  def _get_innermost_type(self, item):
    if isinstance(item, ResultList.input_types):
      return self._get_innermost_type(item[0]) if item else None
    else:
      return type(item)

  def _check_homogeneity(self, item):
    a = self.__getitem__(0)
    b = item
    assert type(a) == type(b)
    if isinstance(a, (int, float, str, np.generic)):
      pass
    elif isinstance(a, torch.Tensor):
      assert a.size() == b.size()
    elif isinstance(a, np.ndarray):
      assert np.shape(a) == np.shape(b)
    elif isinstance(a, (list, tuple, dict)):
      assert len(a) == len(b)
    else:
      raise TypeError(f'Unexpected type of a and b: {type(a)}')


class ResultPack(object):
  def __init__(self, name):
    self._data_frame = pd.DataFrame()

  def append(self, *args, **kwargs):
    kwargs = _maybe_factorize_list(dict(*args, **kwargs))
    dict_ = {}
    for name, value in kwargs.items():
      if isinstance(value, torch.Tensor):
        value = value.data.tolist()[0]
      if not isinstance(value, (int, float, str)):
        raise RuntimeError('Unexpected type!')
      if len(self._data_frame) == 0:
        dict_[name] = pd.Series(value, dtype=value.__class__.__name__)
      else:
        dict_[name] = pd.Series(value, dtype=self._data_frame.dtypes[name])
      self._data_frame.append(dict_, ignore_index=True)
    else:
      assert len(self._data_frame.columns) == len(kwargs)
      # self._data_frame.


      dtypes = [type_.__class__.__name__ for type_ in self.kwargs.values()]
    for k, v in kwargs:
      pd.Series()

  def _maybe_factorize_list(self, kwargs):
    """split lists (if exists in values) to seperate key-value pairs"""
    for name, value in kwargs.items():
      if isinstance(value, (list, tuple)):
        unique_kwargs = {name + "_" + str(i): l for i, l in enumerate(value)}
        kwargs.update(unique_kwargs)
        del kwargs[name]

  def _auto_type_series(self, kwargs):
    dict_ = {}
    get_series = lambda k, v: pd.Series({k: v}, dtype)
    for name, value in kwargs.items():
      if isinstance(value, (int, float, str)):
        pd.Series(kwargs, dtype=type)


class ResultDict(dict):
  _save_ext = '.pickle'

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def __repr__(self):
    return f"ResultDict({str(self.keys())})"

  def sub(self, keys):
    isinstance(keys, (list, tuple))
    return ResultDict({k: v for k, v in self.items() if k in keys})

  def getitem(self, key):
    return ResultDict({k: v[key] for k, v in self.items()})

  def update(self, *args, **kwargs):
    dict_ = dict(*args, **kwargs)
    for key, value in dict_.items():
      self.__setitem__(key, value)
    return self

  def append(self, *args, **kwargs):
    dict_ = dict(*args, **kwargs)
    for key, value in dict_.items():
      if isinstance(value, torch.Tensor):
        value = float(value.data.cpu().numpy())
      if key not in self:
        self.__setitem__(key, ResultList([value]))
      else:
        self.__getitem__(key).append(value)
    return self

  def numpy(self):
    return ResultDict({k: np.array(v) for k, v in self.items()})

  def w_postfix(self, postfix):
    assert isinstance(postfix, str)
    return ResultDict({k + '_' + postfix : v for k, v in self.items()})

  def mean(self, *args, **kwargs):
    return ResultDict({k: np.array(v).mean(*args, **kwargs)\
      for k, v in self.items()})

  def var(self, *args, **kwargs):
    return ResultDict({k: np.array(v).var(*args, **kwargs)\
      for k, v in self.items()})

  def save(self, name, save_dir, tag=''):
    if save_dir is None:
      return self
    tag = '_' + tag if tag else tag
    filename = os.path.join(save_dir, name + tag + ResultDict._save_ext)
    with open(filename, 'wb') as f:
      pickle.dump(dict(self), f)
    print(f'\nSaved test result as: {filename}')
    return self

  def save_as_csv(self, name, save_dir, tag='', trans_1d=False):
    if save_dir is None:
      return self
    tag = '_' + tag if tag else tag
    filename = os.path.join(save_dir, name + tag + '.csv')
    with open(filename, 'wb') as f:
      for k, v in dict(self).items():
        v = (v,) if trans_1d else v
        np.savetxt(f, v, delimiter=',', header=k)
    print(f'\nSaved test result as: {filename}')
    return self

  @property
  def shape(self):
    """return dictionary of shapes(tuple) corresponing to each keys."""
    return {k: v.shape for k, v in self.numpy().items()}

  @property
  def dtype(self):
    """return dictionary of types corresponing to each keys."""
    return {k: v.dtype for k, v in self.items()}

  @property
  def full_shape(self):
    return max([value for value in self.shape.values()])

  @classmethod
  def load(cls, name, load_dir):
    filename = os.path.join(load_dir, name + ResultDict._save_ext)
    if not os.path.exists(filename):
      return None
    with open(filename, 'rb') as f:
      loaded = cls(pickle.load(f))
    print(f'\nLoaded test result from: {filename}')
    return loaded

  @staticmethod
  def is_loadable(name, load_dir):
    filepath = os.path.join(load_dir, name + ResultDict._save_ext)
    return True if os.path.exists(filepath) and load_dir else False

  def data_frame(self, opt_name, dim_names):
    dict_ = self.numpy()
    #values = [array for array in results.values()]
    #shapes = [array.shape for array in dict_.values()]
    ndims = [len(shape) for shape in self.shape.values()]
    mask = [ndim == len(dim_names) for ndim in ndims]
    dict_ = {k: v for i, (k,v) in enumerate(dict_.items()) if mask[i]}

    index2values = {}
    for array in dict_.values():
      for index, value in np.ndenumerate(array):
        if index in index2values:
          index2values[index].append(value)
        else:
          index2values[index] = [value]

    data = []
    names = ['optimizer'] + dim_names + [name for name in dict_.keys()]
    for index, values in index2values.items():
      data.append([opt_name] + [id for id in index] + values)

    series = {}
    try:
      dtypes = [d.__class__.__name__ for d in data[0]]
    except:
      import pdb; pdb.set_trace()
    data = np.array(data)
    for i, name in enumerate(names):
      series.update({name: pd.Series(data[:, i], dtype=dtypes[i])})
    return pd.DataFrame(series)

"""
  Usage:

    ResultList(List) -> ResultDict(OrderedDict) -> ResultFrame(pd.DataFrame)

  Example:

    result_dict = ResultDict()
    result_dict.append(key_0=value_00, key_1=value_01), ...)
    result_dict.append(key_0=value_01, key_1=value_11), ...)

    >>> result_dict = ResultDict({
          key_0=ResultList([value_00, value_01, ...]),
          key_1=ResultList([value_10, value_11, ...]),
        })

    result_frame = ResultFrame()
    result_frmae = result_frame.append_tensors(result_dict.index_all(0))
    result_frmae = result_frame.append_tensors(result_dict.mean_all())

    >>> result_frame =
            key_0        |      key_1       |       ...
        ----------------------------------------------------
          tensor_00      |   tensor_10      |       ...
          tensor_0_mean  |   tensor_1_mean  |       ...
"""

import os
import pickle
from collections import OrderedDict

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


class ResultList(list):
  def __repr__(self):
    return 'ResultList(' + super(ResultList, self).__repr__() + ')'

  def __getitem__(self, key):
    item = super(ResultList, self).__getitem__(key)
    if isinstance(item, list):
      item = ResultList(item)
    return item

  def append(self, value):
    if isinstance(value, torch.Tensor):
      value = value.squeeze().tolist()
    super(ResultList, self).append(value)

  def mean(self, dim=None):
    mean_ = np.mean(list(self), axis=dim)
    if isinstance(mean_, list):
      mean_ = ResultList(mean_)
    return mean_

  def plot_heatmap(self, every_n_rows, annot):
    df = pd.DataFrame(list(self))
    df.index += 1
    if len(df.index) >= every_n_rows:
      df = df[df.index % every_n_rows == 0]
    heatmap = sns.heatmap(df, vmin=0, vmax=1, annot=annot, linewidth=0.5,
                          fmt="4.2f", cmap="YlGnBu")
    return heatmap

  def save_fig(self, file_name, save_path, i, every_n_rows=5, annot=False):
    if save_path is None:
      return
    file_name = file_name + '_' + str(i).zfill(4)
    file_path = os.path.join(save_path, file_name + '.png')
    if not os.path.exists(os.path.dirname(file_path)):
      os.makedirs(os.path.dirname(file_path))
    figure = self.plot_heatmap(every_n_rows, annot).get_figure()
    figure.savefig(file_path)
    plt.close(figure)

  def save_csv(self, file_name, save_path, i):
    if save_path is None:
      return
    file_name = file_name + '_' + str(i).zfill(4)
    file_path = os.path.join(save_path, file_name + '.csv')
    if not os.path.exists(os.path.dirname(file_path)):
      os.makedirs(os.path.dirname(file_path))
    pd.DataFrame(list(self)).to_csv(file_path, mode='w')
    print(f'Saved csv file: {file_path}')


class ResultDict(OrderedDict):
  def append(self, **kwargs):
    if len(self) == 0:
      self.update(OrderedDict({k: ResultList() for k in kwargs.keys()}))
    else:
      if not self.keys() == kwargs.keys():
        raise Exception(f'mismatched keys: {kwargs.keys()} and {self.keys()}')
    for key, value in kwargs.items():
      self.__getitem__(key).append(value)

  def index_all(self, key):
    """indexes every elements by the same key."""
    assert isinstance(key, int)
    return ResultDict({k: v[key] for k, v in self.items()})

  def mean_all(self, dim=None):
    return ResultDict(
      {k: v.mean(dim) if hasattr(v, 'mean') else v
          for k, v in self.items()})

  def get_items(self, keys):
    assert isinstance(keys, (list, tuple))
    return ResultDict({k: v for k, v in self.items() if k in keys})

  def save_csv(self, prefix, save_path, i):
    if save_path is None:
      return
    for k, v in self.items():
      v.save_csv(f'{prefix}_{k}', save_path, i)


class ResultFrame(pd.DataFrame):
  """Result recoder and plotter subclassing pd.DataFrame."""
  @property
  def model_names(self):
    return list(set([col.split('_')[0] for col in self.columns]))

  def append_dict(self, dict_, to_num=True):
    assert isinstance(dict_, ResultDict)
    if to_num:
      dict_ = {k: v.tolist() if isinstance(v, torch.Tensor) else v
               for k, v in dict_.items()}
    return ResultFrame(self.append(dict_, ignore_index=True))

  def group_fn(self, fn_name, group_name, cond):
    assert isinstance(group_name, str)
    assert callable(cond)
    group = self.groupby([group_name], sort=True)
    columns = [c for c in self.columns if cond(c)]
    return ResultFrame(getattr(group[columns], fn_name)())

  def group_max(self, column_name, group_name):
    return self.group_fn('max', group_name, lambda x: x == column_name)

  def group_min(self, column_name, group_name):
    return self.group_fn('min', group_name, lambda x: x == column_name)

  def get_best(self, name, min_or_max):
    columns = [col for col in self.columns
               if '_'.join(col.split('_')[1:]) == name]
    new_columns = [col.split('_')[0] for col in columns]
    best = self.groupby(['outer_step'], sort=True)[columns]
    best = getattr(best, min_or_max)()
    best.columns = new_columns
    return ResultFrame(best)

  def get_best_loss(self):
    columns = [col for col in self.columns
               if '_'.join(col.split('_')[1:]) == 'q_loss']
    new_columns = [col.split('_')[0] for col in columns]
    best_loss = self.groupby(['outer_step'], sort=True)[columns].min()
    best_loss.columns = new_columns
    return ResultFrame(best_loss)

  def get_best_acc(self):
    columns = [col for col in self.columns
               if '_'.join(col.split('_')[1:]) == 'q_acc']
    new_columns = [col.split('_')[0] for col in columns]
    best_acc = self.groupby(['outer_step'], sort=True)[columns].max()
    best_acc.columns = new_columns
    return ResultFrame(best_acc)

  def find(self, name, whitelist=[]):
    return ResultFrame(self[[col for col in self
                        if name in col or col in whitelist]])

  def save_final_lineplot(self, name, save_path=None):
    if save_path is None:
      return
    df = self.find(name=name, whitelist=['inner_step'])
    df = df.melt(id_vars='inner_step', var_name='model', value_name=name)
    df['model'] = [d.split('_')[0] for d in df['model']]
    plot = sns.lineplot(data=df, x='inner_step', y=name, hue='model')
    figure = plot.get_figure()
    file_path = os.path.join(save_path, 'plot_' + name + '.png')
    figure.savefig(file_path)
    plt.close(figure)
    print(f'Saved line plot: {file_path}')

  def save_mean_std(self, header, save_path=None):
    msg = ''
    for col in self.columns:
      msg += (f'({col}) mean: {self[col].mean():8.5f} / '
              f'std: {self[col].std():8.5f}\n')
    msg = header + '\n' + msg
    print(msg)
    if save_path:
      with open(os.path.join(save_path, 'final.txt'), 'a') as f:
        f.write(msg + '\n')

  def save_csv(self, file_name, save_path=None):
    if save_path is None:
      return
    file_path = os.path.join(save_path, file_name + '.csv')
    if not os.path.exists(os.path.dirname(file_path)):
      os.makedirs(os.path.dirname(file_path))
    self.to_csv(file_path, mode='w')
    print(f'Saved csv file: {file_path}')

  def save_fig(self, file_name, save_path=None):
    if save_path is None:
      return
    file_path = os.path.join(save_path, file_name + '.csv')
    if not os.path.exists(os.path.dirname(file_path)):
      os.makedirs(os.path.dirname(file_path))

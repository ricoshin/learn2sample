import os
import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


class MaskRecoder(object):
  """A class for tracking"""
  def __init__(self):
    self._masks = []

  def append(self, mask):
    assert isinstance(mask, torch.Tensor)
    self._masks.append(mask.squeeze().tolist())

  def plot(self, every_n_rows):
    masks = pd.DataFrame(self._masks)
    masks.index += 1
    if len(masks.index) >= every_n_rows:
      masks = masks[masks.index % every_n_rows == 0]
    heatmap = sns.heatmap(masks, vmin=0, vmax=1, annot=True, linewidth=0.5,
                          fmt="4.2f", cmap="YlGnBu")
    return heatmap

  def save_fig(self, file_name, save_path, every_n_rows=5):
    if save_path is None:
      return
    file_path = os.path.join(save_path, file_name + '.png')
    if not os.path.exists(os.path.dirname(file_path)):
      os.makedirs(os.path.dirname(file_path))
    figure = self.plot(every_n_rows).get_figure()
    figure.savefig(file_path)
    plt.close(figure)


class Result(pd.DataFrame):
  """Result recoder and plotter subclassing pd.DataFrame."""
  @property
  def model_names(self):
    return list(set([col.split('_')[0] for col in self.columns]))

  def append_tensors(self, dict_, to_num=True):
    assert isinstance(dict_, OrderedDict)
    if to_num:
      dict_ = {k: v.tolist() if isinstance(v, torch.Tensor) else v
               for k, v in dict_.items()}
    return Result(self.append(dict_, ignore_index=True))

  def group_fn(self, fn_name, group_name, cond):
    assert isinstance(group_name, str)
    assert callable(cond)
    group = self.groupby([group_name], sort=True)
    columns = [c for c in self.columns if cond(c)]
    return Result(getattr(group[columns], fn_name)())

  def group_max(self, column_name, group_name):
    return self.group_fn('max', group_name, lambda x: x == column_name)

  def group_min(self, column_name, group_name):
    return self.group_fn('min', group_name, lambda x: x == column_name)

  def get_best_loss(self):
    columns = [col for col in self.columns
               if '_'.join(col.split('_')[1:]) == 'loss_q_m']
    new_columns = [col.split('_')[0] for col in columns]
    best_loss = self.groupby(['outer_step'], sort=True)[columns].min()
    best_loss.columns = new_columns
    return Result(best_loss)

  def get_best_acc(self):
    columns = [col for col in self.columns
               if '_'.join(col.split('_')[1:]) == 'acc_q_m']
    new_columns = [col.split('_')[0] for col in columns]
    best_acc = self.groupby(['outer_step'], sort=True)[columns].max()
    best_acc.columns = new_columns
    return Result(best_acc)

  def find(self, name, whitelist=[]):
    return Result(self[[col for col in self
                        if name in col or col in whitelist]])

  def save_final_lineplot(self, name, save_path):
    df = self.find(name=name, whitelist=['inner_step'])
    df = df.melt(id_vars='inner_step', var_name='model', value_name=name)
    df['model'] = [d.split('_')[0] for d in df['model']]
    plot = sns.lineplot(data=df, x='inner_step', y=name, hue='model')
    figure = plot.get_figure()
    file_path = os.path.join(save_path, 'plot_' + name + '.png')
    figure.savefig(file_path)
    plt.close(figure)

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

  def save_to_csv(self, file_name, save_path=None):
    if save_path is None:
      return
    file_path = os.path.join(save_path, file_name + '.csv')
    if not os.path.exists(os.path.dirname(file_path)):
      os.makedirs(os.path.dirname(file_path))
    self.to_csv(file_path, mode='w')
    print(f'Saved csv file: {file_path}')

  def save_to_fig(self, file_name, save_path=None):
    if save_path is None:
      return
    file_path = os.path.join(save_path, file_name + '.csv')
    if not os.path.exists(os.path.dirname(file_path)):
      os.makedirs(os.path.dirname(file_path))

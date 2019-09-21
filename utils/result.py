import os
import numpy as np
import torch
import pickle
import pandas as pd
from collections import OrderedDict


class Result(pd.DataFrame):
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

  def get_best_acc(self):
    columns = [col for col in self.columns
               if '_'.join(col.split('_')[1:]) == 'acc_q_m']
    new_columns = [col.split('_')[0] for col in columns]
    best_acc = self.groupby(['outer_step'], sort=True)[columns].max()
    best_acc.columns = new_columns
    return Result(best_acc)

  def get_best_loss(self):
    columns = [col for col in self.columns
               if '_'.join(col.split('_')[1:]) == 'loss_q_m']
    new_columns = [col.split('_')[0] for col in columns]
    best_loss = self.groupby(['outer_step'], sort=True)[columns].min()
    best_loss.columns = new_columns
    return Result(best_loss)

  def print_mean_std(self, header, save_path=None):
    msg = ''
    for col in self.columns:
      msg += (f'({col}) mean: {self[col].mean():8.5f} / '
             f'std: {self[col].std():8.5f}\n')
    msg = header + '\n' + msg
    print(msg)
    if save_path:
      with open(os.path.join(save_path, 'final.txt'), 'a') as f:
        f.write(msg + '\n')

  def _get_ready_dir(self, file_path):
    dir_names = os.path.dirname(file_path)
    if not os.path.exists(dir_names):
      os.makedirs(dir_names)

  def save_to_csv(self, file_name, save_path):
    file_path = os.path.join(save_path, file_name + '.csv')
    self._get_ready_dir(file_path)
    self.to_csv(file_path, mode='w')

  def save_to_fig(self, filename, path):
    file_path = os.path.join(save_path, file_name + '.csv')
    self._get_ready_dir(file_path)

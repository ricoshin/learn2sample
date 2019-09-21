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

  def _get_max(self, group, cond_fn):
    assert isinstance(group, str)
    assert callable(cond_fn)
    group = self.groupby([group], sort=True)
    columns = [c for c in self.columns if cond_fn(c)]
    return Result(group[columns].max())

  def get_max(self, col, group):
    return self._get_max(group, lambda x: x == col)

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
    

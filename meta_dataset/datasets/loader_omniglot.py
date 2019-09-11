import itertools
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datasets.datasets import IterDataLoader
from datasets.imagenet import ImageNet
from models.meprop import SBLinear, UnifiedSBLinear
from models.model_helpers import ParamsFlattener
from torch.utils import data
from torchvision import datasets, transforms
from utils import utils

from datasets.omniglot_nshot import OmniglotNShot

C = utils.getCudaManager('default')


class Loader(object):
  """For compatibility with other datasets. replaces datasets.IterDataLoader.
  """
  def __init__(self, dataset):
    self.dataset = dataset

  def __len__(self):
    return 1

  @property
  def iterator(self):
    return [self.dataset]

  def load(self):
    return self.dataset

  @classmethod
  def from_dataset(cls, dataset, batch_size, rand_with_replace):
    return cls(dataset)


class OmniglotData:
  def __init__(self, n_way=10, k_shot=10, k_query=10, imgsz=64):
    path = 'datasets/db/omniglot'
    print(f'Dataset path: {os.path.abspath(path)}')
    self.data = OmniglotNShot(path, batchsz=1, n_way=n_way, k_shot=k_shot,
      k_query=k_query, imgsz=imgsz)

  def inner_split(self, x_spt, y_spt, x_qry, y_qry):
    x_spt = torch.from_numpy(x_spt[0])
    x_qry = torch.from_numpy(x_qry[0])
    y_spt = torch.from_numpy(y_spt[0])
    y_qry = torch.from_numpy(y_qry[0])
    return {
      'in_train': Loader((x_spt, y_spt)), 'in_test':Loader((x_qry, y_qry))}

  def sample_meta_train(self):
    return self.inner_split(*self.data.next('train'))

  def sample_meta_valid(self):
    return self.inner_split(*self.data.next('test'))

  def sample_meta_test(self):
    return self.inner_split(*self.data.next('test'))



  def _get_wrapped_dataloader(self, dataset, batch_size):
    sampler = data.sampler.RandomSampler(dataset, replacement=True)
    loader = data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return IterDataLoader(loader)

  def pseudo_sample(self):
    """return pseudo sample for checking activation size."""
    return (torch.zeros(1, 1, 28, 28), None)

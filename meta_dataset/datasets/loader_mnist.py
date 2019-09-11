import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.model_helpers import ParamsFlattener
from datasets.datasets import IterDataLoader

from models.meprop import SBLinear, UnifiedSBLinear
from torch.utils import data
from torchvision import datasets
from utils import utils

C = utils.getCudaManager('default')


class MNISTData:
  """Current data scheme is as follows:
    - meta-train data(30K)
      - inner-train data(15K)
      - inner-test data(15K)
    - meta-valid data(20K)
      - inner-train data(15K)
      - inner-test data(5K)
    - meta-test data(20K)
      - inner-train data(15K)
      - inner-test data(5K)
  """

  def __init__(self, batch_size=128, fixed=False):
    self.fixed = fixed
    self.batch_size = batch_size
    train_data = datasets.MNIST('./mnist', train=True, download=True,
                                transform=torchvision.transforms.ToTensor())
    test_data = datasets.MNIST('./mnist', train=False, download=True,
                               transform=torchvision.transforms.ToTensor())
    self.m_train_d, self.m_valid_d, self.m_test_d = self._meta_data_split(
      train_data, test_data)
    self.meta_train = self.meta_valid = self.meta_test = {}

  def sample_meta_train(self, ratio=0.5, fixed=None):
    fixed = fixed if fixed is not None else self.fixed
    if fixed and self.meta_train:
      return self.meta_train
    inner_train, inner_test = self._random_split(self.m_train_d, ratio)
    self.meta_train['in_train'] = IterDataLoader.from_dataset(
      inner_train, self.batch_size)
    self.meta_train['in_test'] = IterDataLoader.from_dataset(
      inner_test, self.batch_size)
    return self.meta_train

  def sample_meta_valid(self, ratio=0.75, fixed=None):
    fixed = fixed if fixed is not None else self.fixed
    if fixed and self.meta_valid:
      return self.meta_valid
    inner_train, inner_test = self._random_split(self.m_valid_d, ratio)
    self.meta_valid['in_train'] = IterDataLoader.from_dataset(
      inner_train, self.batch_size)
    self.meta_valid['in_test'] = IterDataLoader.from_dataset(
      inner_test, self.batch_size)
    return self.meta_valid

  def sample_meta_test(self, ratio=0.75, fixed=None):
    fixed = fixed if fixed is not None else self.fixed
    if fixed and self.meta_test:
      return self.meta_test
    inner_train, inner_test = self._random_split(self.m_test_d, ratio)
    self.meta_test['in_train'] = IterDataLoader.from_dataset(
      inner_train, self.batch_size)
    self.meta_test['in_test'] = IterDataLoader.from_dataset(
      inner_test, self.batch_size)
    return self.meta_test

  def _meta_data_split(self, train_data, test_data):
    data_ = data.dataset.ConcatDataset([train_data, test_data])
    meta_train, meta_valid_test = self._fixed_split(data_, 30/70)
    meta_valid, meta_test = self._fixed_split(meta_valid_test, 20/40)
    return meta_train, meta_valid, meta_test

  def _random_split(self, dataset, ratio=0.5):
    assert isinstance(dataset, data.dataset.Dataset)
    n_total = len(dataset)
    n_a = int(n_total * ratio)
    n_b = n_total - n_a
    data_a, data_b = data.random_split(dataset, [n_a, n_b])
    return data_a, data_b

  def _fixed_split(self, dataset, ratio=0.5):
    assert isinstance(dataset, data.dataset.Dataset)
    n_total = len(dataset)
    thres = int(n_total * ratio)
    id_a = range(len(dataset))[:thres]
    id_b = range(len(dataset))[thres:]
    data_a = data.Subset(dataset, id_a)
    data_b = data.Subset(dataset, id_b)
    return data_a, data_b

  # def _fixed_split(self, dataset, ratio):
  #   import pdb; pdb.set_trace()
  #   data_a = dataset[int(len(dataset) * ratio):]
  #   data_b = dataset[:int(len(dataset) * ratio)]
  #   return data_a, data_b

  def _get_wrapped_dataloader(self, dataset, batch_size):
    sampler = data.sampler.RandomSampler(dataset, replacement=True)
    loader = data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return IterDataLoader(loader)

  def pseudo_sample(self):
    """return pseudo sample for checking activation size."""
    return (torch.zeros(1, 1, 28, 28), None)

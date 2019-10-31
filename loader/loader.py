from collections import OrderedDict

import torch
from loader.episode import Episode
from loader.metadata import Metadata
from torch.utils import data
from utils import utils

C = utils.getCudaManager('default')


class DataFromMetadata(data.Dataset):
  def __init__(self, meta):
    assert isinstance(meta, Metadata)
    self.meta = meta

  def __len__(self):
    return sum([v for v in self.meta.idx_to_len.values()])

  def __getitem__(self, idx):
    class_idx, sample_idx = self.meta.idx_uni_to_bi(idx)
    filename = self.meta[class_idx][sample_idx][0]
    with open(filename, 'rb') as f:
      img = torch.load(f)
    label = torch.tensor(class_idx)
    ids = torch.tensor(self.meta.abs_idx[class_idx])
    return (img, label, ids)


class ClassBalancedSampler(data.Sampler):
  def __init__(self, meta, batch_size=None, samples_per_class=None):
    assert isinstance(meta, Metadata)
    self.meta = meta
    if not ((batch_size is None) ^ (samples_per_class is None)):
      raise ValueError("batch_size and samples_per_class are exclusive.")
    if batch_size is not None:
      self.samples_per_class = batch_size // len(meta)
    else:
      self.samples_per_class = samples_per_class

  def __iter__(self):
    while True:
      batch = []
      for class_idx, samples in self.meta.idx_to_samples.items():
        class_idx = self.meta.rel_idx[class_idx]
        batch.extend([self.meta.idx_bi_to_uni(class_idx, sample_idx)
                      for sample_idx in torch.randint(
            len(samples), (self.samples_per_class,))])
      yield batch


class EpisodeIterator(object):
  def __init__(self, support, query, split_ratio, resample_every_iteration,
               inner_steps, batch_size=None, samples_per_class=None,
               num_workers=8, pin_memory=True):
    assert isinstance(support, Metadata)
    assert isinstance(query, Metadata)
    if not ((batch_size is None) ^ (samples_per_class is None)):
      raise ValueError("batch_size and samples_per_class are exclusive.")
    self.support = support
    self.query = query
    self.split_ratio = split_ratio
    self.resample_every_iteration = resample_every_iteration
    self.inner_steps = inner_steps
    self.batch_size = batch_size
    self.samples_per_class = samples_per_class
    self.num_workers = num_workers
    self.pin_memory = pin_memory
    self._loaders = None

  @property
  def loaders(self):
    if self._loaders is None:
      raise Exception('EpisodeIterator.sample_episode() has to be called '
                      'to initialize loaders.')
    return self._loaders

  def sample_episode(self):
    ss, sq = self.support.split_instance(self.split_ratio)
    qs, qq = self.query.split_instance(self.split_ratio)
    self._loaders = self.get_loaders([ss, sq, qs, qq])
    return self

  def get_loader(self, metadata):
    dataset = DataFromMetadata(metadata)
    batch_sampler = ClassBalancedSampler(
        metadata, self.batch_size, self.samples_per_class)
    return iter(data.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=self.num_workers,
        pin_memory=self.pin_memory
    ))

  def get_loaders(self, multi_metadata):
    assert isinstance(multi_metadata, (list, tuple))
    assert isinstance(multi_metadata[0], Metadata)
    return [self.get_loader(metadata) for metadata in multi_metadata]

  def __iter__(self):
    if self.resample_every_iteration:
      self.sample_episode()
    for _ in range(self.inner_steps):
      ss, sq, qs, qq = [loader.next() for loader in self.loaders]
      meta_s = Episode.from_tensors(ss, sq, 'Support')
      meta_q = Episode.from_tensors(qs, qq, 'Query')
      yield meta_s, meta_q

from collections import OrderedDict

from numpy.random import randint
import torch
from loader.episode import Episode, Dataset
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


class RandomBatchSampler(data.Sampler):
  """Retain the same sampled classes(but different instances) until
     resample_classes() is called. Can be used for metric-based learning while
     following real data distribution unlike ClassBalancedBatchSampler.
  """
  def __init__(self, meta, batch_size):
    assert isinstance(meta, Metadata)
    self.meta = meta
    self.batch_size = batch_size

  @classmethod
  def get(self, batch_size):
    return lambda meta: cls(meta, batch_size)

  def resample_classes(self):
    self._memory = []

  def __iter__(self):
    self._memory = []
    while True:
      batch = []
      if not self._memory:
        n_classes = len(self.meta)
        max_n_samples = sum([v for v in self.meta.idx_to_len.values()])
        sampled_idx = randint(0, max_n_samples, self.batch_size)
        for i in range(self.batch_size):
          batch.append(sampled_idx[i])
          self._memory.append(self.meta.idx_uni_to_bi(sampled_idx[i]))
      else:
        for class_idx, sample_idx in self._memory:
          max_n_samples = self.meta.idx_to_len[class_idx]
          class_idx = self.meta.rel_idx[class_idx]
          batch.extend([self.meta.idx_bi_to_uni(class_idx, i)
                       for i in randint(0, max_n_samples, len(sample_idx))])
      yield batch


class BalancedBatchSampler(data.Sampler):
  def __init__(self, meta, class_size, sample_size):
    assert isinstance(meta, Metadata)
    self.meta = meta
    self.class_size = class_size
    self.sample_size = sample_size

  @classmethod
  def get(self, class_size, sample_size):
    return lambda meta: cls(meta, class_size, sample_size)

  def resample_classes(self):
    self._memory = None

  def __iter__(self):
    while True:
      batch = []
      if not self._memory:
        meta_sampled = self.meta.sample_classes(self.class_size)
        self._memory = meta_sampled
      else:
        meta_sampled = self._memory
      for class_idx, samples in meta_sampled.idx_to_samples.items():
        class_idx = self.meta.rel_idx[class_idx]
        max_n_samples = len(samples)
        batch.extend(
          [self.meta.idx_bi_to_uni(class_idx, sample_idx)
           for sample_idx in randint(0, max_n_samples, self.sample_size)])
      yield batch


class MetaEpisodeIterator(object):
  def __init__(self, meta_support, meta_query, batch_sampler, inner_steps,
               split_support_query, sample_split_ratio=None, num_workers=8,
               pin_memory=True):
    assert isinstance(meta_support, Metadata)
    assert isinstance(meta_query, Metadata)
    assert callable(batch_sampler)
    self.meta_s = meta_support
    self.meta_q = meta_query
    self.meta_s_sub = meta_support
    self.meta_q_sub = meta_query
    self.batch_sampler = batch_sampler
    self.inner_steps = inner_steps
    self.sample_split_ratio = sample_split_ratio
    self.num_workers = num_workers
    self.pin_memory = pin_memory
    self._loaders = None
    self.next_batch = self.get_batch_fn()
    # self._meta_s_resampled = True
    # self._meta_q_resampled = True
    # self.sample_meta_s_classes_()

  @property
  def loaders(self):
    if self._loaders is None:
      raise Exception('EpisodeIterator.sample_episode() has to be called '
                      'to initialize loaders.')
    return self._loaders

  # def sample_meta_s_classes_(self):
  #   self.meta_s_sub = self.meta_s.sample_classes(self.classes_per_episode)
  #   self._meta_s_resampled = True
  #
  # def sample_meta_q_classes_(self):
  #   self.meta_q_sub = self.meta_q.sample_classes(self.classes_per_episode)
  #   self._meta_q_resampled = True
  #
  # def _split_meta_s_instances(self):
  #   return self.meta_s_sub.split_instances(self.sample_split_ratio)
  #
  # def _split_meta_q_instances(self):
  #   return self.meta_q_sub.split_instances(self.sample_split_ratio)

  def _get_loader(self, metadata, batch_sampler):
    assert isinstance(metadata, Metadata)
    assert callable(batch_sampler)
    return iter(data.DataLoader(
        dataset=DataFromMetadata(metadata),
        batch_sampler=batch_sampler(metadata),
        num_workers=self.num_workers,
        pin_memory=self.pin_memory
    ))

  def _get_loaders(self, meta_list, batch_sampler):
    """All the elements of metadata have to incorporate the same classes
        in it to share the same batch sampler.
    """
    assert isinstance(meta_list, (list, tuple))
    assert isinstance(meta_list[0], Metadata)
    assert all([meta_list[0] == meta_list[i] for i in len(meta_list)])
    return [self.get_loader(metadata, batch_sampler)
            for metadata in meta_list]

  def get_next_batch_fn(self):
    if self.sample_split_ratio is None:
      meta_s_loader = self._get_loader(self.meta_s, self.batch_sampler)
      meta_q_loader = self._get_loader(self.meta_q, self.batch_sampler)
      def next_batch():
        ss, sq = meta_s_loader.next(), meta_s_loader.next()
        qs, qq = meta_q_loader.next(), meta_q_loader.next()
        meta_s_loader.batch_sampler.resample_classes()
        meta_q_loader.batch_sampler.resample_classes()
        return ss, sq, qs, qq
    else:
      # checkpoint
      ss, sq = self.meta_s.split_instances(self.sample_split_ratio)
      qs, qq = self.meta_q.split_instances(self.sample_split_ratio)
      ss, sq = self._get_loaders([ss, sq], self.batch_sampler)
      qs, qq = self._get_loaders([qs, qq], self.batch_sampler)
      def next_batch(resample_support):
        if resample_support:
          ss.batch_sampler.resample_classes()
          sq.batch_sampler.resample_classes()
        ss, sq = [loader.next() for loader in meta_s_loaders]
        qs, qq = [loader.next() for loader in meta_q_loaders]
        return ss, sq, qs, qq
      return next_batch




  def get_episode(self):
    if self.do_sample:
      self.batch_sampler.resample()
    if self.sample_split_ratio is None:
      meta_s_loader = self._get_loader(self.meta_s, self.batch_sampler)
      meta_q_loader = self._get_loader(self.meta_q, self.batch_sampler)
    else:



    # reload support loader (every unrolls)
    if self._meta_s_resampled:
      self._meta_s_resampled = False
      if self.split_support_query:
        meta_s_loaders = self.get_loaders(self._split_meta_s_instances())
      else:
        meta_s_loader = self.get_loader(self.meta_s_sub)
    # reload query loader (very first step only)
    if self._meta_q_resampled:
      self._meta_q_resampled = False
      if self.split_support_query:
        meta_q_loaders = self.get_loaders(self._split_meta_q_instances())
      else:
        meta_q_loader = self.get_loader(self.meta_q_sub)
    # for different batch samplers
    if self.split_support_query:
      #
      ss, sq = [loader.next() for loader in meta_s_loaders]
      qs, qq = [loader.next() for loader in meta_q_loaders]
    else:
      meta_s_loader.resample()
      meta_q_loader.resample()
      ss, sq = meta_s_loader.next(), meta_s_loader.next()
      qs, qq = meta_s_loader.next(), meta_s_loader.next()

  def __call__(self, do_sample):
    """can determine whether to resample classes at every iteration."""
    if do_sample and not self._meta_s_resampled:
      self.sample_meta_s_classes_()
      print('Resampled classes for meta support set.')
    return self

  def __iter__(self):
    for k in range(1, self.inner_steps):
      ss, sq, qs, qq = self.next_batch()
      # wrap tensors with Episode class
      meta_s = Episode.from_tensors(ss, sq, 'Support')
      meta_q = Episode.from_tensors(qs, qq, 'Query')
      import pdb; pdb.set_trace()

      yield k, (meta_s, meta_q)

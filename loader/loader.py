import random
from collections import OrderedDict

import torch
from loader import metadata
from loader.episode import Dataset, Episode
from numpy.random import randint
from torch.utils import data
from utils import utils

C = utils.getCudaManager('default')


class DataFromMetadata(data.Dataset):
  """Convert Metadata to torch.data.Dataset."""

  def __init__(self, meta):
    assert isinstance(meta, metadata.Metadata)
    self.meta = meta

  def __len__(self):
    return sum([v for v in self.meta.idx_to_len.values()])

  def __getitem__(self, idx):
    class_idx, sample_idx = self.meta.idx_uni_to_bi(idx)
    filename = self.meta[class_idx][sample_idx][0]
    with open(filename, 'rb') as f:
      img = torch.load(f)
    rel_idx = class_idx
    abs_idx = self.meta.abs_idx[class_idx]
    return (img, rel_idx, abs_idx)


class RandomBatchSampler(data.Sampler):
  """Can be used for metric-based learning while following real data
     distribution unlike ClassBalancedBatchSampler.
  """

  def __init__(self, meta, batch_size, n_repeat_classes=1, replacement=False,
               drop_last=False):
    assert isinstance(meta, metadata.Metadata)
    assert isinstance(n_repeat_classes, int) and n_repeat_classes >= 1
    self.meta = meta
    self.batch_size = batch_size
    self.n_repeat_classes = n_repeat_classes
    self.replacement = replacement
    self.indices = list(range(sum([v for v in self.meta.idx_to_len.values()])))

  def sampled_idx(self):
    if self.replacement:
      batch = []
      for idx in random.sample(self.indices, len(self.indices)):
        batch.append(idx)
        if len(batch) == self.batch_size:
          yield batch
          batch = []
      if len(batch) > 0 and not self.drop_last:
        yield batch
    else:
      while True:
        yield random.sample(self.indices, self.batch_size)

  def __iter__(self):
    _n_repeated = 0
    _to_repeat = []
    for idx in self.sampled_idx():
      batch = []
      if _n_repeated == 0:
        n_classes = len(self.meta)
        for i in range(self.batch_size):
          batch.append(idx[i])
          _to_repeat.append(self.meta.idx_uni_to_bi(idx[i]))
      else:
        for class_idx, sample_idx in _to_repeat:
          max_n_samples = self.meta.idx_to_len[self.meta.abs_idx[class_idx]]
          batch.extend([self.meta.idx_bi_to_uni(class_idx, i)
                        for i in randint(0, max_n_samples, 1)])
      _n_repeated += 1
      if _n_repeated == self.n_repeat_classes:
        _n_repeated = 0
        _to_repeat = []
      # else:
      #   continue
      yield batch


class BalancedBatchSampler(data.Sampler):
  def __init__(self, meta, class_size, sample_size, n_repeat_classes=1,
               replacement=False, drop_last=False):
    assert isinstance(meta, metadata.Metadata)
    assert isinstance(n_repeat_classes, int) and n_repeat_classes >= 1
    self.meta = meta
    self.class_size = class_size
    self.sample_size = sample_size
    self.n_repeat_classes = n_repeat_classes
    self.replacement = replacement
    self.drop_last = drop_last
    self.cls_indices = list(range(len(self.meta)))

  def sampled_cls_idx(self):
    if self.replacement:
      batch = []
      for cls_idx in random.sample(self.cls_indices, self.class_size):
        batch.append(cls_idx)
        if len(batch) == self.class_size:
          yield batch
          batch = []
      if len(batch) > 0 and not self.drop_last:
        yield batch
    else:
      while True:
        yield random.sample(self.cls_indices, self.class_size)

  def __iter__(self):
    _n_repeated = 0
    _to_repeat = []
    for cls_idx in self.sampled_cls_idx():
      batch = []
      if _n_repeated == 0:
        meta_sampled = self.meta.select_classes(cls_idx)
        if self.n_repeat_classes > 1:
          _to_repeat = meta_sampled
      elif _n_repeated <= self.n_repeat_classes:
        meta_sampled = _to_repeat
      for class_idx, samples in meta_sampled.idx_to_samples.items():
        class_idx = self.meta.rel_idx[class_idx]
        max_n_samples = len(samples)
        batch.extend(
            [self.meta.idx_bi_to_uni(class_idx, sample_idx)
             for sample_idx in randint(0, max_n_samples, self.sample_size)])
      _n_repeated += 1
      if _n_repeated == self.n_repeat_classes:
        _n_repeated = 0
        _to_repeat = []
      # else:
      #   continue
      yield batch


class LoaderConfig(object):
  def __init__(self, batch_size=None, class_size=None, sample_size=None,
               class_balanced=False, sort_in_batch=True, num_workers=2,
               pin_memory=False, replacement=False, drop_last=False,
               gpu_id='cpu'):
    """(batch_size) and (class_size, sample_size) are exclusive."""
    if batch_size is None:
      assert (class_size is not None) and (sample_size is not None)
    else:
      assert (class_size is None) and (sample_size is None)
    self.sort_in_batch = sort_in_batch
    self.num_workers = num_workers
    self.pin_memory = pin_memory
    self.batch_size = batch_size
    self.class_size = class_size
    self.sample_size = sample_size
    self.replacement = replacement
    self.drop_last = drop_last
    self.device = 'cpu'
    self.non_blocking = False

  def to(self, device, non_blocking=False):
    self.device = device
    self.non_blocking = non_blocking
    return self

  def _stack_batch(self, batch):
    out = None
    if torch.utils.data.get_worker_info() is not None:
      elem = batch[0]
      elem_type = type(elem)
      numel = sum([x.numel() for x in batch])
      storage = elem.storage()._new_shared(numel)
      out = elem.new(storage)
    return torch.stack(batch, 0, out=out)

  def collate_fn(self, batch):
    if self.sort_in_batch:
      # sort elements in batch by relative class labels
      batch = sorted(batch, key=lambda batch: batch[1])
    imgs = self._stack_batch([b[0] for b in batch])
    labels = self._stack_batch([torch.tensor(b[1]) for b in batch])
    ids = self._stack_batch([torch.tensor(b[2]) for b in batch])
    return imgs, labels, ids

  def _get_batch_sampler(self, metadata, n_repeat_classes):
    if self.batch_size:
      batch_sampler = RandomBatchSampler(
          meta=metadata,
          batch_size=self.batch_size,
          n_repeat_classes=n_repeat_classes,
          replacement=self.replacement,
          drop_last=self.drop_last,
      )
    else:
      batch_sampler = BalancedBatchSampler(
          meta=metadata,
          class_size=self.class_size,
          sample_size=self.sample_size,
          n_repeat_classes=n_repeat_classes,
          replacement=self.replacement,
          drop_last=self.drop_last,
      )
    return batch_sampler

  def get_dataset_loader(self, metadata, name):
    batch_sampler = self._get_batch_sampler(metadata, 1)
    _loader = iter(data.DataLoader(
        dataset=DataFromMetadata(metadata),
        batch_sampler=batch_sampler,
        collate_fn=self.collate_fn,
        num_workers=self.num_workers,
        pin_memory=self.pin_memory,
    ))

    def loader():
      return Dataset(*_loader.next(), name).to(
          device=self.device, non_blocking=self.non_blocking)
    return loader

  def get_episode_loader(self, metadata, name):
    batch_sampler = self._get_batch_sampler(metadata, 2)
    _loader = iter(data.DataLoader(
        dataset=DataFromMetadata(metadata),
        batch_sampler=batch_sampler,
        collate_fn=self.collate_fn,
        num_workers=self.num_workers,
        pin_memory=self.pin_memory,
    ))

    def loader():
      s = Dataset(*_loader.next(), 'Support')
      q = Dataset(*_loader.next(), 'Query')
      # import pdb; pdb.set_trace()
      return Episode(s, q, len(s.classwise), name).to(
          device=self.device, non_blocking=self.non_blocking)
    return loader

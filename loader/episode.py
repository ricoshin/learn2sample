import os
import pdb
import random
import sys
from collections import Counter
from collections.abc import Iterable

import gin
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch


class DatasetClassIndexer(object):
  def __init__(self, dataset):
    assert isinstance(dataset, Dataset)
    self.dataset = dataset

  def __len__(self):
    return self.dataset.n_classes

  def __repr__(self):
    return __class__.__name__ + f' for \n{self.dataset}'

  def __getitem__(self, key):
    """Class indexing"""
    mask = 0
    class_idx = list(self.dataset.n_samples.keys())
    if isinstance(key, torch.Tensor) and not key.dim() == 0:
      key = key.tolist()

    if isinstance(key, Iterable):  # list
      selcted_class_idx = [class_idx[k] for k in key]
    else:  # int, slice
      selcted_class_idx = class_idx[key]

    if not isinstance(selcted_class_idx, Iterable):
      selcted_class_idx = [selcted_class_idx]  # int

    for id in selcted_class_idx:
      mask += self.dataset.labels == id
    imgs = self.dataset.imgs[mask]
    labels = self.dataset.labels[mask]
    ids = self.dataset.ids[mask]
    return Dataset(imgs, labels, ids, self.dataset.name)

  def masked_select(self, mask):
    assert isinstance(mask, torch.Tensor)
    idx = mask.squeeze().nonzero().view(-1)
    # assert len(idx) == len(self)
    if len(idx) == 0:
      return None
    else:
      return self.__getitem__(idx)


class Dataset(object):
  """Dataset class that should contain images, labels, and ids.
  Support set or query set can be a proper candidate of Dataset."""

  def __init__(self, imgs, labels, ids, name):
    assert all([isinstance(t, torch.Tensor) for t in [imgs, labels, ids]])
    assert name in ['Support', 'Query']
    self.imgs = imgs
    self.labels = labels
    self.ids = ids
    self.name = name
    class_counter = Counter(self.labels.tolist())
    self.n_samples = {i: n for i, n in sorted(class_counter.items())}
    self.n_classes = len(self.n_samples.keys())
    self.classwise = DatasetClassIndexer(self)  # support classwise operation

  def __repr__(self):
    return __class__.__name__ + f'(labels={self.labels}, name={self.name})'

  def __len__(self):
    assert len(self.labels) == len(self.ids)
    return len(self.labels)

  def __iter__(self):
    return iter([self.imgs, self.labels, self.ids])

  def __getitem__(self, key):
    """instacne indexing"""
    imgs = self.imgs[key].view(-1)
    labels = self.labels[key].view(-1)
    ids = self.ids[key].view(-1)
    # view() to avoid 0-dim when choosing single instance
    return Dataset(imgs, labels, ids, self.name)

  def new_named(self, name):
    self.name = name
    return self

  def masked_select(self, mask):
    assert isinstance(mask, torch.Tensor)
    idx = mask.squeeze().nonzero().view(-1)
    # assert len(idx) == len(self)
    if len(idx) == 0:
      return None
    else:
      return self.__getitem__(idx)

  def offset_indices(self, offset_labels, offset_ids):
    labels = self.labels + offset_labels
    ids = self.ids + offset_ids
    return Dataset(self.imgs, labels, ids, self.name)

  def get_view_classwise_fn(self):
    def view_classwise_fn(x):
      # assert x.size(0) == self.n_classes * self.n_samples
      rest_dims = [x.size(i) for i in range(1, len(x.shape))]
      return x.view(self.n_classes, -1, *rest_dims)
    return view_classwise_fn

  def get_view_elementwise_fn(self):
    def view_elementwise_fn(x):
      assert x.size(0) == self.n_classes
      assert x.size(1) == self.n_samples
      rest_dims = [x.size(i) for i in range(2, len(x.shape))]
      return x.view(self.n_classes * self.n_samples, *rest_dims)
    return view_elementwise_fn

  def cuda(self, device=None):
    imgs = self.imgs.cuda(device)
    labels = self.labels.cuda(device)
    ids = self.ids  # .cuda()  # useless for now
    return Dataset(imgs, labels, ids, self.name)

  def cpu(self):
    imgs = self.imgs.cpu()
    labels = self.labels.cpu()
    ids = self.ids  # .cuda()  # useless for now
    return Dataset(imgs, labels, ids, self.name)

  def numpy(self):
    """
    Returns:
      tuple(imgs(numpy.ndarray), labels(numpy.ndarray), ids(numpy.ndarray))
    """
    imgs = np.transpose(self.imgs.cpu().numpy(), (0, 3, 2, 1))
    labels = self.labels.cpu().numpy()
    ids = self.ids.cpu().numpy()
    return (imgs, labels, ids)

  @classmethod
  def from_numpy(cls, imgs, labels, ids, name):
    """Args:
        imgs(numpy.ndarray): images
        labels(numpy.ndarray): temporal class id for current episode
        ids(numpy.ndarray): fixed class with repect to entire classes

      Returns:
        tuple(torch.FloatTensor, torch.LongTensor, torch.LongTensor)
    """
    def to_torch_imgs(imgs):
      return torch.from_numpy(np.transpose(imgs, (0, 3, 2, 1)))

    def to_torch_ids(labels):
      return torch.from_numpy(labels).long()
    # plot_episode(support_images=e[0], support_class_ids=e[2],
    #              query_images=e[3], query_class_ids=e[5])
    return cls(
        to_torch_imgs(imgs), to_torch_ids(labels), to_torch_ids(ids), name)

  @classmethod
  def from_tf(cls, imgs, labels, ids, name):
    """Args:
        imgs(tf.Tensor): images
        labels(tf.Tensor): temporal class indices for current episode
        ids(tf.Tensor): fixed class indices with repect to entire classes

      Returns:
        tuple(torch.FloatTensor, torch.LongTensor, torch.LongTensor)
    """
    return cls.from_numpy(imgs.numpy(), labels.numpy(), ids.numpy(), name)

  @classmethod
  def concat(cls, datasets):
    assert isinstance(datasets, Iterable)
    if all([datasets[0].name == datasets[i].name
            for i in range(len(datasets))]):
      name = dataset[0].name
    else:
      name = 'Support + Query'
    # TODO: heterogeneous concat?
    return cls(
        *map(lambda x: torch.cat(x, dim=0), zip(*datasets)), datasets[0].name)

  def plot(self, size_multiplier=1, max_imgs_per_col=10, max_imgs_per_row=10,
           show=False):
    images, _, class_ids = self.numpy()
    # FIX LATER: This is only for the tensors already up on GPU.
    n_samples_per_class = Counter(class_ids)
    n_samples_per_class = {k: min(v, max_imgs_per_col)
                           for k, v in n_samples_per_class.items()}
    id_plot_index_map = {k: i for i, k
                         in enumerate(n_samples_per_class.keys())}
    num_classes = min(max_imgs_per_row, len(n_samples_per_class.keys()))
    max_n_sample = max(n_samples_per_class.values())

    figwidth = max_n_sample
    figheight = num_classes
    figsize = (figheight * size_multiplier, figwidth * size_multiplier)
    fig, axarr = plt.subplots(figwidth, figheight, figsize=figsize)
    fig.suptitle('%s Set' % self.name, size='20')
    fig.tight_layout(pad=3, w_pad=0.1, h_pad=0.1)
    reverse_id_map = {v: k for k, v in id_plot_index_map.items()}

    for i, ax in enumerate(axarr.flat):
      ax.patch.set_alpha(0)
      # Print the class ids, this is needed since, we want to set the x axis
      # even there is no picture.
      ax.set(xlabel=reverse_id_map[i % figheight], xticks=[], yticks=[])
      ax.label_outer()

    for image, class_id in zip(images, class_ids):
      # First decrement by one to find last spot for the class id.
      n_samples_per_class[class_id] -= 1
      # If class column is filled or not represented: pass.
      if (n_samples_per_class[class_id] < 0 or
              id_plot_index_map[class_id] >= max_imgs_per_row):
        continue
      # If width or height is 1, then axarr is a vector.
      if axarr.ndim == 1:
        ax = axarr[n_samples_per_class[class_id]
                   if figheight == 1 else id_plot_index_map[class_id]]
      else:
        ax = axarr[n_samples_per_class[class_id],
                   id_plot_index_map[class_id]]
      ax.imshow(image * 0.5 + 0.5)
    return plt

  def save_fig(self, file_name, save_path, i):
    if save_path is None:
      return
    file_name = file_name + '_' + str(i).zfill(4)
    file_path = os.path.join(save_path, file_name + '.png')
    if not os.path.exists(os.path.dirname(file_path)):
      os.makedirs(os.path.dirname(file_path))
    plot = self.plot()
    plot.savefig(file_path)
    plot.close()


class EpisodeClassIndexer(object):
  def __init__(self, episode):
    assert isinstance(episode, Episode)
    self.episode = episode

  def __len__(self):
    return self.episode.n_classes

  def __repr__(self):
    return __class__.__name__ + f' for \n{self.episode}'

  def __getitem__(self, key):
    """Class indexing"""
    s = self.episode.s.classwise[key]
    q = self.episode.q.classwise[key]
    return Episode(s, q, s.n_classes, self.episode.name)

  def masked_select(self, mask):
    assert isinstance(mask, torch.Tensor)
    idx = mask.squeeze().nonzero().view(-1)
    # assert len(idx) == len(self)
    if len(idx) == 0:
      return None
    else:
      return self.__getitem__(idx)


class Episode(object):
  """Collection of support and query set. Single episode for a task adaptation
  can be wrapped with this class."""

  def __init__(self, support, query, n_total_classes, name='Episode'):
    assert all([isinstance(set, Dataset) for set in [support, query]])
    self.s = support
    self.q = query
    self.name = name
    self.n_total_classes = n_total_classes
    self.classwise = EpisodeClassIndexer(self)

  def __iter__(self):
    return iter([self.s, self.q])

  def __repr__(self):
    return (__class__.__name__ + f'(\n\tsupport={self.s}, \n\tquery={self.q}, '
      f'\n\tname={self.name}), n_total_classes={self.n_total_classes})')

  def __getitem__(self, key):
    return Episode(self.s[key], self.q[key], self.n_total_classes, self.name)

  @property
  def n_classes(self):
    assert self.s.n_classes == self.q.n_classes
    return self.s.n_classes

  def masked_select(self, mask):
    assert isinstance(mask, torch.Tensor)
    idx = mask.squeeze().nonzero().view(-1)
    if len(idx) == 0:
      # TODO: exception handling
      # s = None
      # q = None
      raise Exception('All classes were dropped!')
    else:
      s = self.s[idx]
      q = self.q[idx]
    return Episode(s, q, self.n_total_classes, self.name)

  def offset_indices(self, offset_labels, offset_ids):
    return Episode(*[set.offset_indices(offset_labels, offset_ids)
                     for set in self], self.n_total_classes)

  def cuda(self, device=None):
    s = self.s.cuda(device)
    q = self.q.cuda(device)
    return Episode(s, q, self.n_total_classes, self.name)

  def cpu(self):
    s = self.s.cpu()
    q = self.q.cpu()
    return Episode(s, q, self.n_total_classes, self.name)

  def numpy(self):
    """Returns tuple of numpy arrays.
    Returns:
      (support_imgs, support_labels, support_ids,
       query_imgs, query_labels, query_ids)
    """
    return (*self.s.numpy(), *self.q.numpy())

  def subset(self):
    pass

  @classmethod
  def concat(cls, episodes):
    """concatenate heterogeneous(e.g. Omniglot + ImageNet) episodes.
       class labels and ids in its own dataset need renumbering.
       (For Meta-dataset)
    """
    assert isinstance(episodes, Iterable)
    episodes_ = []
    prev_n_cls = prev_n_total_cls = 0
    new_n_total_cls = 0
    for episode in episodes:
      episodes_.append(episode.offset_indices(prev_n_cls, prev_n_total_cls))
      prev_n_cls = episode.n_classes
      prev_n_total_cls = episode.n_total_classes
      new_n_total_cls += episode.n_total_classes
    return cls(
      *map(Dataset.concat, zip(*episodes_)), new_n_total_cls, self.name)

  @property
  def concatenated(self):
    """concatenate support and query. returns loader.episode.Dataset."""
    return self.s.concat([self.s, self.q])


  @classmethod
  def from_numpy(cls, episode, n_total_classes):
    """Args:
        episode[0:3]: support image / support labels / support class ids
        episode[4:6]: query image / query labels / query class ids
        (In accordance with the format of meta-dataset pipline.)

      Returns:
        loader.Dataset
    """
    support = Dataset.from_numpy(*episode[0:3], 'Support')
    query = Dataset.from_numpy(*episode[3:6], 'Query')
    return cls(support, query, n_total_classes)

  @classmethod
  def from_tf(cls, episode, n_total_classes):
    """Args:
        episode[0:3]: support image / support labels / support class ids
        episode[4:6]: query image / query labels / query class ids
        (In accordance with the format of meta-dataset pipline.)

      Returns:
        loader.Dataset
    """
    support = Dataset.from_tf(*episode[0:3], 'Support')
    query = Dataset.from_tf(*episode[3:6], 'Query')
    return cls(support, query, n_total_classes)

  @classmethod
  def from_tensors(cls, support, query, name):
    """TODO: n_total_classes is meaningless."""
    support = Dataset(*support, 'Support')
    query = Dataset(*query, 'Query')
    return cls(support, query, 0, name)

  @classmethod
  def get_sampled_episode_from_loaders(cls, meta_s, meta_q, split_ratio):
    Episode(meta_s.split_instances(split_ratio).get_loader())
    meta_q_loader = meta_q.split_instances(split_ratio).get_loader()


  def show(self, size_multiplier=1, max_imgs_per_col=10, max_imgs_per_row=10):
    self.s.plot(size_multiplier, max_imgs_per_col, max_imgs_per_row).show()
    self.q.plot(size_multiplier, max_imgs_per_col, max_imgs_per_row).show()

import os
import pdb
import random
import sys
from collections import Counter
from collections.abc import Iterable
from random import randrange

import gin
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from meta_dataset.data import config
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec, pipeline

# BASE_PATH = '/v14/records'  # SSD
BASE_PATH = '/st1/dataset/meta-dataset/records'  # HDD
GIN_FILE_PATH = 'meta_dataset/learn/gin/setups/learn2sample.gin'
gin.parse_config_file(GIN_FILE_PATH)
tf.enable_eager_execution()


class Dataset(object):
  """Dataset class that should contain images, labels, and ids.
  Support set or query set can be a proper candidate of Dataset."""
  def __init__(self, imgs, labels, ids, name='Dataset'):
    assert all([isinstance(t, torch.Tensor) for t in [imgs, labels, ids]])
    assert name in ['Dataset', 'Support', 'Query']
    self.imgs = imgs
    self.labels = labels
    self.ids = ids
    self._n_classes = None
    self._n_samples = None
    self._name = name

  def __iter__(self):
    return iter([self.imgs, self.labels, self.ids])

  @property
  def n_classes(self):
    if self._n_classes is None:
      self._n_classes = len(Counter(self.labels.cpu().numpy()).keys())
    return self._n_classes

  @property
  def n_samples(self):
    if self._n_samples is None:
      self._n_samples = int(self.imgs.size(0) / self.n_classes)
    return self._n_samples

  def offset_indices(self, offset_labels, offset_ids):
    labels = self.labels + offset_labels
    ids = self.ids + offset_ids
    return Dataset(self.imgs, labels, ids)

  def get_view_classwise_fn(self):
    def view_classwise_fn(x):
      assert x.size(0) == self.n_classes * self.n_samples
      rest_dims = [x.size(i) for i in range(1, len(x.shape))]
      return x.view(self.n_classes, self.n_samples, *rest_dims)
    return view_classwise_fn

  def get_view_elementwise_fn(self):
    def view_elementwise_fn(x):
      assert x.size(0) == self.n_classes
      assert x.size(1) == self.n_samples
      rest_dims = [x.size(i) for i in range(2, len(x.shape))]
      return x.view(self.n_classes * self.n_samples, *rest_dims)
    return view_elementwise_fn

  def cuda(self, device):
    imgs = self.imgs.cuda(device)
    labels = self.labels.cuda(device)
    ids = self.ids # .cuda()  # useless for now
    return Dataset(imgs, labels, ids)

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
    return cls(*map(lambda x: torch.cat(x, dim=0), zip(*datasets)))

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
    fig.suptitle('%s Set' % self._name, size='20')
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
      ax.imshow(image / 2 + 0.5)
    return plt

  def save_fig(self, file_name, save_path):
    if save_path is None:
      return
    file_path = os.path.join(save_path, file_name + '.png')
    if not os.path.exists(os.path.dirname(file_path)):
      os.makedirs(os.path.dirname(file_path))
    plot = self.plot()
    plot.savefig(file_path)
    plot.close()


class Episode(object):
  """Collection of support and query set. Single episode for a task adaptation
  can be wrapped with this class."""
  def __init__(self, support, query, n_total_classes):
    assert all([isinstance(set, Dataset) for set in [support, query]])
    self.s = support
    self.q = query
    self.n_total_classes = n_total_classes

  def __iter__(self):
    return iter([self.s, self.q])

  @property
  def n_classes(self):
    assert self.s.n_classes == self.q.n_classes
    return self.s.n_classes

  def offset_indices(self, offset_labels, offset_ids):
    return Episode(*[set.offset_indices(offset_labels, offset_ids)
                   for set in self], self.n_total_classes)

  def cuda(self, device):
    s = self.s.cuda(device)
    q = self.q.cuda(device)
    return Episode(s, q, self.n_total_classes)

  def numpy(self):
    """Returns tuple of numpy arrays.
    Returns:
      (support_imgs, support_labels, support_ids,
       query_imgs, query_labels, query_ids)
    """
    return (*self.s.numpy(), *self.q.numpy())

  @classmethod
  def concat(cls, episodes):
    assert isinstance(episodes, Iterable)
    episodes_ = []
    prev_n_cls = prev_n_total_cls = 0
    new_n_total_cls = 0
    for episode in episodes:
      episodes_.append(episode.offset_indices(prev_n_cls, prev_n_total_cls))
      prev_n_cls = episode.n_classes
      prev_n_total_cls = episode.n_total_classes
      new_n_total_cls += episode.n_total_classes
    return cls(*map(Dataset.concat, zip(*episodes_)), new_n_total_cls)

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

  def show(self, size_multiplier=1, max_imgs_per_col=10, max_imgs_per_row=10):
    self.s.plot(size_multiplier, max_imgs_per_col, max_imgs_per_row).show()
    self.q.plot(size_multiplier, max_imgs_per_col, max_imgs_per_row).show()


@gin.configurable
class MetaDataset(object):
  """MetaDataset(https://arxiv.org/abs/1903.03096.) converting tfrecord into
  torch.Tensor. Reference: https://github.com/google-research/meta-dataset/blob/
  master/Intro_to_Metadataset.ipynb """
  def __init__(self, datasets, split, fixed_ways=None, fixed_support=None,
               fixed_query=None, use_ontology=True):
    assert split in ['train', 'valid', 'test']
    assert isinstance(datasets, (list, tuple))
    assert isinstance(datasets[0], str)

    print(f'Loading MetaDataset for {split}..')
    split = getattr(learning_spec.Split, split.upper())
    # Reading datasets
    # self.datasets = ['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012',
    # 'omniglot', 'quickdraw', 'vgg_flower']
    self.datasets = datasets

    # Ontology setting
    use_bilevel_ontology_list = []
    use_dag_ontology_list = []

    for dataset in self.datasets:
      bilevel = dag = False
      if dataset == 'omniglot' and use_ontology:
        bilevel = True
      elif dataset == 'ilsvrc_2012' and use_ontology:
        dag = True
      use_bilevel_ontology_list.append(bilevel)
      use_dag_ontology_list.append(dag)

    assert len(self.datasets) == len(use_bilevel_ontology_list)
    assert len(self.datasets) == len(use_dag_ontology_list)

    all_dataset_specs = []
    for dataset_name in self.datasets:
      dataset_records_path = os.path.join(BASE_PATH, dataset_name)
      dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
      all_dataset_specs.append(dataset_spec)

    if fixed_ways and use_ontology:
      max_ways_upper_bound = min_ways = fixed_ways
      self.episode_config = config.EpisodeDescriptionConfig(
        num_query=fixed_query, num_support=fixed_support, min_ways=min_ways,
        max_ways_upper_bound=max_ways_upper_bound, num_ways=None)
    else:
      # Episode description config (if num is None, use gin configuration)
      self.episode_config = config.EpisodeDescriptionConfig(
        num_query=fixed_query, num_support=fixed_support, num_ways=fixed_ways)

    # Episode pipeline
    self.episode_pipeline, self.n_total_classes = \
      pipeline.make_multisource_episode_pipeline2(
        dataset_spec_list=all_dataset_specs,
        use_dag_ontology_list=use_dag_ontology_list,
        use_bilevel_ontology_list=use_bilevel_ontology_list,
        episode_descr_config=self.episode_config,
        split=split, image_size=84)

    print('MetaDataset loaded: ', ', '.join([d for d in self.datasets]))

  def loader(self, n_batches):
    if not tf.executing_eagerly():
      iterator = self.episode_pipeline.make_one_shot_iterator()
      next_episode = iterator.get_next()
      with tf.Session() as sess:
        for _ in range(n_batches):
          episode = Episode.from_numpy(
            sess.run(next_episode), self.n_total_classes)
          yield episode
    else:
      # iterator = iter(self.episode_pipeline)
      for i, episode in enumerate(self.episode_pipeline):
        if i == n_batches:
          break
        episode = Episode.from_tf(episode, self.n_total_classes)
        yield episode


class PseudoMetaDataset(object):
  """This class can be used for debugging at less cost in time(hopefully)."""

  def __init__(self, in_dim=[3, 84, 84], n_ways=[10, 25],
               n_support=[5, 5], n_query=[15, 15]):
    self.in_dim = in_dim
    self.n_ways = n_ways
    self.n_support = n_support
    self.n_query = n_query

  def sample_shape(self):
    return [random.randint(*n) for n in
            [self.n_ways, self.n_support, self.n_query]]

  def make_fakeset(self, n_ways, n_samples):
    imgs = torch.rand(n_ways * n_samples, *self.in_dim)
    labels = torch.LongTensor([([i] * n_samples) for i in range(n_ways)])
    ids = labels = labels.view(-1, 1).squeeze()
    return Dataset(imgs, labels, ids)

  def loader(self, n_batches):
    for i in range(n_batches):
      n_ways, n_support, n_query = self.sample_shape()
      episode = Episode(self.make_fakeset(n_ways, n_support),
                        self.make_fakeset(n_ways, n_query))
      yield episode


@gin.configurable
class MetaMultiDataset(object):
  """A class for merging multiple MetaDataset into one.
  Entire ways(number of classes) will be taken up by the multiple datasets in
  equal propotion as much as possible.
  """
  def __init__(self, multi_mode, datasets, split, fixed_ways=None,
               fixed_support=None, fixed_query=None, use_ontology=True):
    assert split in ['train', 'valid', 'test']
    assert isinstance(datasets, (list, tuple))
    assert isinstance(datasets[0], str)
    self.multi_mode = multi_mode

    if self.multi_mode:
      each_ways = []
      for i in range(len(datasets)):
        ways = fixed_ways // len(datasets)
        if i < len(datasets) - 1:
          ways += fixed_ways % len(datasets)
        each_ways.append(ways)

      self.meta_dataset = []
      for dataset, n_way in zip(datasets, each_ways):
        self.meta_dataset.append(
            MetaDataset([dataset], split, n_way,
                        fixed_support, fixed_query, use_ontology))
    else:
      self.meta_dataset = MetaDataset(datasets, split, fixed_ways,
                                      fixed_support, fixed_query, use_ontology)

  def loader(self, n_batches):
    if self.multi_mode:
      for i in range(n_batches):
        if i == n_batches:
          break
        yield Episode.concat([next(meta_d.loader(1))
                             for meta_d in self.meta_dataset])
    else:
      return self.meta_dataset.loader(n_batches)


if __name__ == '__main__':

  # test code
  metadata = MetaDataset(split='train')

  for i, epi in enumerate(metadata.loader(n_batches=2)):
    print(epi.s.imgs.shape, epi.s.labels.shape,
          epi.q.imgs.shape, epi.q.labels.shape)
    epi.show()

  metadata = MetaDataset(split='test')

  for i, epi in enumerate(metadata.loader(n_batches=2)):
    print(epi.s.imgs.shape, epi.s.labels.shape,
          epi.q.imgs.shape, epi.q.labels.shape)
    epi.show()

import os
import pdb
import random
import sys
from collections import Counter

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
  def __init__(self, imgs, labels, ids):
    assert all([isinstance(t, torch.Tensor) for t in [imgs, labels, ids]])
    self.imgs = imgs
    self.labels = labels
    self.ids = ids
    self._n_classes = None
    self._n_samples = None

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

  def get_view_classwise_fn(self):
    def view_classwise_fn(x):
      assert x.size(0) == self.n_classes*self.n_samples
      rest_dims = [x.size(i) for i in range(1, len(x.shape))]
      return x.view(self.n_classes, self.n_samples, *rest_dims)
    return view_classwise_fn

  def get_view_elementwise_fn(self):
    def view_elementwise_fn(x):
      assert x.size(0) == self.n_classes
      assert x.size(1) == self.n_samples
      rest_dims = [x.size(i) for i in range(2, len(x.shape))]
      return x.view(self.n_classes*self.n_samples, *rest_dims)
    return view_elementwise_fn

  def cuda(self):
    self.imgs = self.imgs.cuda()
    self.labels = self.labels.cuda()
    # self.ids = self.ids.cuda()  # useless for now
    return self

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
  def from_numpy(cls, imgs, labels, ids):
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
    return cls(to_torch_imgs(imgs), to_torch_ids(labels), to_torch_ids(ids))

  @classmethod
  def from_tf(cls, imgs, labels, ids):
    """Args:
        imgs(tf.Tensor): images
        labels(tf.Tensor): temporal class indices for current episode
        ids(tf.Tensor): fixed class indices with repect to entire classes

      Returns:
        tuple(torch.FloatTensor, torch.LongTensor, torch.LongTensor)
    """
    return cls.from_numpy(imgs.numpy(), labels.numpy(), ids.numpy())


class Episode(object):
  def __init__(self, support, query):
    assert all([isinstance(set, Dataset) for set in [support, query]])
    self.s = support
    self.q = query

  @property
  def n_classes(self):
    assert self.s.n_classes == self.q.n_classes
    return self.s.n_classes

  def cuda(self):
    self.s = self.s.cuda()
    self.q = self.q.cuda()
    return self

  def numpy(self):
    """Returns tuple of numpy arrays.
    Returns:
      (support_imgs, support_labels, support_ids,
       query_imgs, query_labels, query_ids)
    """
    return (*self.s.numpy(), *self.q.numpy())

  @classmethod
  def from_numpy(cls, episode):
    """Args:
        episode[0:3]: support image / support labels / support class ids
        episode[4:6]: query image / query labels / query class ids
        (In accordance with the format of meta-dataset pipline.)

      Returns:
        loader.Dataset
    """
    support = Dataset.from_numpy(*episode[0:3])
    query = Dataset.from_numpy(*episode[3:6])
    return cls(support, query)

  @classmethod
  def from_tf(cls, episode):
    """Args:
        episode[0:3]: support image / support labels / support class ids
        episode[4:6]: query image / query labels / query class ids
        (In accordance with the format of meta-dataset pipline.)

      Returns:
        loader.Dataset
    """
    support = Dataset.from_tf(*episode[0:3])
    query = Dataset.from_tf(*episode[3:6])
    return cls(support, query)

  def plot(self, size_multiplier=1, max_imgs_per_col=10, max_imgs_per_row=10):
    support_imgs, _, support_ids = self.s.numpy()
    query_imgs, _, query_ids = self.q.numpy()
    for name, images, class_ids in zip(('Support', 'Query'),
                                       (support_imgs, query_imgs),
                                       (support_ids, query_ids)):
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
      if name == 'Support':
        print('#Classes: %d' % len(n_samples_per_class.keys()))
      figsize = (figheight * size_multiplier, figwidth * size_multiplier)
      fig, axarr = plt.subplots(figwidth, figheight, figsize=figsize)
      fig.suptitle('%s Set' % name, size='20')
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
      plt.show()


class MetaDataset(object):
  def __init__(self, split):
    assert split in ['train', 'test']
    split = getattr(learning_spec.Split, split.upper())

    # Reading datasets
    # self.datasets = ['aircraft', 'cu_birds', 'dtd', 'fungi', 'ilsvrc_2012',
    # 'omniglot', 'quickdraw', 'vgg_flower']
    self.datasets = ['omniglot']

    # Ontology setting
    use_bilevel_ontology_list = [False] * len(self.datasets)
    use_dag_ontology_list = [False] * len(self.datasets)

    # # all_dataset
    # use_bilevel_ontology_list[5] = True
    # use_dag_ontology_list[4] = True

    # Omniglot only
    use_bilevel_ontology_list[0] = True

    all_dataset_specs = []
    for dataset_name in self.datasets:
      dataset_records_path = os.path.join(BASE_PATH, dataset_name)
      dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_records_path)
      all_dataset_specs.append(dataset_spec)

    # Fixed way / support / query
    # (single dataset without ontology)
    NUM_WAYS = None
    NUM_SUPPORT = 15
    NUM_QUERY = 5

    # Episode description config
    self.episode_config = config.EpisodeDescriptionConfig(
      num_query=NUM_QUERY, num_support=NUM_SUPPORT, num_ways=NUM_WAYS)

    # Episode pipeline
    self.episode_pipeline = pipeline.make_multisource_episode_pipeline(
      dataset_spec_list=all_dataset_specs,
      use_dag_ontology_list=use_dag_ontology_list,
      use_bilevel_ontology_list=use_bilevel_ontology_list,
      episode_descr_config=self.episode_config,
      split=split, image_size=84)

  def loader(self, n_batches):
    if not tf.executing_eagerly():
      iterator = self.episode_pipeline.make_one_shot_iterator()
      next_episode = iterator.get_next()
      with tf.Session() as sess:
        for _ in range(n_batches):
          episode = Episode.from_numpy(sess.run(next_episode))
          yield episode
    else:
      # iterator = iter(self.episode_pipeline)
      for i, episode in enumerate(self.episode_pipeline):
        if i == n_batches:
          break
        episode = Episode.from_tf(episode)
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


if __name__ == '__main__':

  # test code
  metadata = MetaDataset(split='train')

  for i, epi in enumerate(metadata.loader(n_batches=2)):
    print(epi.s.imgs.shape, epi.s.labels.shape,
          epi.q.imgs.shape, epi.q.labels.shape)
    epi.plot()

  metadata = MetaDataset(split='test')

  for i, epi in enumerate(metadata.loader(n_batches=2)):
    print(epi.s.imgs.shape, epi.s.labels.shape,
          epi.q.imgs.shape, epi.q.labels.shape)
    epi.plot()

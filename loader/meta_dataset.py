import os
import pdb
import random
from random import randrange

import gin
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from loader.episode import Episode
from meta_dataset.data import config
from meta_dataset.data import dataset_spec as dataset_spec_lib
from meta_dataset.data import learning_spec, pipeline

# BASE_PATH = '/v14/records'  # SSD
BASE_PATH = '/st1/dataset/meta-dataset/records'  # HDD
GIN_FILE_PATH = 'meta_dataset/learn/gin/setups/learn2sample.gin'
gin.parse_config_file(GIN_FILE_PATH)


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

# coding=utf-8
# Copyright 2019 The Meta-Dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Configuration classes for data processing.

Config classes that parametrize the behaviour of different stages of the data
processing pipeline, and are set up via `gin`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin.tf

import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


@gin.configurable
class DataConfig(object):
  """Common configuration options for creating data processing pipelines."""

  def __init__(self, image_height, shuffle_buffer_size, read_buffer_size_bytes,
               num_prefetch):
    """Initialize a DataConfig.

    Args:
      image_height: An integer, the desired height for the images output by the
        data pipeline. Images are squared and have 3 channels (RGB), so each
        image will have shape [image_height, image_height, 3],
      shuffle_buffer_size: An integer, the size of the example buffer in the
        tf.data.Dataset.shuffle operations (there is typically one shuffle per
        class in the episodic setting, one per dataset in the batch setting).
        Classes with fewer examples as this number are shuffled in-memory.
      read_buffer_size_bytes: An integer, the size (in bytes) of the read buffer
        for each tf.data.TFRecordDataset (there is typically one for each class
        of each dataset).
      num_prefetch: int, the number of examples to prefetch for each class of
        each dataset. Prefetching occurs just after the class-specific Dataset
        object is constructed. If < 1, no prefetching occurs.
    """
    self.image_height = image_height
    self.shuffle_buffer_size = shuffle_buffer_size
    self.read_buffer_size_bytes = read_buffer_size_bytes
    self.num_prefetch = num_prefetch


class DataAugmentation(object):
  """Configurations for performing data augmentation."""

  def __init__(self, enable_jitter, jitter_amount, enable_gaussian_noise,
               gaussian_noise_std):
    """Initialize a DataAugmentation.

    Args:
      enable_jitter: bool whether to use image jitter (pad each image using
        reflection along x and y axes and then random crop).
      jitter_amount: amount (in pixels) to pad on all sides of the image.
      enable_gaussian_noise: bool whether to use additive Gaussian noise.
      gaussian_noise_std: Standard deviation of the Gaussian distribution.
    """
    self.enable_jitter = enable_jitter
    self.jitter_amount = jitter_amount
    self.enable_gaussian_noise = enable_gaussian_noise
    self.gaussian_noise_std = gaussian_noise_std


@gin.configurable
class SupportSetDataAugmentation(DataAugmentation):
  """Configurations for performing data augmentation on support set."""
  pass


@gin.configurable
class QuerySetDataAugmentation(DataAugmentation):
  """Configurations for performing data augmentation on query set."""
  pass


@gin.configurable
class BatchDataAugmentation(DataAugmentation):
  """Configurations for performing data augmentation for batch."""
  pass


@gin.configurable
class EpisodeDescriptionConfig(object):
  """Configuration options for episode characteristics."""

  def __init__(self, num_ways, num_support, num_query, min_ways,
               max_ways_upper_bound, max_num_query, max_support_set_size,
               max_support_size_contrib_per_class, min_log_weight,
               max_log_weight, ignore_dag_ontology, ignore_bilevel_ontology):
    """Initialize a EpisodeDescriptionConfig.

    This is used in sampling.py in Trainer and in EpisodeDescriptionSampler to
    determine the parameters of episode creation relating to the ways and shots.

    Args:
      num_ways: Integer, fixes the number of classes ("ways") to be used in each
        episode. None leads to variable way.
      num_support: Integer, fixes the number of examples for each class in the
        support set.
      num_query: Integer, fixes the number of examples for each class in the
        query set.
      min_ways: Integer, the minimum value when sampling ways.
      max_ways_upper_bound: Integer, the maximum value when sampling ways. Note
        that the number of available classes acts as another upper bound.
      max_num_query: Integer, the maximum number of query examples per class.
      max_support_set_size: Integer, the maximum size for the support set.
      max_support_size_contrib_per_class: Integer, the maximum contribution for
        any given class to the support set size.
      min_log_weight: Float, the minimum log-weight to give to any particular
        class when determining the number of support examples per class.
      max_log_weight: Float, the maximum log-weight to give to any particular
        class.
      ignore_dag_ontology: Whether to ignore ImageNet's DAG ontology when
        sampling classes from it. This has no effect if ImageNet is not part of
        the benchmark.
      ignore_bilevel_ontology: Whether to ignore Omniglot's DAG ontology when
        sampling classes from it. This has no effect if Omniglot is not part of
        the benchmark.
    """
    self.num_ways = num_ways
    self.num_support = num_support
    self.num_query = num_query
    self.min_ways = min_ways
    self.max_ways_upper_bound = max_ways_upper_bound
    self.max_num_query = max_num_query
    self.max_support_set_size = max_support_set_size
    self.max_support_size_contrib_per_class = max_support_size_contrib_per_class
    self.min_log_weight = min_log_weight
    self.max_log_weight = max_log_weight
    self.ignore_dag_ontology = ignore_dag_ontology
    self.ignore_bilevel_ontology = ignore_bilevel_ontology

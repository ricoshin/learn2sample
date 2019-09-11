import bisect
import copy
import os
import random
import sys
from collections import OrderedDict as odict
from os import path

import numpy as np
import torch
from datasets.metadata import Metadata
from PIL import Image
from torch.utils import data
from torch.utils.data.dataset import ConcatDataset, Subset
from utils import utils

C = utils.getCudaManager('default')


def has_file_allowed_extension(filename, extensions):
  """Checks if a file is an allowed extension.

  Args:
    filename (string): path to a file
    extensions (tuple of strings): extensions to consider (lowercase)

  Returns:
    bool: True if the filename ends with one of given extensions
  """
  return filename.lower().endswith(extensions)


def is_image_file(filename):
  """Checks if a file is an allowed image extension.

  Args:
    filename (string): path to a file

  Returns:
    bool: True if the filename ends with a known image extension
  """
  return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
  idx_to_samples = odict()
  dir = os.path.expanduser(dir)
  if not ((extensions is None) ^ (is_valid_file is None)):
    raise ValueError("Both extensions and is_valid_file cannot be None "
                     "or not None at the same time")
  if extensions is not None:
    def is_valid_file(x):
      return has_file_allowed_extension(x, extensions)
  for target in sorted(class_to_idx.keys()):
    samples = []
    d = os.path.join(dir, target)
    idx = class_to_idx[target]
    if not os.path.isdir(d):
      continue
    for root, _, fnames in sorted(os.walk(d)):
      for fname in sorted(fnames):
        path = os.path.join(root, fname)
        if is_valid_file(path):
          samples.append((path, idx))
    idx_to_samples[idx] = samples
  return idx_to_samples


def pil_loader(path):
  # open path as file to avoid ResourceWarning
  #   (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


def accimage_loader(path):
  import accimage
  try:
    return accimage.Image(path)
  except IOError:
    # Potentially a decoding problem, fall back to PIL.Image
    return pil_loader(path)


def default_loader(path):
  from torchvision import get_image_backend
  if get_image_backend() == 'accimage':
    return accimage_loader(path)
  else:
    return pil_loader(path)


class VisionDataset(data.Dataset):
  _repr_indent = 4

  def __init__(self, root, transforms=None, transform=None, target_transform=None):
    if isinstance(root, torch._six.string_classes):
      root = os.path.expanduser(root)
    self.root = root

    has_transforms = transforms is not None
    has_separate_transform = transform is not None or target_transform is not None
    if has_transforms and has_separate_transform:
      raise ValueError("Only transforms or transform/target_transform can "
                       "be passed as argument")

    # for backwards-compatibility
    self.transform = transform
    self.target_transform = target_transform

    if has_separate_transform:
      transforms = StandardTransform(transform, target_transform)
    self.transforms = transforms

  def __getitem__(self, index):
    raise NotImplementedError

  def __len__(self):
    raise NotImplementedError

  def __repr__(self):
    head = "Dataset " + self.__class__.__name__
    body = ["Number of datapoints: {}".format(self.__len__())]
    if self.root is not None:
      body.append("Root location: {}".format(self.root))
    body += self.extra_repr().splitlines()
    if self.transforms is not None:
      body += [repr(self.transforms)]
    lines = [head] + [" " * self._repr_indent + line for line in body]
    return '\n'.join(lines)

  def _format_transform_repr(self, transform, head):
    lines = transform.__repr__().splitlines()
    return (["{}{}".format(head, lines[0])] +
            ["{}{}".format(" " * len(head), line) for line in lines[1:]])

  def extra_repr(self):
    return ""


class DatasetClassSampler(VisionDataset):
  """A generic data loader where the samples are arranged in this way: ::

      root/class_x/xxx.ext
      root/class_x/xxy.ext
      root/class_x/xxz.ext

      root/class_y/123.ext
      root/class_y/nsdf3.ext
      root/class_y/asd932_.ext

  Args:
    root (string): Root directory path.
    loader (callable): A function to load a sample given its path.
    extensions (tuple[string]): A list of allowed extensions.
      both extensions and is_valid_file should not be passed.
    transform (callable, optional): A function/transform that takes in
      a sample and returns a transformed version.
      E.g, ``transforms.RandomCrop`` for images.
    target_transform (callable, optional): A function/transform that takes
      in the target and transforms it.
    is_valid_file (callable, optional): A function that takes path of an Image
      file and check if the file is a valid_file(used to check of corrupt files)
      both extensions and is_valid_file should not be passed.

   Attributes:
    classes (list): List of the class names.
    class_to_idx (dict): Dict with items (class_name, class_index).
    samples (list): List of (sample path, class_index) tuples
    targets (list): The class_index value for each image in the dataset
  """

  def __init__(self, root, loader, extensions=None, transform=None,
               target_transform=None, is_valid_file=None, meta_data=None,
               remake=False, visible_subdirs=None):
    super(DatasetClassSampler, self).__init__(root)
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader
    self.extensions = extensions
    self.preloaded = False

    if meta_data is not None and not remake:
      self.meta = meta_data
    else:
      # generate new meta data or load if posiible
      self.meta = Metadata.load_or_make(
          # meta_dir=path.join(root, os.pardir),
          # name=path.basename(root),
          remake=remake,
          data_dir=root,
          visible_subdirs=visible_subdirs,
          extensions=extensions,
          is_valid_file=is_valid_file,
      )
    # self.classes = metadata.classes
    # self.class_to_idx = metadata.class_to_idx
    # self.idx_to_samples = idx_to_samples
    self.samples = self.meta.idx_to_samples
    self.visible_classes = tuple(self.meta.idx_to_class.keys())
    # self.cumulative_sizes = self._cumsum(self.meta.idx_to_samples.values())
    # self.samples, self.targets = self.samples()

  @property
  def visible_classes(self):
    return self._visible_classes

  @visible_classes.setter
  def visible_classes(self, indices):
    self._visible_classes = indices
    bound_indices, cumsum = [], 0
    for idx in self.visible_classes:
      cumsum += len(self.meta.idx_to_samples[idx])
      bound_indices.append(cumsum)
    self.bound_indices = bound_indices

  def sample_visible_classes(self, rel_idx, preload):
    self.visible_classes = [self.visible_classes[i] for i in rel_idx]
    if preload:
      self.preload_visible_classes()
      self.preloaded = True
    else:
      self.preloaded = False
    return self

  def preload_visible_classes(self):
    classes = [self.meta.idx_to_bin_fname(idx) for idx in self.visible_classes]
    self.samples = []
    for clss in classes:
      self.samples.append(torch.load(path.join(self.root, 'bin', clss)))

  def _get_samples_n_targets(self):
    self.samples = []
    # self.targets = []
    for idx, sample in self.meta.idx_to_samples.items():
      self.samples.extend(sample)
      # self.targets.extend([idx] * len(sample))

  def _make_metadata(self, extensions, is_valid_file):
    print('Making dataset dictionaries..')
    classes, class_to_idx, idx_to_class = self._find_classes(self.root)
    idx_to_samples = make_dataset(
        self.root, class_to_idx, extensions, is_valid_file)
    if any([len(v) == 0 for v in idx_to_samples.values()]):
      raise (RuntimeError(
          "Found 0 files in subfolders of: " + self.root + "\n"
          "Supported extensions are: " + ",".join(extensions)))
    print('Done!')
    return Metadata(classes, class_to_idx, idx_to_class, idx_to_samples)

  def _find_classes(self, dir):
    """
    Finds the class folders in a dataset.

    Args:
      dir (string): Root directory path.

    Returns:
      tuple: (classes, class_to_idx) where classes are relative to (dir),
        and class_to_idx is a dictionary.

    Ensures:
      No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
      # Faster and available in Python 3.5 and above
      classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
      classes = [d for d in os.listdir(
          dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = odict({classes[i]: i for i in range(len(classes))})
    idx_to_class = odict({i: classes[i] for i in range(len(classes))})
    return classes, class_to_idx, idx_to_class

  def _get_bi_level_index(self, idx):
    if idx < 0:
      if -idx > len(self):
        raise ValueError(
            "absolute value of index should not exceed dataset length")
      idx = len(self) + idx
    class_idx = bisect.bisect_right(self.bound_indices, idx)
    if class_idx == 0:
      sample_idx = idx
    else:
      sample_idx = idx - self.bound_indices[class_idx - 1]
    return class_idx, sample_idx

  def __getitem__(self, index):
    """
    Args:
      index (int): Index

    Returns:
      tuple: (sample, target) where target is class_index of the target class.
    """
    visible_class_idx, sample_idx = self._get_bi_level_index(index)
    class_idx = self.visible_classes[visible_class_idx]
    if self.preloaded:
      # sample, target = self.samples[visible_class_idx][sample_idx]
      sample = self.samples[visible_class_idx][sample_idx]
      target = class_idx
    else:
      path, target = self.meta.idx_to_samples[class_idx][sample_idx]
      sample = self.loader(path)

    if self.transform is not None:
      sample = self.transform(sample)
    if self.target_transform is not None:
      target = self.target_transform(target)

    return sample, target

  def subset(self, idx, preload=False):
    if self.preloaded:
      preload = self.preloaded
    return copy.copy(self).sample_visible_classes(idx, preload)


  def __len__(self):
    return self.bound_indices[-1]

  def inter_class_split(self, ratio, shuffle):
    n_classes = len(self.visible_classes)
    idx = list(range(n_classes))
    if shuffle:
      random.shuffle(idx)
    thres = int(n_classes * ratio)
    return self.subset(idx[:thres]), self.subset(idx[thres:])

  # def intra_class_split(self, ratio, shuffle):
  #   n_class = len(self.visible_classes)
  #   classes = [self.subset(self, [i]) for i in range(n_class)]
  #   n_samples = [len(clss) for clss in classes]
  #   idx = list(range(max(n_samples)))
  #   if shuffle:
  #     random.shuffle(idx)  # single sampling can be enough
  #   part_a, part_b = [], []
  #   for clss, size in zip(classes, n_samples):
  #     thres = int(size * ratio)
  #     part_a.append(SubsetInClass(clss, idx[:thres]))
  #     part_b.append(SubsetInClass(clss, idx[thres:size]))
  #
  #   return ConcatClass([a for a in part_a]), ConcatClass([b for b in part_b])

  def intra_class_split(self, ratio, shuffle):
    n_class = len(self.visible_classes)
    classes = [self.subset([i]) for i in range(n_class)]
    n_samples = [len(clss) for clss in classes]
    part_a, part_b = [], []
    for clss, size in zip(classes, n_samples):
      idx = list(range(size))
      if shuffle:
        random.shuffle(idx)
      thres = int(size * ratio)
      part_a.append(Subset(clss, idx[:thres]))
      part_b.append(Subset(clss, idx[thres:]))
    return (ConcatDatasetWithNewLabel([a for a in part_a]),
            ConcatDatasetWithNewLabel([b for b in part_b]))

  def class_sample(self, num, preload):
    sampled_idx = np.random.choice(
        len(self.visible_classes), num, replace=False).tolist()
    return self.subset(sampled_idx, preload)


class ConcatDatasetWithNewLabel(ConcatDataset):
  def __init__(self, datasets):
    super(ConcatDatasetWithNewLabel, self).__init__(datasets)
    self.visible_classes = []
    for dataset in self.datasets:
      self.visible_classes.extend(dataset.dataset.visible_classes)
    self.new_label_map = {c: i for i, c in enumerate(self.visible_classes)}

  def __getitem__(self, idx):
    sample, target = super().__getitem__(idx)
    return sample, self.new_label_map[target]


class SubsetClass(DatasetClassSampler):
  def __init__(self, dataset, idx=None, load_all_at_once=True, debug=False):
    assert isinstance(dataset, DatasetClassSampler)
    self.dataset = dataset
    self.root = dataset.root
    self.loader = dataset.loader
    self.meta = dataset.meta
    self.transform = dataset.transform
    self.transforms = dataset.transforms
    self.target_transform = dataset.target_transform
    self.samples = dataset.meta.idx_to_samples
    self.visible_classes = [dataset.visible_classes[i] for i in idx]


class SubsetInClass(DatasetClassSampler):
  def __init__(self, dataset, idx=None):
    assert isinstance(dataset, DatasetClassSampler)
    self.dataset = dataset
    self.indices = idx
    self.meta = dataset.meta
    self.samples = dataset.meta.idx_to_samples
    self.visible_classes = dataset.visible_classes

    def __getitem__(self, idx):
      return self.dataset[self.indices[idx]]

    def __len__(self):
      return len(self.indices)


class ConcatClass(DatasetClassSampler):
  def __init__(self, datasets):
    assert all([isinstance(dset, DatasetClassSampler) for dset in datasets])
    self.dataset = dataset
    self.meta = dataset.meta
    self.samples = dataset.meta.idx_to_samples
    self.visible_classes = []
    for dset in datasets:
      self.visible_classes.extend(dset.visible_classes)


def _valid_dataset(dataset):
  assert isinstance(dataset, Dataset)
  if not (hasattr(dataset, 'meta') and isinstance(dataset.meta, Metadata)):
    import pdb
    pdb.set_trace()
    raise Exception("Dataset should have attributes 'meta', instance of "
                    "datasets.metadata.Metadata.")
  else:
    return True


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


class ImageDatasetClassSampler(DatasetClassSampler):
  def __init__(self, root, transform=None, target_transform=None,
               loader=default_loader, visible_subdirs=None,
               is_valid_file=None):
    extensions = IMG_EXTENSIONS if is_valid_file is None else None
    super(ImageDatasetClassSampler, self).__init__(root=root,
                                                   loader=loader,
                                                   extensions=extensions,
                                                   transform=transform,
                                                   target_transform=target_transform,
                                                   visible_subdirs=visible_subdirs,
                                                   is_valid_file=is_valid_file)
    self.imgs = self.samples


class IterDataLoader(object):
  """Just a simple custom dataloader to load data whenever neccessary
  without forcibley using iterative loops.
  """
  def __init__(self, dataset, batch_size, sampler=None):
    self.dataset = dataset
    self.dataloader = data.DataLoader(dataset, batch_size, sampler)
    self.iterator = iter(self.dataloader)

  def __len__(self):
    return len(self.dataloader)

  def load(self, eternal=True):
    if eternal:
      try:
        return next(self.iterator)
      except StopIteration:
        self.iterator = iter(self.dataloader)
        return next(self.iterator)
      except AttributeError:
        import pdb; pdb.set_trace()
    else:
      return next(self.iterator)

  @property
  def batch_size(self):
    return self.dataloader.batch_size

  @property
  def full_size(self):
    return self.batch_size * len(self)

  def new_batchsize(self):
    self.dataloader

  @classmethod
  def from_dataset(cls, dataset, batch_size, rand_with_replace=True):
    if rand_with_replace:
      sampler = data.sampler.RandomSampler(dataset, replacement=True)
    else:
      sampler = None
    # loader = data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return cls(dataset, batch_size, sampler)

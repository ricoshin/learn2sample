import bisect
import copy
import os
import pdb
import random
import sys
from os import path

import _pickle as pickle
import numpy as np
import scipy.io as sio
from loader import loader
from torch.utils import data
from tqdm import tqdm

EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
              '.pgm', '.tif', '.tiff', '.webp', '.pt')

# scanning function
def scandir(dir):
  if sys.version_info >= (3, 5):
    return [d.name for d in os.scandir(dir) if d.is_dir()]
  else:
    return [d for d in os.listdir(dir) if path.isdir(path.join(dir, d))]


_METADATA_DEFAULT_NAME = 'Metadata'

class Metadata(object):
  def __init__(self, name=_METADATA_DEFAULT_NAME, **kwargs):
    self.name = name
    self.dict_names = []
    self._cumulative_num_samples = None
    self._add_metadata(**kwargs)

  def __len__(self):
    if hasattr(self, 'classes'):
      return len(self.classes)
    else:
      return 0

  def __repr__(self):
    return 'Metadata' + self.dict_names.__repr__() + f'(name={self.name})'

  def __getitem__(self, rel_idx):
    return self.idx_to_samples[self.abs_idx[rel_idx]]

  def __eq__(self, other):
    return set(self.classes) == set(other.classes)

  def __add__(self, other):
    return self.merge([self, other])

  def _add_metadata(self, **kwargs):
    for name, data in kwargs.items():
      try:
        setattr(self, name, data)
      except:
        import pdb
        pdb.set_trace()
      self.dict_names.append(name)

  def _del_metadata(self, names):
    for name in names:
      if name in self.dict_names:
        delattr(self, name)
        self.dict_names.remove(name)

  def _change_dname(self, from_to):
    assert isinstance(from_to, dict)
    self._add_metadata(**{from_to[n]: getattr(self, n) for n in self.dict_names
                          if n in from_to.keys()})
    self._del_metadata([v for v in from_to.keys()])

  @property
  def cumulative_num_samples(self):
    cumsum_list, cumsum = [], 0
    for idx, samples in self.idx_to_samples.items():
      cumsum += len(samples)
      cumsum_list.append(cumsum)
    return cumsum_list

  @property
  def idx_to_len(self):
    if hasattr(self, 'idx_to_samples'):
      return {k: len(v) for k, v in self.idx_to_samples.items()}
    else:
      return 0

  @property
  def idx_to_class(self):
    return dict({v: k for k, v in self.class_to_idx.items()})

  @property
  def abs_idx(self):
    return {i: v for i, v in enumerate(self.class_to_idx.values())}

  @property
  def rel_idx(self):
    return {v: i for i, v in enumerate(self.class_to_idx.values())}

  def idx_uni_to_bi(self, sample_idx):
    cumsum = self.cumulative_num_samples
    class_idx = bisect.bisect_right(cumsum, sample_idx)
    if class_idx > 0:
      sample_idx -= cumsum[class_idx - 1]
    return class_idx, sample_idx

  def idx_bi_to_uni(self, class_idx, sample_idx):
    """class_idx: relative idx"""
    cumsum = self.cumulative_num_samples
    if class_idx > 0:
      sample_idx += cumsum[class_idx - 1]
    return sample_idx

  @classmethod
  def merge(cls, others, name=_METADATA_DEFAULT_NAME):
    assert len(others) > 1
    # assert all([others[0] == other for other in others])
    classes = [set(other.classes) for other in others]
    classes = list(classes[0].union(*classes[1:]))
    # import pdb; pdb.set_trace()
    classes.sort()
    class_to_idx = dict({classes[i]: i for i in range(len(classes))})
    idx_to_class = dict({i: classes[i] for i in range(len(classes))})
    idx_to_samples = dict()
    for idx, class_ in idx_to_class.items():
      samples = []
      for other in others:
        samples.extend(other.idx_to_samples[idx])
      idx_to_samples[idx] = list(set(samples))
    return cls(classes, class_to_idx, idx_to_class, idx_to_samples, name=name)

  @classmethod
  def get_filepath(cls, root):
    return path.normpath(path.join(root, 'meta.pickle'))

  @classmethod
  def is_loadable(cls, root):
    return path.exists(cls.get_filepath(root))

  @classmethod
  def load(cls, root):
    filepath = cls.get_filepath(root)
    with open(filepath, 'rb') as f:
      meta_data = cls(**pickle.load(f))
    print(f'Loaded preprocessed dataset dictionaries: {filepath}')
    return meta_data

  def save(self, root):
    filepath = self.get_filepath(root)
    with open(filepath, 'wb') as f:
      pickle.dump({n: getattr(self, n) for n in self.dict_names}, f)
    print(f'Saved processed dataset dictionaries: {filepath}')

  @classmethod
  def new(cls, *args, name=_METADATA_DEFAULT_NAME):
    return cls(**cls._template_base(*args), name=name)

  @classmethod
  def _template_base(cls, classes, class_to_idx, idx_to_samples):
    return dict(classes=classes, class_to_idx=class_to_idx,
                idx_to_samples=idx_to_samples)

  @classmethod
  def load_or_make(cls, data_dir, remake=False, *args, **kwargs):
    if cls.is_loadable(data_dir) and not remake:
      metadata = cls.load(data_dir)
    else:
      if cls.is_loadable(data_dir) and remake:
        print("Enforced to rebuild meta data.")
      metadata = cls.make(data_dir, *args, **kwargs)
      metadata.save(data_dir)
    return metadata

  @classmethod
  def make(cls, data_dir, visible_subdirs=None, extensions=EXTENSIONS,
           is_valid_file=None):
    print("Generating metadata..")
    data_dir = path.expanduser(data_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
      raise ValueError("Both extensions and is_valid_file cannot be None "
                       "or not None at the same time")
    if extensions is not None:
      def is_valid_file(filename):
        return filename.lower().endswith(extensions)

    classes, class_to_idx = cls.scan_classes(
        data_dir, visible_subdirs, extensions)

    idx_to_samples = cls.scan_files(
        data_dir, class_to_idx, visible_subdirs, is_valid_file)

    if any([len(v) == 0 for v in idx_to_samples.values()]):
      raise (RuntimeError(
          "Found 0 files in subfolders of: " + data_dir + "\n"
          "Supported extensions are: " + ",".join(extensions)))
    return cls(**cls._template_base(classes, class_to_idx, idx_to_samples))

  @staticmethod
  def scan_classes(data_dir, visible_subdirs=None, extensions=None):
    """Scan class directories.
      Returns:
        classes (list)
        class_to_idx (OrderedDict)
        idx_to_class (OrderedDict)
    """
    subdirs = visible_subdirs if visible_subdirs else scandir(data_dir)
    classes_subdirs = []
    for subdir in subdirs:
      subdir = path.join(data_dir, subdir)
      classes = scandir(subdir)
      # deterministic shuffle to maintain consistency in data splits
      #   between multiple runs
      classes.sort()
      random.Random(1234).shuffle(classes)
      classes_subdirs.append(classes)
    print(f'Scanned sub-dirs: {subdirs}')
    any_classes = classes_subdirs[0]
    if not all([any_classes == classes_subdirs[i]
                for i in range(len(classes_subdirs))]):
      raise Exception("'train' and 'val' splits have different classes.")
    class_to_idx = dict({classes[i]: i for i in range(len(any_classes))})
    return any_classes, class_to_idx

  @staticmethod
  def scan_files(data_dir, class_to_idx, visible_subdirs=None,
                 is_valid_file=None, debug=False):
    """Scan files in each class directories.
      Returns:
        idx_to_samples (OrderedDict): This will help us to maintain class group
          information so that class-level sub-sampling can be easier.
    """
    subdirs = visible_subdirs if visible_subdirs else scandir(data_dir)
    idx_to_samples = dict()  # the order must be preserved!
    desc = 'Scanning files'
    pbar = tqdm(class_to_idx.items(), desc=desc)
    for class_, idx in pbar:
      pbar.set_description(desc + f" in {class_}")
      samples = []
      for subdir in subdirs:
        dir = path.join(data_dir, subdir, class_)
        if not path.isdir(dir):
          continue
        for base, _, fnames in sorted(os.walk(dir)):
          for fname in sorted(fnames):
            fpath = path.join(base, fname)
            if is_valid_file(fpath):
              samples.append((fpath, idx))
      idx_to_samples[idx] = samples
    return idx_to_samples

  def _select_class(self, rel_idx, name=_METADATA_DEFAULT_NAME):
    assert isinstance(rel_idx, (list, tuple))
    classes = []
    class_to_idx = dict()
    idx_to_samples = dict()
    for i in rel_idx:
      abs_idx = self.abs_idx[i]
      class_ = self.classes[i]
      classes.append(class_)
      class_to_idx[abs_idx] = self.class_to_idx[class_]
      idx_to_samples[abs_idx] = self.idx_to_samples[abs_idx]
    return self.new(
      class_to_idx, class_to_idx, idx_to_samples, name=_METADATA_DEFAULT_NAME)

  def sample_classes(self, num, name=_METADATA_DEFAULT_NAME):
    sampled_idx = np.random.choice(
        len(self), num, replace=False).tolist()
    return self._select_class(sampled_idx, name=name)

  def sample_instances(self, num, name=_METADATA_DEFAULT_NAME):
    meta = copy.deepcopy(self)
    idx_to_samples = dict()
    for class_idx, samples in self.idx_to_samples.items():
      sampled_idx = np.random.choice(len(samples), num, replace=False).tolist()
      idx_to_samples[class_idx] = [samples[i] for i in sampled_idx]
    meta.idx_to_samples = idx_to_samples
    meta.name = name
    return meta

  def _get_split_name_and_ratio(self, ratio):
    if isinstance(ratio[0], (list,tuple)):
      for i in range(len(ratio)):
        assert len(ratio[i]) == 2
        assert isinstance(ratio[i][0], str)
        assert isinstance(ratio[i][1], (int, float))
      names, ratio = list(zip(*ratio))
    else:
      for i in range(len(ratio)):
        assert isinstance(ratio[i], (int, float))
      names = None
    ratio = [r / sum(ratio) for r in ratio]
    assert all([0. < r < 1. for r in ratio])
    return names, ratio

  def split_classes(self, ratio, shuffle=True):
    """Usage:
      meta_data = Metadata(...)
      # 1 : 4 support/query split
      support, query = meta_data.split_classes(
        (('Support', 0.1), ('Query', 0.4))
      )
    """
    assert isinstance(ratio, (list, tuple))
    assert isinstance(shuffle, bool)
    names, ratio = self._get_split_name_and_ratio(ratio)
    class_idx = list(range(len(self)))
    if shuffle:
      random.shuffle(class_idx)
    prev_thres = 0
    meta_data_list = []
    for i, r in enumerate(ratio):
      thres = int(len(self) * r)
      meta_data = copy.deepcopy(self)
      meta_data = meta_data._select_class(class_idx[prev_thres:thres])
      if names:
        meta_data.name = names[i]
      meta_data_list.append(meta_data)
      prev_thres = thres
    return meta_data_list

  def split_instances(self, ratio, shuffle=True):
    """Usage:
      meta_data = Metadata(...)
      # 1 : 4 support/query split
      support, query = meta_data.split_instances(
        (('Support', 1), ('Query', 4))
      )
    """
    assert isinstance(ratio, (list, tuple))
    assert isinstance(shuffle, bool)
    names, ratio = self._get_split_name_and_ratio(ratio)
    idx_to_samples_all = [{} for i in range(len(ratio))]
    for class_idx, samples in self.idx_to_samples.items():
      len_ = len(samples)
      sample_idx = list(range(len_))
      if shuffle:
        random.shuffle(sample_idx)
      prev_thres = 0
      for i, r in enumerate(ratio):
        thres = int(len_ * r)
        sampled_idx = sample_idx[prev_thres:thres]
        idx_to_samples_all[i][class_idx] = [samples[i] for i in sampled_idx]
        prev_thres = thres
    meta_data_list = []
    for i, idx_to_samples in enumerate(idx_to_samples_all):
      meta_data = copy.deepcopy(self)
      meta_data.idx_to_samples = idx_to_samples
      if names:
        meta_data.name = names[i]
      meta_data_list.append(meta_data)
    return meta_data_list

  def dataset_loader(self, loader_config):
    assert isinstance(loader_config, loader.LoaderConfig)
    return loader_config.get_dataset_loader(self, self.name)

  def episode_loader(self, loader_config):
    assert isinstance(loader_config, loader.LoaderConfig)
    return loader_config.get_episode_loader(self, self.name)


class ImagenetMetadata(Metadata):
  @property
  def abs_idx(self):
    return {i: v for i, v in enumerate(self.wnid_to_idx.values())}

  @property
  def rel_idx(self):
    return {v: i for i, v in enumerate(self.wnid_to_idx.values())}

  @property
  def idx_to_wnid(self):
    return {v: k for k, v in self.wnid_to_idx.items()}

  @property
  def wnid_to_class(self):
    return {v: k for k, v in self.class_to_wnid.items()}

  def to_imagenet(self, classes, class_to_wnid):
    # class -> wninds
    from_to = {
        'classes': 'wnids',
        'class_to_idx': 'wnid_to_idx',
    }
    self._change_dname(from_to)
    self._add_metadata(**self._template_wordnet(classes, class_to_wnid))
    return self

  @classmethod
  def load_or_make(cls, data_dir, devkit_dir=None, remake=False,
                   *args, **kwargs):
    if cls.is_loadable(data_dir) and not remake:
      metadata = cls.load(data_dir)
    else:
      assert devkit_dir is not None
      if cls.is_loadable(data_dir) and remake:
        print("Enforced to rebuild meta data.")
      metadata = cls.make(data_dir, devkit_dir, *args, **kwargs)
      metadata.save(data_dir)
    return metadata

  @classmethod
  def make(cls, data_dir, devkit_dir, *args, **kwargs):
    metadata = super(ImagenetMetadata, cls).make(data_dir, *args, **kwargs)
    return metadata.to_imagenet(
        *cls.scan_imagenet_devkit(data_dir, devkit_dir))

  @classmethod
  def new(cls, *args, name=_METADATA_DEFAULT_NAME):
    return cls(**cls._template_imagenet(*args), name=name)

  @classmethod
  def _template_wordnet(cls, classes, class_to_wnid):
    return dict(classes=classes, class_to_wnid=class_to_wnid)

  @classmethod
  def _template_imagenet(cls, wnids, wnid_to_idx, idx_to_samples,
                         classes, class_to_wnid):
    return dict(wnids=wnids, wnid_to_idx=wnid_to_idx,
                idx_to_samples=idx_to_samples, classes=classes,
                class_to_wnid=class_to_wnid)

  @staticmethod
  def scan_imagenet_devkit(data_dir, devkit_dir):
    devkit_dir = path.join(data_dir, devkit_dir)
    # load mat 'data/meta.mat'
    mat_fpath = path.join(devkit_dir, 'data', 'meta.mat')
    meta = sio.loadmat(mat_fpath, squeeze_me=True)['synsets']
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children)
            if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(', ')) for clss in classes]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
    # wnid_to_class = {wnid: clss for wnid, clss in zip(wnids, classes)}
    class_to_wnid = {clss: wnid for wnid, clss in zip(wnids, classes)}
    # load 'data/ILSVRC2012_validation_ground_truth.txt'
    val_gt_fname = 'ILSVRC2012_validation_ground_truth.txt'
    val_gt_fpath = path.join(devkit_dir, 'data', val_gt_fname)
    with open(val_gt_fpath, 'r') as f:
      val_idcs = f.readlines()
    val_idcs = [int(val_idx) for val_idx in val_idcs]
    val_wnids = [idx_to_wnid[idx] for idx in val_idcs]
    return classes, class_to_wnid

  def _select_class(self, rel_idx, name=_METADATA_DEFAULT_NAME):
    assert isinstance(rel_idx, (list, tuple))
    classes = []
    wnids = []
    wnid_to_idx = dict()
    class_to_wnid = dict()
    idx_to_samples = dict()
    for i in rel_idx:
      abs_idx = self.abs_idx[i]
      wnid = self.wnids[i]
      class_ = self.classes[i]
      wnids.append(wnid)
      classes.append(class_)
      wnid_to_idx[wnid] = self.wnid_to_idx[wnid]
      class_to_wnid[class_] = self.class_to_wnid[class_]
      idx_to_samples[abs_idx] = self.idx_to_samples[abs_idx]
    return self.new(
      wnids, wnid_to_idx, idx_to_samples, classes, class_to_wnid, name=name)

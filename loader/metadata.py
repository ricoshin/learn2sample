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


class Metadata(object):
  def __init__(self, **kwargs):
    self.dnames = []
    self._cumulative_num_samples = None
    self._add_metadata(**kwargs)

  def __len__(self):
    if hasattr(self, 'classes'):
      return len(self.classes)
    else:
      return 0

  def __repr__(self):
    return 'Metadata' + self.dnames.__repr__()

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
      self.dnames.append(name)

  def _del_metadata(self, names):
    for name in names:
      if name in self.dnames:
        delattr(self, name)
        self.dnames.remove(name)

  def _change_dname(self, from_to):
    assert isinstance(from_to, dict)
    self._add_metadata(**{from_to[n]: getattr(self, n) for n in self.dnames
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
  def merge(cls, others):
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
    return cls(classes, class_to_idx, idx_to_class, idx_to_samples)

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
      pickle.dump({n: getattr(self, n) for n in self.dnames}, f)
    print(f'Saved processed dataset dictionaries: {filepath}')

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

  def _select_class(self, rel_idx):
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
    return self.new(class_to_idx, class_to_idx, idx_to_samples)

  def sample_classes(self, num):
    sampled_idx = np.random.choice(
        len(self), num, replace=False).tolist()
    return self._select_class(sampled_idx)

  def split_classes(self, ratio, shuffle=True):
    assert 0. < ratio < 1.
    meta_a = copy.deepcopy(self)
    meta_b = copy.deepcopy(self)
    class_idx = list(range(len(self)))
    if shuffle:
      random.shuffle(class_idx)
    thres = int(len(self) * ratio)
    return (meta_a._select_class(class_idx[:thres]),
            meta_b._select_class(class_idx[thres:]))

  def sample_instances(self, num):
    meta = copy.deepcopy(self)
    idx_to_samples = dict()
    for class_idx, samples in self.idx_to_samples.items():
      sampled_idx = np.random.choice(len(samples), num, replace=False).tolist()
      idx_to_samples[class_idx] = [samples[i] for i in sampled_idx]
    meta.idx_to_samples = idx_to_samples
    return meta

  def split_instances(self, ratio, shuffle=True):
    assert 0. < ratio < 1.
    meta_a = copy.deepcopy(self)
    meta_b = copy.deepcopy(self)
    idx_to_samples_a = dict()
    idx_to_samples_b = dict()
    for class_idx, samples in self.idx_to_samples.items():
      len_ = len(samples)
      inst_idx = list(range(len_))
      if shuffle:
        random.shuffle(inst_idx)
      thres = int(len_ * ratio)
      # import pdb; pdb.set_trace()
      idx_to_samples_a[class_idx] = [samples[i] for i in inst_idx[:thres]]
      idx_to_samples_b[class_idx] = [samples[i] for i in inst_idx[thres:]]
    meta_a.idx_to_samples = idx_to_samples_a
    meta_b.idx_to_samples = idx_to_samples_b
    return meta_a, meta_b


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
  def new(cls, *args):
    return cls(**cls._template_imagenet(*args))

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

  def _select_class(self, rel_idx):
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
    return self.new(wnids, wnid_to_idx, idx_to_samples, classes, class_to_wnid)

from __future__ import print_function

import os
import shutil
from os import path

import torch
from datasets.datasets import ImageDatasetClassSampler
from datasets.metadata import Metadata
from torchvision.datasets.utils import check_integrity, download_url

ARCHIVE_DICT = {
    'train': {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/'
        'ILSVRC2012_img_train.tar',
        'md5': '1d675b47d978889d74fa0da5fadfb00e',
    },
    'val': {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/'
        'ILSVRC2012_img_val.tar',
        'md5': '29b22e2961454d5413ddabcf34fc5622',
    },
    'devkit': {
        'url': 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/'
        'ILSVRC2012_devkit_t12.tar.gz',
        'md5': 'fa75699e90414af021442c21a62c3abf',
    }
}


class ImageNet(ImageDatasetClassSampler):
  """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

  Args:
    root (string): Root directory of the ImageNet Dataset.
    split (string, optional): The dataset split, supports ``train``, or ``val``.
    download (bool, optional): If true, downloads the dataset from the
      internet and puts it in root directory. If dataset is already
      downloaded, it is not downloaded again.
    transform (callable, optional): A function/transform that  takes in an PIL
      image and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in
      the target and transforms it.
    loader (callable, optional): A function to load an image given its path.

  Attributes:
    classes (list): List of the class names.
    class_to_idx (dict): Dict with items (class_name, class_index).
    wnids (list): List of the WordNet IDs.
    wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
    imgs (list): List of (image path, class_index) tuples
    targets (list): The class_index value for each image in the dataset
  """

  def __init__(self, root, splits=None, download=False,
               rebuild_metadata=True, **kwargs):
    root = self.root = os.path.expanduser(root)
    if splits is None:
      splits = self.valid_splits
    else:
      splits = self._verify_splits(splits)
    self.splits = splits
    # self.split = self._verify_split(split)

    if download:
      self.download()

    # metadata = Metadata.load_or_make(
    #     meta_dir=path.join(root, os.pardir),
    #     name=path.basename(root),
    #     remake=rebuild_metadata,
    #     data_dir=root,
    #     extensions=extensions,
    #     is_valid_file=is_valid_file
    # ).to_imagenet(self._load_devkit_metafile()[:-1])

    super(ImageNet, self).__init__(root, visible_subdirs=splits, **kwargs)
    self.root = root  # FIX LATER: to recover the original path
    self.meta.to_imagenet(*self._load_devkit_metafile()[:-1])

    # self.meta.wnids = self.meta.classes
    # self.meta.wnid_to_idx = self.meta.class_to_idx
    # self.meta.idx_to_wnid = self.meta.idx_to_class
    # self.meta.classes = [self.meta.wnid_to_class[wnid]
    #                      for wnid in self.meta.wnids]
    # self.meta.class_to_idx = {cls: idx
    #   for idx, clss in enumerate(self.meta.classes) for cls in clss}
    # self.meta.idx_to_class = {idx: cls
    #   for idx, clss in enumerate(self.meta.classes) for cls in clss}

  def download(self):
    if not os.path.isfile(self.devkit_metafile):
      archive_dict = ARCHIVE_DICT['devkit']
      devkit_folder = _splitexts(os.path.basename(archive_dict['url']))[0]
      devkit_dir = path.join(self.root, devkit_folder)
      if not path.isdir(devkit_dir):
        download_and_extract_tar(archive_dict['url'], self.root,
                                 extract_root=self.root,
                                 md5=archive_dict['md5'])
      meta = parse_devkit(os.path.join(self.root, devkit_folder))
      self._save_devkit_metafile(*meta)
      # shutil.rmtree(tmpdir)

    for split in self.splits:
      split_path = path.join(self.root, split)
      if not os.path.isdir(split_path):
        archive_dict = ARCHIVE_DICT[split]
        download_and_extract_tar(archive_dict['url'], self.root,
                                 extract_root=split_path,
                                 md5=archive_dict['md5'])

        if split == 'train':
          prepare_train_folder(split_path)
        elif split == 'val':
          val_wnids = self._load_devkit_metafile()[-1]
          prepare_val_folder(split_path, val_wnids)
      else:
        print(f"Found downloaded dataset: '{split_path}'")

  @property
  def devkit_metafile(self):
    return os.path.join(self.root, 'meta.bin')

  def _load_devkit_metafile(self):
    if check_integrity(self.devkit_metafile):
      devkit_metafile = torch.load(self.devkit_metafile)
      print(f'Loaded devkit metafile: {self.devkit_metafile}')
      return devkit_metafile
    else:
      raise RuntimeError("Meta file not found or corrupted.",
                         "You can use download=True to create it.")

  def _save_devkit_metafile(self, *args):
    torch.save(args, self.devkit_metafile)
    print(f'Saved metafile: {self.devkit_metafile}')

  def _verify_splits(self, splits):
    assert len(splits) > 0
    for split in splits:
      if split not in self.valid_splits:
        msg = "Unknown data split {} .".format(split)
        msg += "Valid splits are {{}}.".format(", ".join(self.valid_splits))
        raise ValueError(msg)
    return splits

  @property
  def valid_splits(self):
    return ('train', 'val')

  def extra_repr(self):
    return "Splits: {splits}".format(**self.__dict__)


def extract_tar(src, dest=None, gzip=None, delete=False):
  import tarfile

  if dest is None:
    dest = os.path.dirname(src)
  if gzip is None:
    gzip = src.lower().endswith('.gz')

  mode = 'r:gz' if gzip else 'r'
  with tarfile.open(src, mode) as tarfh:
    tarfh.extractall(path=dest)

  if delete:
    os.remove(src)


def download_and_extract_tar(url, download_root, extract_root=None,
                             filename=None, md5=None, **kwargs):
  download_root = os.path.expanduser(download_root)
  if extract_root is None:
    extract_root = download_root
  if filename is None:
    filename = os.path.basename(url)

  if not check_integrity(os.path.join(download_root, filename), md5):
    download_url(url, download_root, filename=filename, md5=md5)

  extract_tar(os.path.join(download_root, filename), extract_root, **kwargs)


def parse_devkit(root):
  classes, wnid_to_classes, class_to_wnid, idx_to_wnid = parse_meta(root)
  val_idcs = parse_val_groundtruth(root)
  val_wnids = [idx_to_wnid[idx] for idx in val_idcs]
  return classes, wnid_to_classes, class_to_wnid, val_wnids


def parse_meta(devkit_root, path='data', filename='meta.mat'):
  import scipy.io as sio
  metafile = os.path.join(devkit_root, path, filename)
  meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
  nums_children = list(zip(*meta))[4]
  meta = [meta[idx] for idx, num_children in enumerate(nums_children)
          if num_children == 0]
  idcs, wnids, classes = list(zip(*meta))[:3]
  classes = [tuple(clss.split(', ')) for clss in classes]
  idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
  wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
  class_to_wnid = {clss: wnid for wnid, clss in zip(wnids, classes)}
  return classes, wnid_to_classes, class_to_wnid, idx_to_wnid


def parse_val_groundtruth(devkit_root, path='data',
                          filename='ILSVRC2012_validation_ground_truth.txt'):
  with open(os.path.join(devkit_root, path, filename), 'r') as txtfh:
    val_idcs = txtfh.readlines()
  return [int(val_idx) for val_idx in val_idcs]


def prepare_train_folder(folder):
  for archive in [os.path.join(folder, archive)
                  for archive in os.listdir(folder)]:
    extract_tar(archive, os.path.splitext(archive)[0], delete=True)


def prepare_val_folder(folder, wnids):
  img_files = sorted([os.path.join(folder, file)
                      for file in os.listdir(folder)])

  for wnid in set(wnids):
    os.mkdir(os.path.join(folder, wnid))

  for wnid, img_file in zip(wnids, img_files):
    shutil.move(img_file, os.path.join(
        folder, wnid, os.path.basename(img_file)))


def _splitexts(root):
  exts = []
  ext = '.'
  while ext:
    root, ext = os.path.splitext(root)
    exts.append(ext)
  return root, ''.join(reversed(exts))

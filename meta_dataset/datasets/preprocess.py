"""Preprocess ImageNet 1K in advance, in order to avoid transform
overhead one would have amortized at each batch collection using dataloader.

Recommend you to install Pillow-SIMD first.
  $ pip uninstall pillow
  $ CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
"""
import multiprocessing as mp
import os
import sys
from os import path
from time import sleep

import PIL
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

RESIZE = (32, 32)
IMAGENET_DIR = '/v9/whshin/imagenet'
# VISIBLE_SUBDIRS = ['train', 'val']
VISIBLE_SUBDIRS = ['val']
NEW_DIR_POSTFIX = 'test'
PASS_IF_EXIST = True
DEBUG = False
MAX_N_PROCESS = 9999

RESIZE_FILTER = {
    0: Image.NEAREST,
    1: Image.BILINEAR,
    2: Image.BICUBIC,
    3: Image.ANTIALIAS,
}[3]

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def is_image_file(filepath):
  return filepath.lower().endswith(IMG_EXTENSIONS)


def chunkify(list_, n):
  return [[list_[i::n], i] for i in range(n)]


def scandir(supdir, subdirs):
  image_paths = []
  for subdir in subdirs:
    dir = path.join(supdir, subdir)
    print(f'Scanning subdir: {subdir}')
    if sys.version_info >= (3, 5):
      # Faster and available in Python 3.5 and above
      classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
      classes = [d for d in os.listdir(dir)
                 if path.isdir(path.join(dir, d))]

    for class_ in tqdm(classes):
      d = path.join(dir, class_)
      if not path.isdir(d):
        continue  # dir only
      for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
          image_path = path.join(root, fname)
          if is_image_file(image_path):
            image_paths.append(image_path)

  return image_paths


def split_path(filepath):
  """Split full filepath into 4 different chunks.
    Args:
      filepath (str): "/../imagenet/train/n01632777/n01632777_1000.JPEG"

    Returns:
      base (str): '..'
      dataset (str): 'imagenet'
      subdirs (str): 'train/n01632777'
      filename (str): 'n01632777_1000.JPEG'
  """
  splits = filepath.split('/')
  filename = splits[-1]
  subdirs = path.join(*splits[-3:-1])
  dataset = splits[-4]
  base = path.join('/', *splits[:-4])
  return base, dataset, subdirs, filename


def process(chunk):
  filepaths, i = chunk

  composed_transforms = transforms.Compose([
      # transforms.RandomResizedCrop(32),
      # transforms.RandomHorizontalFlip(0.5),
      transforms.Resize(RESIZE, interpolation=RESIZE_FILTER),
      # transforms.ToTensor(),
  ])

  saved = 0
  passed = 0
  error_msgs = []
  process_desc = '[Process #%2d] ' % i

  for filepath in tqdm(filepaths, desc=process_desc, position=i):
    # make new path ready
    base, dataset, subdirs, filename = split_path(filepath)
    dataset_new = "_".join([dataset, NEW_DIR_POSTFIX, *map(str, RESIZE)])
    classpath_new = path.join(base, dataset_new, subdirs)
    os.makedirs(classpath_new, exist_ok=True)
    filepath_new = path.join(classpath_new, filename)
    # save resized image
    if os.path.exists(filepath_new) and PASS_IF_EXIST:
      passed += 1
      continue  # or not
    # processing part
    try:
      transform_image(
          path_old=filepath,
          path_new=filepath_new,
          transforms=composed_transforms,
      )
      saved += 1
    except Exception as e:
      error_msgs.append(f"[{filepath}: {e})")

  return (saved, passed, error_msgs)


def transform_image(path_old, path_new, transforms=None):
  with Image.open(path_old) as img:
    img = img.convert("RGB")  # there are a few RGBA images
    img = transforms(img)
    if isinstance(img, Image.Image):
      img.save(path_new)
    elif isinstance(img, torch.Tensor):
      with open(path_new, 'wb') as f:
        torch.save(path_new)


def make_dataset(dir, class_to_idx):
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


def make_metadata(self):
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


def load_metadata(metapath, basename):
  if Metadata.is_loadable(metapath, basename):
    metadata = Metadata.load(metapath, basename)
  else:
    metadata = self.make_metadata(extensions, is_valid_file)
    if load_metadata:
      metadata.save(metafile, basename)


def run():
  print("Image preprocessing with multi-workers.")
  print(f"RESIZE: {RESIZE}")
  print(f"IMAGENET_DIR: {IMAGENET_DIR}")
  print(f"VISIBLE_SUB_DIRS: {VISIBLE_SUBDIRS}")
  print(f"DEBUG_MODE: {'ON' if DEBUG else 'OFF'}")

  print(f'\nScanning dirs..')
  subdirs = ['val'] if DEBUG else VISIBLE_SUBDIRS
  paths = scandir(IMAGENET_DIR, subdirs)
  n_images = len(paths)
  print(f'Done. {n_images} images are found.')

  num_process = 1 if DEBUG else min(mp.cpu_count(), MAX_N_PROCESS)
  # num_process = mp.cpu_count()
  chunks = chunkify(paths, num_process)

  if DEBUG:
    print('Start single processing for debugging.')
    results = [process(chunks[0])]  # for debugging
  else:
    print(f'Start {num_process} processes.')
    pool = mp.Pool(processes=num_process)
    results = pool.map(process, chunks)
    pool.close()  # no more task
    pool.join()  # wrap up current tasks
    print("Preprocessing completed.")

  saved_total = 0
  passed_total = 0
  error_msgs_total = []
  for saved, passed, error_msgs in results:
    saved_total += saved
    passed_total += passed
    error_msgs_total.extend(error_msgs)
  print(f"[!] {saved_total} saved file(s) / "
        f"{passed_total} ignored (already exist) file(s) / "
        f"{len(error_msgs_total)} error(s).")

  # log errors
  logfile_name = 'errors.txt'
  base = path.normpath(path.join(IMAGENET_DIR, os.pardir))
  dataset = path.basename(IMAGENET_DIR)
  dataset = "_".join([dataset, 'resize', *map(str, RESIZE)])
  logfile_path = path.join(base, dataset, logfile_name)
  with open(logfile_path, 'w') as f:
    for i, error_msg in enumerate(error_msgs_total):
      f.write(f'[Error {i}] {error_msg}\n')
  print(f"Error messages logged in {logfile_path}. "
        "Top 10 lines are as follows:")
  os.system(f'head -n 10 {logfile_path}')
  print('\n' * num_process)


if __name__ == '__main__':
  run()

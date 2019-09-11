"""Preprocess ImageNet 1K in advance, in order to avoid transform
overhead one would have amortized at each batch collection using dataloader.

Recommend you to install Pillow-SIMD first.
  $ pip uninstall pillow
  $ CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
"""
import multiprocessing as mp
import os
import sys
import time
from collections import OrderedDict as odict
from os import path
from time import sleep

import PIL
import torch
from metadata import Metadata
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

RESIZE = (32, 32)
IMAGENET_DIR = '/v9/whshin/imagenet_resized_32_32'
VISIBLE_SUBDIRS = ['train', 'val']
# VISIBLE_SUBDIRS = [val']
SAVE_IMAGE = False
SAVE_BIN = True
REBUILD_METADATA = False
NEW_DIR_POSTFIX = 'test'
PASS_IF_EXIST = False
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

assert (SAVE_IMAGE is False) ^ (SAVE_BIN is False)


def is_image_file(filepath):
  return filepath.lower().endswith(IMG_EXTENSIONS)


def chunkify(list_, n):
  return [list_[i::n] for i in range(n)]


def chunkify_classes(meta, n):
  idx_chunks = chunkify(list(meta.idx_to_samples.keys()), n)
  chunks = []
  for k, idx in enumerate(idx_chunks):
    chunks.append([odict({i: meta.idx_to_samples[i] for i in idx}), k])
  return chunks


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
  idx_to_samples, i = chunk
  composed_transforms = transforms.Compose([
      # transforms.RandomResizedCrop(32),
      # transforms.RandomHorizontalFlip(0.5),
      # transforms.Resize(RESIZE, interpolation=RESIZE_FILTER),
      # transforms.ToTensor(),
  ])
  to_tensor = transforms.ToTensor()
  saved = 0
  passed = 0
  error_msgs = []
  total = sum([len(v) for v in idx_to_samples.values()])
  pbar = tqdm(desc='[Process #%2d] ' % i, position=i, total=total)
  for _, samples in idx_to_samples.items():
    tensors = []
    for filepath, idx in samples:
      pbar.update(1)
      # make new path ready
      base, dataset, subdirs, filename = split_path(filepath)
      dataset_new = "_".join([dataset, NEW_DIR_POSTFIX, *map(str, RESIZE)])
      classpath_new = path.join(base, dataset_new, subdirs)

      filepath_new = path.join(classpath_new, filename)
      # save resized image
      if os.path.exists(filepath_new) and PASS_IF_EXIST:
        passed += 1
        continue  # or not
      # processing part
      try:
        image = transform_image(
            path_old=filepath,
            path_new=filepath_new,
            transforms=composed_transforms,
            save_image=SAVE_IMAGE,
        )
        saved += 1
      except Exception as e:
        error_msgs.append(f"[{filepath}: {e})")

      if SAVE_BIN:
        tensors.append(to_tensor((image, idx))

    if SAVE_BIN:
      tensors = torch.stack(tensors)
      classname = path.basename(subdirs)
      bin_path = path.join(base, dataset_new, 'bin')
      os.makedirs(bin_path, exist_ok=True)
      with open(path.join(bin_path, classname + '.pt'), 'wb') as f:
        torch.save(tensors, f)

  return (saved, passed, error_msgs)


def transform_image(path_old, path_new, transforms=None, save_image=True):
  with Image.open(path_old) as img:
    img = img.convert("RGB")  # there are a few RGBA images
    img = transforms(img)
    if save_image:
      os.makedirs(path_new, exist_ok=True)
      if isinstance(img, Image.Image):
        img.save(path_new)
      elif isinstance(img, torch.Tensor):
        torch.save(path_new)
  return img


def run():
  print("Image preprocessing with multi-workers.")
  print(f"RESIZE: {RESIZE}")
  print(f"IMAGENET_DIR: {IMAGENET_DIR}")
  print(f"VISIBLE_SUB_DIRS: {VISIBLE_SUBDIRS}")
  print(f"DEBUG_MODE: {'ON' if DEBUG else 'OFF'}")

  print("Warming up tqdm.")

  for _ in tqdm(range(10)):
    time.sleep(0.1)

  print(f'\nScanning dirs..')
  meta = Metadata.load_or_make(
      # meta_dir=path.join(root, os.pardir),
      remake=REBUILD_METADATA,
      data_dir=IMAGENET_DIR,
      visible_subdirs=['val'] if DEBUG else VISIBLE_SUBDIRS,
      extensions=IMG_EXTENSIONS,
  )

  n_classes = len([v for v in meta.idx_to_samples])
  n_images = sum([len(v) for v in meta.idx_to_samples.values()])
  print(f'Done. {n_classes} classes and {n_images} images are found.')

  num_process = 1 if DEBUG else min(mp.cpu_count(), MAX_N_PROCESS)
  # num_process = mp.cpu_count()
  chunks = chunkify_classes(meta, num_process)

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
  print('\n' * num_process)

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
  dataset = "_".join([dataset, NEW_DIR_POSTFIX, *map(str, RESIZE)])
  logfile_path = path.join(base, dataset, logfile_name)
  with open(logfile_path, 'w') as f:
    for i, error_msg in enumerate(error_msgs_total):
      f.write(f'[Error {i}] {error_msg}\n')
  print(f"Error messages logged in {logfile_path}. "
        "Top 10 lines are as follows:")
  os.system(f'head -n 10 {logfile_path}')


if __name__ == '__main__':
  run()

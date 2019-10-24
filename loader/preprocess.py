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
from metadata import ImagenetMetadata
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

RESIZE = (84, 84)
IMAGENET_DIR = '/v9/whshin/imagenet'
DEVKIT_DIR = '/v9/whshin/imagenet/ILSVRC2012_devkit_t12'
VISIBLE_SUBDIRS = ['train', 'val']
# VISIBLE_SUBDIRS = ['val']
SAVE_IMAGE = True
SAVE_IMAGE_AS_TENSOR = True
SAVE_CLASS = False
REBUILD_METADATA = True
NEW_DIR_POSTFIX = 'l2s'
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
                  '.pgm', '.tif', '.tiff', '.webp', '.pt')

assert (SAVE_IMAGE is False) ^ (SAVE_CLASS is False)


def is_image_file(filepath):
  return filepath.lower().endswith(IMG_EXTENSIONS)


def new_dataset_path():
  base = path.normpath(path.join(IMAGENET_DIR, os.pardir))
  dataset_name = path.basename(IMAGENET_DIR)
  dataset_name = "_".join([dataset_name, NEW_DIR_POSTFIX, *map(str, RESIZE)])
  return path.join(base, dataset_name)


def change_ext(filepath, ext):
  assert isinstance(ext, str)
  base = path.normpath(path.join(filepath, os.pardir))
  filename = '.'.join(path.basename(filepath).split('.')[:-1] + [ext])
  return path.join(base, filename)


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
  transforms_ = [
    # transforms.RandomResizedCrop(32),
    # transforms.RandomHorizontalFlip(0.5),
    transforms.Resize(RESIZE, interpolation=RESIZE_FILTER),
  ]
  to_tensor = transforms.ToTensor()
  if SAVE_IMAGE_AS_TENSOR:
    transforms_ += [to_tensor]
  composed_transforms = transforms.Compose(transforms_)
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
      _, _, subdirs, filename = split_path(filepath)
      classpath_new = path.join(new_dataset_path(), subdirs)
      filepath_new = path.join(classpath_new, filename)
      if SAVE_IMAGE_AS_TENSOR:
        filepath_new = change_ext(filepath_new, 'pt')
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

      if SAVE_CLASS:
        tensors.append(to_tensor(image))

    if SAVE_CLASS:
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
      os.makedirs(path.normpath(path.join(path_new, os.pardir)), exist_ok=True)
      if SAVE_IMAGE_AS_TENSOR:
        # assert isinstance(img, torch.Tensor)
        with open(path.join(path_new), 'wb') as f:
          torch.save(img, f)
      else:
        # assert isinstance(img, Image.Image)
        img.save(path_new)
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

  meta = ImagenetMetadata.load_or_make(
      meta_dir=IMAGENET_DIR,
      remake=REBUILD_METADATA,
      data_dir=IMAGENET_DIR,
      visible_subdirs=['val'] if DEBUG else VISIBLE_SUBDIRS,
      extensions=IMG_EXTENSIONS,
      devkit_dir=DEVKIT_DIR,
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

  ImagenetMetadata.make(
      data_dir=new_dataset_path(),
      visible_subdirs=['val'] if DEBUG else VISIBLE_SUBDIRS,
      extensions=IMG_EXTENSIONS,
      devkit_dir=DEVKIT_DIR,
  ).save(new_dataset_path())

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

  # save error log
  logfile_name = 'errors.txt'
  logfile_path = path.join(new_dataset_path(), logfile_name)
  with open(logfile_path, 'w') as f:
    for i, error_msg in enumerate(error_msgs_total):
      f.write(f'[Error {i}] {error_msg}\n')
  print(f"Error messages logged in {logfile_path}.")
  if len(error_msgs_total) > 0:
    print("Top 10 lines are as follows:")
    os.system(f'head -n 10 {logfile_path}')


if __name__ == '__main__':
  run()

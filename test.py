import pdb
import random

import numpy as np
from torch.utils import data
from loader.metadata import ImagenetMetadata
from loader.loader import DataFromMetadata, ClassBalancedSampler

IMAGENET_DIR = '/v9/whshin/imagenet_l2s_84_84'
DEVKIT_DIR = '/v9/whshin/imagenet/ILSVRC2012_devkit_t12'


def split_meta(meta, mode):
  if mode == 'class':
    # [episode] i
    s, q = meta.split_classes(0.3)  # meta-s : meta_q = 30 : 70
    #   [meta_train]  j
    s_load = s.get_loader(batch_size=128)
    import pdb; pdb.set_trace()
    # s = s.sample_classes(0.5)  # 35 x 10 = 350


  elif mode == 'instance':
    s, q = meta.split_instances(0.7)
  ss, sq = s.split_instances(0.7)
  qs, qq = q.split_instances(0.7)
  print(f'Split mode: {mode}')
  print('Number of classes:')
  print(f'\ts({len(s)}), q({len(q)})')
  print(f'\tss({len(ss)}), qs({len(qs)})')
  print(f'\tsq({len(sq)}), qq({len(qq)})')
  print('Number of samples in the first class:')
  print(f'\ts({s.idx_to_len[s.abs_idx[0]]}), '
        f'q({q.idx_to_len[q.abs_idx[0]]})')
  print(f'\tss({ss.idx_to_len[ss.abs_idx[0]]}), '
        f'qs({qs.idx_to_len[qs.abs_idx[0]]})')
  print(f'\tsq({sq.idx_to_len[sq.abs_idx[0]]}), '
        f'qq({qq.idx_to_len[qq.abs_idx[0]]})')
  return s, q, ss, sq


split = 'class'
random.seed(1)
np.random.seed(1)
meta = ImagenetMetadata.load_or_make(
  data_dir=IMAGENET_DIR, devkit_dir=DEVKIT_DIR, remake=False)
m_train, m_test = meta.split_classes(0.1)
# s, q, ss, sq = split_meta(m_train, 'instance')
s, q, ss, sq = split_meta(m_train, 'class')





import pdb; pdb.set_trace()

pdb.set_trace()
print('end')

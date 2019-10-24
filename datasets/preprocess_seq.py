"""loads imagenet and writes it into one massive binary file.
https://github.com/BayesWatch/sequential-imagenet-dataloader/blob/master/
preprocess_sequential.py
"""

IMAGENET_DIR = '/v9/whshin/imagenet_resized_32_32'

import os
import numpy as np
from tensorpack.dataflow import *

if __name__ == '__main__':
    class BinaryILSVRC12(dataset.ILSVRC12Files):
        def get_data(self):
            for fname, label in super(BinaryILSVRC12, self).get_data():
                with open(fname, 'rb') as f:
                    jpeg = f.read()
                jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
                yield [jpeg, label]
    imagenet_path = os.environ['IMAGENET']

    for name in ['train', 'val']:
        ds0 = BinaryILSVRC12(imagenet_path, name)
        ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
        filapath = os.path.join(imagenet_path,'ILSVRC-%s.lmdb'%name)
        dftools.dump_dataflow_to_lmdb(ds1, filepath)

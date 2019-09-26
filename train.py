import argparse
import os
import pdb

import gin
import torch
from loop import loop
from networks.sampler import Sampler
from torch.utils.tensorboard import SummaryWriter
from utils import utils
from utils.utils import prepare_dir

C = utils.getCudaManager('default')

parser = argparse.ArgumentParser(description='Learning to sample')
parser.add_argument('--cpu', action='store_true', help='disable CUDA')
parser.add_argument('--volatile', action='store_true', help='no saved files.')
parser.add_argument('--gin', type=str, default='omniglot',
                    help='gin filename to load configuration.')


@gin.configurable
def meta_train(train_loop, valid_loop, test_loop, meta_epoch, tolerance,
               save_path):
  writer = SummaryWriter(os.path.join(save_path, 'tfevent'))
  no_improvement = 0
  best_acc = 0
  for i in range(1, meta_epoch + 1):
    # meta train
    sampler, result_train = train_loop(epoch=i, save_path=save_path)
    # meta valid
    _, result_valid = valid_loop(
        meta_model=sampler, epoch=i, save_path=save_path)

    loss = result_valid.get_best_loss().mean()
    acc = result_valid.get_best_acc().mean()
    # tensorboard
    writer.add_scalars('Loss/valid', {n: loss[n] for n in loss.index}, i)
    writer.add_scalars('Acc/valid', {n: acc[n] for n in acc.index}, i)
    # save numbers
    result_train.save_to_csv(f'records/train_{i}', save_path)
    result_valid.save_to_csv(f'records/valid_{i}', save_path)
    # update the best model
    if acc['ours'] > best_acc:
      sampler.save(save_path)
      best_acc = acc['ours']
      print(f'Best accuracy update: {best_acc*100:6.2f}%')
    else:
      no_improvement += 1
      if no_improvement > tolerance:
        print(f'No improvments for {no_improvement} steps. Early-stopped.')
        break
      else:
        print(f'Early stop counter: {no_improvement}/{tolerance}.')

  # meta test
  _, result_test = test_loop(
      meta_model=C(Sampler.load(save_path)), save_path=save_path)

  result_test.save_to_csv('records/test', save_path)
  result_test.save_final_lineplot('loss_q_m', save_path)
  result_test.save_final_lineplot('acc_q_m', save_path)

  acc = result_test.get_best_acc()
  loss = result_test.get_best_loss()

  print(f'\nFinal result:\n')
  acc.save_mean_std('[Accuracy]', save_path)
  loss.save_mean_std('[Loss]', save_path)
  print('\nend')


if __name__ == '__main__':
  print('Start_of_program.')
  args = parser.parse_args()
  C.set_cuda(not args.cpu and torch.cuda.is_available())
  # gin
  gin_path = os.path.join('gin', args.gin + '.gin')
  gin.parse_config_file(gin_path)
  # prepare dir
  save_path = prepare_dir(gin_path) if not args.volatile else None
  # main loop
  meta_train(save_path=save_path)
  print('End_of_program.')

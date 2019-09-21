import argparse
import itertools
import os
import pdb
import shutil
from collections import OrderedDict
from datetime import datetime

import gin
import numpy as np
import torch
import torch.nn.functional as F
from loader import MetaDataset, PseudoMetaDataset
from networks.model import Model
from networks.sampler import EncoderClass, EncoderInstance, Sampler
from utils import utils
from utils.result import Result
from utils.utils import prepare_dir

C = utils.getCudaManager('default')
sig_1 = utils.getSignalCatcher('SIGINT')
sig_2 = utils.getSignalCatcher('SIGTSTP')

parser = argparse.ArgumentParser(description='Learning to sample')
parser.add_argument('--cpu', action='store_true', help='disable CUDA')
parser.add_argument('--debug', action='store_true', help='kill the bugs!')
parser.add_argument('--gin', type=str, default='omniglot',
                    help='gin filename to load configuration.')


@gin.configurable
def loop(mode, outer_steps, inner_steps, log_steps, inner_lr, outer_lr=None,
         outer_optim=None, unroll_steps=None, meta_batchsize=0,
         meta_model=None, epoch=1):
  """Args:
      meta_batchsize(int): If meta_batchsize |m| > 0, gradients for multiple
        unrollings from each episodes size of |m| will be accumulated in
        sequence but updated all at once. (Which can be done in parallel when
        VRAM is large enough, but will be simulated in this code.)
        If meta_batchsize |m| = 0(default), then update will be performed after
        each unrollings.
  """
  assert mode in ['train', 'valid', 'test']
  assert meta_batchsize >= 0
  print(f'Start_of_{mode}.')
  if mode == 'train' and unroll_steps is None:
    raise Exception("unroll_steps has to be specied when mode='train'.")
  if mode != 'train' and unroll_steps is not None:
    raise Warning("unroll_steps has no effect when mode mode!='train'.")

  train = True if mode == 'train' else False
  force_base = True
  metadata = MetaDataset(split=mode)
  # metadata = PseudoMetaDataset()

  if meta_model is not None:
    sampler = meta_model
  else:
    sampler = C(Sampler())

  if train:
    outer_optim = {'sgd': 'SGD', 'adam': 'Adam'}[outer_optim.lower()]
    outer_optim = getattr(torch.optim, outer_optim)(
        sampler.parameters(), lr=outer_lr)
    if meta_batchsize > 0:
      # serial processing of meta-minibatch
      update_steps = inner_steps * meta_batchsize
    else:
      #  update at every unrollings
      update_steps = unroll_steps

  # for result recordin
  result = Result()
  # result = Result()

  for i in range(1, outer_steps + 1):
    outer_loss = 0

    for j, epi in enumerate(metadata.loader(n_batches=1), 1):
      result_dict = OrderedDict()
      try:
        epi.s = C(epi.s)
        epi.q = C(epi.q)
      except:  # OOM
        continue
      view_classwise = epi.s.get_view_classwise_fn()
      # view_elementwise = epi.s.get_view_elementwise_fn()

      # linear = torch.nn.Linear(32, epi.s.n_classes)
      # params = itertools.chain(
      #     sampler.enc_ins.parameters(), linear.parameters())
      #
      # opt = torch.optim.SGD(params, lr=0.1)
      # for m in range(150):
      #   try:
      #     x = sampler.enc_ins(epi.s.imgs)
      #     x = F.log_softmax(x.squeeze(), dim=1)
      #     loss = torch.nn.NLLLoss()(x, epi.s.labels)
      #     acc = (x.argmax(dim=1) == epi.s.labels).float().mean()
      #     sampler.enc_ins.zero_grad()
      #     linear.zero_grad()
      #     loss.backward(retain_graph=True)
      #     opt.step()
      #   except:
      #     import pdb
      #     pdb.set_trace()
      # print(f'[pretrained encoder] loss:{loss.tolist()}, acc:{acc.tolist()}')

      with torch.set_grad_enabled(train):
        xs = sampler.enc_ins(epi.s.imgs)
        xs = view_classwise(xs)
        xs = sampler.enc_cls(xs)

      # initialize
      model = Model(epi.n_classes)
      params = C(model.get_init_params())
      mask = loss_s = C(sampler.mask_gen.init_mask(epi.n_classes))

      if not train or force_base:
        """baseline 1: naive single task learning
           baseline 2: single task learning with the same loss scale
        """
        params_b0 = params.clone().detach()  # baseline 1
        params_b1 = params.clone().detach()  # baseline 2
        # import pdb; pdb.set_trace()

      for k in range(1, inner_steps + 1):
        debug_1 = sig_1.is_active()
        debug_2 = sig_2.is_active(inner_steps % 10 == 0)

        # generate mask
        with torch.set_grad_enabled(train):
          mask, lr = sampler.mask_gen(xs, mask, loss_s)  # class mask
          # mask = C(torch.ones(epi.n_classes, 1)).detach()

        if not train:
          mask.detach_()
          params.detach_()

        # train on support set
        loss_s, acc_s, loss_s_w, acc_s_w = model(
            epi.s, params, mask, debug_1)

        with torch.set_grad_enabled(train):
          # inner gradient step
          params = params.sgd_step(
              loss_s_w, lr, second_order=True)
          # test on query set
          loss_q_m, acc_q_m = model(epi.q, params, mask=None)

        # record result
        result_dict.update({
            'outer_step': epoch * i, 'inner_step': k,
            'ours_loss_s_w': loss_s_w, 'ours_acc_s_w': acc_s_w,
            'ours_loss_s_m': loss_s.mean(), 'ours_acc_s_m': acc_s.mean(),
            'ours_loss_q_m': loss_q_m, 'ours_acc_q_m': acc_q_m,
        })

        if not train or force_base:
          # feed support set (baseline)
          loss_s_m_b0, acc_s_m_b0 = model(epi.s, params_b0, None)
          loss_s_m_b1, acc_s_m_b1 = model(epi.s, params_b1, None)
          # manaul masking (only for baseline 1)
          loss_s_w_b1 = loss_s_m_b1 * mask.mean().detach()
          acc_s_w_b1 = acc_s_m_b1 * mask.mean().detach()

          with torch.no_grad():
            # inner gradient step (baseline)
            params_b0 = params_b0.sgd_step(
                loss_s_m_b0, inner_lr, second_order=False).detach()
            params_b1 = params_b1.sgd_step(
                loss_s_w_b1, lr.detach(), second_order=False).detach()
            # test on query set
            loss_q_m_b0, acc_q_m_b0 = model(epi.q, params_b0, mask=None)
            loss_q_m_b1, acc_q_m_b1 = model(epi.q, params_b1, mask=None)
          # record result
          result_dict.update({
              'b0_loss_s_m': loss_s_m_b0, 'b0_acc_s_m': acc_s_m_b0,
              'b0_loss_q_m': loss_q_m_b0, 'b0_acc_q_m': acc_q_m_b0,
              'b1_loss_s_w': loss_s_w_b1, 'b1_acc_s_w': acc_s_w_b1,
              'b1_loss_s_m': loss_s_m_b1, 'b1_acc_s_m': acc_s_m_b1,
              'b1_loss_q_m': loss_q_m_b1, 'b1_acc_q_m': acc_q_m_b1,
          })

        # append to the dataframe
        result = result.append_tensors(result_dict)

        # logging
        if k % log_steps == 0:
          msg = (
              f'[epoch:{epoch:2d}|{mode}]'
              f'[out:{i:4d}/{outer_steps}|in:{k:4d}/{inner_steps}][{lr:6.4f}]'
              f'[{"|".join([f"{m:4.2f}" for m in mask.squeeze().tolist()])}]|'
              f'M>0.5:{(mask > 0.5).sum().tolist():2d}|W/S/Q:{epi.n_classes:2d}/'
              f'{epi.s.n_samples:2d}/{epi.q.n_samples:2d}|'
              f'S:w.{loss_s_w.tolist():6.2f}({loss_s.mean().tolist():6.2f})/'
              f'w.{acc_s_w.tolist()*100:6.2f}({acc_s.mean().tolist()*100:6.2f})%|'
              f'Q:{loss_q_m.tolist():6.2f}(m.{np.log(epi.n_classes):3.1f})/'
              f'{acc_q_m.tolist()*100:6.2f}%|'
          )
          if not train or force_base:
            msg += (
                # f'[b0]S:{loss_s_m_b0:6.2f}/{acc_s_m_b0*100:6.2f}%|'
                f'[b0]Q:{loss_q_m_b0:6.2f}/{acc_q_m_b0*100:6.2f}%|'
                # f'[b1]S:{loss_s_m_b1:6.2f}/{acc_s_m_b1*100:6.2f}%|'
                f'[b1]Q:{loss_q_m_b1:6.2f}/{acc_q_m_b1*100:6.2f}%|'
            )
          print(msg)

        # compute outer gradient
        if train and k % unroll_steps == 0:
          outer_loss += loss_q_m
          outer_loss.backward(retain_graph=True)
          outer_loss = 0
          params.detach_()
          sampler.detach_()

        # meta(outer) learning
        if train and k % update_steps == 0:
          outer_optim.step()
          sampler.zero_grad()

      # distinguishable episodes
      if not i == outer_steps:
        print(f'End_of_episode: {i}')
      if debug_2:
        import pdb; pdb.set_trace()
        epi.plot()

  print(f'End_of_{mode}.')
  # del metadata
  return sampler, result


@gin.configurable
def meta_train(train_loop, valid_loop, test_loop, meta_epoch, tolerance,
               save_path):
  no_improvement = 0
  best_acc = 0
  for i in range(1, meta_epoch + 1):
    sampler, result_train = train_loop(epoch=i)
    _, result_valid = valid_loop(meta_model=sampler)
    valid_max_acc_mean = result_valid.get_max(
        col='ours_acc_q_m', group='outer_step').mean()[0]

    result_train.save_to_csv(f'train/epoch_{i}', save_path)
    result_valid.save_to_csv(f'valid/epoch_{i}', save_path)

    if valid_max_acc_mean > best_acc:
      sampler.save(save_path)
      best_acc = valid_max_acc_mean
      print(f'Best accuracy update: {valid_max_acc_mean*100:6.2f}%')
    else:
      no_improvement += 1
      if no_improvement > tolerance:
        print(f'No improvments for {no_improvement} steps. Early-stopped.')
        break
      else:
        print(f'Early stop counter: {no_improvement}/{tolerance}.')

  _, result_test = test_loop(meta_model=C(Sampler.load(save_path)))
  result_test.save_to_csv(f'test/epoch_{i}', save_path)

  ours = result_test.get_max(col='ours_acc_q_m', group='outer_step')
  b0 = result_test.get_max(col='b0_acc_q_m', group='outer_step')
  b1 = result_test.get_max(col='b1_acc_q_m', group='outer_step')

  print(f'\n[Final result]')
  print(f'(ours) mean:{ours.mean()[0]:6.2f} / std:{ours.std()[0]:6.2f}')
  print(f'(b0) mean:{b0.mean()[0]:6.2f} / std:{b0.std()[0]:6.2f}')
  print(f'(b1) mean:{b1.mean()[0]:6.2f} / std:{b1.std()[0]:6.2f}\n')
  import pdb; pdb.set_trace()
  print('end')


if __name__ == '__main__':
  print('Start_of_program.')
  args = parser.parse_args()
  C.set_cuda(not args.cpu and torch.cuda.is_available())
  # gin
  args.gin = 'debug' if args.debug else args.gin
  gin_path = os.path.join('gin', args.gin + '.gin')
  gin.parse_config_file(gin_path)
  # prepare dir
  save_path = prepare_dir(gin_path)
  # main loop
  meta_train(save_path=save_path)
  print('End_of_program.')

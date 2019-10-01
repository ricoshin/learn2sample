import os
import pdb
from collections import OrderedDict

import gin
import numpy as np
import torch
from loader import MetaDataset, MetaMultiDataset, PseudoMetaDataset
from networks.model import Model
from torch.utils.tensorboard import SummaryWriter
from utils import utils
from utils.color import Color
from utils.result import MaskRecoder, Result
from utils.utils import print_colorized_mask, print_confidence

C = utils.getCudaManager('default')
sig_1 = utils.getSignalCatcher('SIGINT')
sig_2 = utils.getSignalCatcher('SIGTSTP')


@gin.configurable
def loop(mode, outer_steps, inner_steps, log_steps, fig_epochs, inner_lr,
         outer_lr=None, outer_optim=None, unroll_steps=None, meta_batchsize=0,
         sampler=None, epoch=1, save_path=None):
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
  metadata = MetaMultiDataset(split=mode)
  # metadata = PseudoMetaDataset()

  if train:
    outer_optim = {'sgd': 'SGD', 'adam': 'Adam'}[outer_optim.lower()]
    outer_optim = getattr(torch.optim, outer_optim)(
        sampler.parameters(), lr=outer_lr)
    if meta_batchsize > 0:
      # serial processing of meta-minibatch
      update_epochs = meta_batchsize
      update_steps = None
    else:
      #  update at every unrollings
      update_steps = unroll_steps
      update_epochs = None
    assert (update_epochs is None) != (update_steps is None)

  # for result recordin
  result = Result()
  writer = SummaryWriter(os.path.join(save_path, 'tfevent'))

  for i in range(1, outer_steps + 1):
    outer_loss = 0
    masks = MaskRecoder()

    for j, epi in enumerate(metadata.loader(n_batches=1), 1):
      result_dict = OrderedDict()
      try:
        epi.s = C(epi.s, device=0)
        epi.q = C(epi.q, device=1)
        base_s = C(epi.s, device=2)
        base_q = C(epi.q, device=2)
      except Exception as ex:
        print(ex) # may be OOM
        continue
      view_classwise = epi.s.get_view_classwise_fn()

      with torch.set_grad_enabled(train):
        xs = sampler.enc_ins(epi.s.imgs)
        xs = view_classwise(xs)
        xs = sampler.enc_cls(xs)

      # initialize
      model = Model(epi.n_classes)
      params = C(model.get_init_params(), device=0)
      mask = loss_s = C(sampler.mask_gen.init_mask(epi.n_classes), device=0)

      if not train or force_base:
        """baseline 1: naive single task learning
           baseline 2: single task learning with the same loss scale
        """
        params_b0 = C(params.clone(), device=2)  # baseline 1
        params_b1 = C(params.clone(), device=2)  # baseline 2

      for k in range(1, inner_steps + 1):
        debug_1 = sig_1.is_active()
        debug_2 = sig_2.is_active(inner_steps % 10 == 0)

        # generate mask
        with torch.set_grad_enabled(train):
          mask_gen_out = sampler.mask_gen(xs, mask, loss_s)  # class mask

        if isinstance(mask_gen_out, tuple):
          # mask = C(torch.ones(epi.n_classes, 1)).detach()
          mask = mask_gen_out[0]
          lr = inner_lr * mask_gen_out[1]
        else:
          mask = mask_gen_out
          lr = inner_lr

        if not train:
          mask.detach_()
          params.detach_()

        # record mask
        masks.append(mask)

        # train on support set
        loss_s, acc_s, loss_s_w, acc_s_w = model(
            epi.s, params, mask, debug_1)

        with torch.set_grad_enabled(train):
          # inner gradient step
          params = params.sgd_step(
              loss_s_w, lr, second_order=True)
          # test on query set
          params = C(params, device=1)
          loss_q_m, acc_q_m, conf = model(epi.q, params, mask=None)
          params = C(params, device=0)

        # record result
        result_dict.update({
            'outer_step': epoch * i, 'inner_step': k,
            'ours_loss_s_w': loss_s_w, 'ours_acc_s_w': acc_s_w,
            'ours_loss_s_m': loss_s.mean(), 'ours_acc_s_m': acc_s.mean(),
            'ours_loss_q_m': loss_q_m, 'ours_acc_q_m': acc_q_m,
        })

        if not train or force_base:
          # feed support set (baseline)
          loss_s_m_b0, acc_s_m_b0, _ = model(base_s, params_b0, None)
          loss_s_m_b1, acc_s_m_b1, _ = model(base_s, params_b1, None)
          # manaul masking (only for baseline 1)
          loss_s_w_b1 = loss_s_m_b1 * mask.mean().detach()
          acc_s_w_b1 = acc_s_m_b1 * mask.mean().detach()

          with torch.no_grad():
            # inner gradient step (baseline)
            params_b0 = params_b0.sgd_step(
                loss_s_m_b0, inner_lr, second_order=False).detach()
            params_b1 = params_b1.sgd_step(
                loss_s_w_b1, lr.tolist(), second_order=False).detach()
            # test on query set
            loss_q_m_b0, acc_q_m_b0, conf_b0 = model(
                base_q, params_b0, mask=None)
            loss_q_m_b1, acc_q_m_b1, conf_b1 = model(
                base_q, params_b1, mask=None)
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
              f'[out:{i:3d}/{outer_steps}|in:{k:4d}/{inner_steps}][{lr:4.3f}]'
              f'[{print_colorized_mask(mask, 0.0, 1.0, 100, "2d")}]|'
              # f'[epoch:{epoch:2d}|{mode}]'
              # f'[out:{i:3d}/{outer_steps}|in:{k:4d}/{inner_steps}][{lr:4.3f}]'
              # f'[{print_colorized_mask(acc_s, 0.0, 1.0, 100, "4d")}]|\n'
              # f'[epoch:{epoch:2d}|{mode}]'
              # f'[out:{i:3d}/{outer_steps}|in:{k:4d}/{inner_steps}][{lr:4.3f}]'
              # f'[{print_colorized_mask(loss_s, 0.0, 2.3, 1.0, "4.2f", [40])}]|'

              # f'M>0.5:{(mask > 0.5).sum().tolist():2d}|W/S/Q:{epi.n_classes:2d}/'
              # f'{epi.s.n_samples:2d}/{epi.q.n_samples:2d}|'
              # f'S:w.{loss_s_w.tolist():5.2f}({loss_s.mean().tolist():5.2f})/'
              f'S:{loss_s_w.tolist(): 5.2f}/{acc_s_w.tolist()*100:5.1f}%|'
              f'[ours]Q:{Color.GREEN}{loss_q_m.tolist():5.2f}{Color.END}'
              f'(m.{np.log(epi.n_classes):3.1f})/'
              f'{Color.RED}{acc_q_m.tolist()*100:5.1f}{Color.END}%/'
              f'{print_confidence(conf, "2d")}|'
          )
          if not train or force_base:
            msg += (
                # f'[b0]S:{loss_s_m_b0:5.2f}/{acc_s_m_b0*100:5.2f}%|'
                f'[b0]Q:{Color.GREEN}{loss_q_m_b0:5.2f}{Color.END}/'
                f'{Color.RED}{acc_q_m_b0*100:5.1f}{Color.END}%/'
                f'{print_confidence(conf_b0, "2d")}|'
                # f'[b1]S:{loss_s_m_b1:5.2f}/{acc_s_m_b1*100:5.2f}%|'
                f'[b1]Q:{Color.GREEN}{loss_q_m_b1:5.2f}{Color.END}/'
                f'{Color.RED}{acc_q_m_b1*100:5.1f}{Color.END}%/'
                f'{print_confidence(conf_b1, "2d")}|'
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
        if train and update_steps and k % update_steps == 0:
          outer_optim.step()
          sampler.zero_grad()

      if train and update_epochs and i % update_epochs == 0:
        outer_optim.step()
        sampler.zero_grad()
        if update_epochs:
          print(f'Meta-batchsize is {meta_batchsize}: Sampler updated.')

      if train and update_steps:
        print(f'Meta-batchsize is zero. Updating after every unrollings.')

      # tensorboard
      if train:
        step = (epoch * (outer_steps - 1)) + i
        res = Result(result[result['outer_step'] == i])
        loss = res.get_best_loss().mean()
        acc = res.get_best_acc().mean()
        writer.add_scalars(
            'Loss/train', {n: loss[n] for n in loss.index}, step)
        writer.add_scalars('Acc/train', {n: acc[n] for n in acc.index}, step)

      # dump figures
      if i % fig_epochs == 0:
        epi.s.save_fig(f'imgs/support_{i}', save_path)
        epi.q.save_fig(f'imgs/query_{i}', save_path)
        masks.save_fig(f'imgs/masks_{i}', save_path)

      # distinguishable episodes
      if not i == outer_steps:
        print(f'Path for saving: {save_path}')
        print(f'End_of_episode: {i}')

  print(f'End_of_{mode}.')
  # del metadata
  return sampler, result

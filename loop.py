import os
import pdb
from collections import OrderedDict

import gin
import numpy as np
import torch
from loader.meta_dataset import MetaDataset, MetaMultiDataset, PseudoMetaDataset
from nn.model import Model
from torch.utils.tensorboard import SummaryWriter
from utils import utils
from utils.color import Color
from utils.result import MaskRecoder, Result
from utils.utils import Printer

C = utils.getCudaManager('default')
sig_1 = utils.getSignalCatcher('SIGINT')
sig_2 = utils.getSignalCatcher('SIGTSTP')


@gin.configurable
def loop(mode, outer_steps, inner_steps, log_steps, fig_epochs, inner_lr,
         log_mask=True, unroll_steps=None, meta_batchsize=0, sampler=None,
         epoch=1, outer_optim=None, save_path=None):
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

  if train:
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
  if save_path:
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
        print(ex)  # may be OOM
        continue

      with torch.set_grad_enabled(train):
        x = sampler.encoder(epi.s.imgs, epi.s.n_classes)

      # initialize
      model = Model(epi.n_classes)
      params = C(model.get_init_params().with_name('ours'), device=0)
      # mask_dim = epi.n_classes if mask_mode == 'class' else len(epi.s)
      mask = loss_s = C(sampler.mask_gen.init_mask(
        epi.s.n_classes, epi.s.n_samples), device=0)

      if not train or force_base:
        """baseline 1: naive single task learning
           baseline 2: single task learning with the same loss scale
        """
        # baseline parameters
        params_b0 = C(params.copy().with_name('b0'), device=2)
        params_b1 = C(params.copy().with_name('b1'), device=2)

      for k in range(1, inner_steps + 1):
        if sig_2.is_active():
          import pdb; pdb.set_trace()

        # generate mask
        with torch.set_grad_enabled(train):
          mask, lr = sampler.mask_gen(x, mask, loss_s)  # class mask

        # use learned lr if available
        lr = inner_lr if lr is None else lr
        # record mask
        masks.append(mask)

        # train on support set
        out_s = model(epi.s, params, mask)
        # inner gradient step
        params = params.sgd_step(out_s.loss_m, lr, 'second')

        # test on query set
        with torch.set_grad_enabled(train):
          params = C(params, device=1)
          out_q = model(epi.q, params, mask=None)
          params = C(params, device=0)

        # record result
        result_dict.update(outer_step=epoch * i, inner_step=k)
        result_dict.update(**out_s.as_dict(), **out_q.as_dict())

        if not train or force_base:
          # feed support set (baseline)
          out_s_b0 = model(base_s, params_b0, None)
          out_s_b1 = model(base_s, params_b1, None)

          # attach mask to get loss_s
          out_s_b1.attach_mask(mask)

          # inner gradient step (baseline)
          params_b0 = params_b0.sgd_step(out_s_b0.loss, inner_lr, 'no_grad')
          params_b1 = params_b1.sgd_step(out_s_b1.loss_s, lr, 'no_grad')

          with torch.no_grad():
            # test on query set
            out_q_b0 = model(base_q, params_b0, mask=None)
            out_q_b1 = model(base_q, params_b1, mask=None)
          # record result
          result_dict.update(**out_s_b0.as_dict(), **out_s_b1.as_dict())
          result_dict.update(**out_q_b0.as_dict(), **out_q_b1.as_dict())

        # append to the dataframe
        result = result.append_tensors(result_dict)

        # logging
        if k % log_steps == 0:
          # print info
          msg = Printer.step_info(
            epoch, mode, i, outer_steps, k, inner_steps, lr)
          msg += Printer.way_shot_query(epi)
          # print mask
          if not sig_1.is_active() and log_mask:
            msg += Printer.colorized_mask(mask, fmt="2d")
          # print outputs (loss, acc, etc.)
          msg += Printer.outputs([out_s, out_q], sig_1.is_active())
          if not train or force_base:
            msg += Printer.outputs([out_q_b0, out_q_b1], sig_1.is_active())
          print(msg)

        # compute outer gradient
        if train and (k % unroll_steps == 0 or k == inner_steps):
          outer_loss += out_q.loss
          outer_loss.backward(retain_graph=True)
          outer_loss = 0
          params.detach_().requires_grad_()
          sampler.detach_()

        if not train:
          params.requires_grad_()

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
      if save_path and train:
        step = (epoch * (outer_steps - 1)) + i
        res = Result(result[result['outer_step'] == i])
        loss = res.get_best_loss().mean()
        acc = res.get_best_acc().mean()
        writer.add_scalars(
            'Loss/train', {n: loss[n] for n in loss.index}, step)
        writer.add_scalars('Acc/train', {n: acc[n] for n in acc.index}, step)

      # dump figures
      if save_path and i % fig_epochs == 0:
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

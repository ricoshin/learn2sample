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
from utils.result import ResultDict, ResultFrame
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
  result_frame = ResultFrame()
  result_dict = ResultDict()
  if save_path:
    writer = SummaryWriter(os.path.join(save_path, 'tfevent'))

  for i in range(1, outer_steps + 1):
    outer_loss = 0

    for j, epi in enumerate(metadata.loader(n_batches=1), 1):
      try:
        epi_s_sampler = C(epi.s, 0)  # support set for sampler
        epi_s_model = C(epi.s, 2)  # support set for learner
        epi_q_model = C(epi.q, 3)  # query set for learner
        epi_s_base = C(epi.s, 4)  # support set for baselines
        epi_q_base = C(epi.q, 4)  # query set for baselines
      except Exception as ex:
        print(ex)  # may be OOM
        continue

      mask = out_s_loss_masked = sampler.mask_gen.init_mask(
        epi.s.n_classes, epi.s.n_samples)

      # initialize base learner
      model = Model(epi.n_classes)
      params = model.get_init_params('ours')

      if not train or force_base:
        """baseline 1: naive single task learning
           baseline 2: single task learning with the same loss scale
        """
        # baseline parameters
        params_b0 = C(params.copy('b0'), 4)
        params_b1 = C(params.copy('b1'), 4)

      for k in range(1, inner_steps + 1):
        # task encoding (very first step and right after a meta-update)
        if (k == 1) or (train and (k - 1) % unroll_steps == 0):
          with torch.set_grad_enabled(train):
            x = sampler.encoder(epi_s_sampler.imgs, epi.s.n_classes)

        # generate mask
        with torch.set_grad_enabled(train):
          x, mask, out_s_loss_masked = C([x, mask, out_s_loss_masked], 1)
          mask, lr = sampler.mask_gen(x, mask, out_s_loss_masked)

        # use learned learning rate if available
        #   inner_lr: preset / lr: learned
        lr = inner_lr if lr is None else lr

        # train on support set
        params, mask = C([params, mask], 2)
        out_s = model(epi_s_model, params, mask)
        out_s_loss_masked = out_s.loss_masked

        # inner gradient step
        out_s_loss_masked_mean, lr = C([out_s.loss_masked_mean, lr], 2)
        params = params.sgd_step(out_s_loss_masked_mean, lr, 'second')

        # test on query set
        with torch.set_grad_enabled(train):
          params = C(params, 3)
          out_q = model(epi_q_model, params, mask=None)

        if not train or force_base:
          # feed support set (baseline)
          out_s_b0 = model(epi_s_base, params_b0, None)
          out_s_b1 = model(epi_s_base, params_b1, None)

          # attach mask to get loss_s
          out_s_b1.attach_mask(mask)

          # inner gradient step (baseline)

          # lll = out_s_b0.loss * 0.5
          params_b0 = params_b0.sgd_step(
            out_s_b0.loss.mean(), inner_lr, 'no_grad')
          params_b1 = params_b1.sgd_step(
            out_s_b1.loss_scaled_mean, lr, 'no_grad')

          with torch.no_grad():
            # test on query set
            out_q_b0 = model(epi_q_base, params_b0, mask=None)
            out_q_b1 = model(epi_q_base, params_b1, mask=None)

          # record result
          result_dict.append(
            outer_step=epoch * i, inner_step=k,
            **out_s.as_dict(), **out_q.as_dict(),
            **out_s_b0.as_dict(), **out_s_b1.as_dict(),
            **out_q_b0.as_dict(), **out_q_b1.as_dict())
          ### end of inner steps (k) ###

        # append to the dataframe
        result_frame = result_frame.append_dict(
          result_dict.index_all(-1).mean_all(-1))

        # logging
        if k % log_steps == 0:
          # print info
          msg = Printer.step_info(
            epoch, mode, i, outer_steps, k, inner_steps, lr)
          msg += Printer.way_shot_query(epi)
          # print mask
          if not sig_1.is_active() and log_mask:
            msg += Printer.colorized_mask(mask, fmt="2d", vis_num=20)
          # print outputs (loss, acc, etc.)
          msg += Printer.outputs([out_s, out_q], sig_1.is_active())
          if not train or force_base:
            msg += Printer.outputs([out_q_b0, out_q_b1], sig_1.is_active())
          print(msg)

        # to debug in the middle of running process.
        if k == inner_steps and sig_2.is_active():
          import pdb; pdb.set_trace()

        # compute outer gradient
        if train and (k % unroll_steps == 0 or k == inner_steps):
          outer_loss += out_q.loss.mean()
          outer_loss.backward()
          outer_loss = 0
          params.detach_().requires_grad_()
          out_s_loss_masked.detach_()
          sampler.detach_()
          mask.detach_()

        if not train:
          params.requires_grad_()

        # meta(outer) learning
        if train and update_steps and k % update_steps == 0:
          outer_optim.step()
          sampler.zero_grad()
        ### end of meta minibatch (j) ###


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
        res = ResultFrame(result_frame[result_frame['outer_step'] == i])
        loss = res.get_best_loss().mean()
        acc = res.get_best_acc().mean()
        writer.add_scalars(
            'Loss/train', {n: loss[n] for n in loss.index}, step)
        writer.add_scalars('Acc/train', {n: acc[n] for n in acc.index}, step)

      # dump figures
      if save_path and i % fig_epochs == 0:
        epi.s.save_fig(f'imgs/support', save_path, i)
        epi.q.save_fig(f'imgs/query', save_path, i)
        result_dict['ours_s_mask'].save_fig(f'imgs/masks', save_path, i)
        result_dict.get_items(['ours_s_mask', 'ours_s_loss',
          'ours_s_loss_masked', 'b0_s_loss', 'b1_s_loss']).save_csv(
            f'classwise/{mode}', save_path, i)

      # distinguishable episodes
      if not i == outer_steps:
        print(f'Path for saving: {save_path}')
        print(f'End_of_episode: {i}')
      ### end of episode (i) ###

  print(f'End_of_{mode}.')
  # del metadata
  return sampler, result_frame

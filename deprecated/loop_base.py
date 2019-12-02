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
  mask_based = 'query'
  mask_type = 5
  mask_sample = False
  mask_scale = True
  easy_ratio = 17/50 # 0.3
  scale_manual = 0.8  # 1.0 when lr=0.001 and 0.8 when lr=0.00125
  inner_lr *= scale_manual

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
  if save_path:
    writer = SummaryWriter(os.path.join(save_path, 'tfevent'))

  for i in range(1, outer_steps + 1):
    outer_loss = 0

    for j, epi in enumerate(metadata.loader(n_batches=1), 1):
      # initialize base learner
      model = Model(epi.n_classes)
      params = model.get_init_params('ours')
      epi.s = C(epi.s)
      epi.q = C(epi.q)

      # baseline parameters
      params_b0 = C(params.copy('b0'))
      params_b1 = C(params.copy('b1'))
      params_b2 = C(params.copy('b2'))

      result_dict = ResultDict()
      for k in range(1, inner_steps + 1):
        # feed support set (baseline)
        out_s_b0 = model(epi.s, params_b0, None)
        out_s_b1 = model(epi.s, params_b1, None)
        out_s_b2 = model(epi.s, params_b2, None)

        if mask_based == 'support':
          out = out_s_b1
        elif mask_based == 'query':
          with torch.no_grad():
            # test on query set
            out_q_b1 = model(epi.q, params_b1, mask=None)
          out = out_q_b1
        else:
          print('WARNING')

        # attach mask to get loss_s
        if mask_type == 1:
          mask = (out.loss.exp().mean().log() - out.loss).exp()
        elif mask_type == 2:
          mask = (out.loss.exp().mean().log() / out.loss)
        elif mask_type == 3:
          mask = out.loss.mean() / out.loss
        elif mask_type == 4:
          mask = out.loss.min() / out.loss
        elif mask_type == 5 or mask_type == 6:
          mask_scale = False
          # weight by magnitude
          if mask_type == 5:
            mask = [scale_manual]*5 + [(1-easy_ratio)*scale_manual]*5
          # weight by ordering
          elif mask_type == 6:
            if k < inner_steps * easy_ratio:
              mask = [scale_manual]*5 + [0.0]*5
            else:
              mask = [scale_manual]*5 + [scale_manual]*5
          # sampling from 0 < p < 1
          if mask_sample:
            mask = [np.random.binomial(1, m) for m in mask]
          mask = C(torch.tensor(mask).float())
        else:
          print('WARNING')
        if mask_scale:
          mask = (mask / (mask.max() + 0.05))

        # to debug in the middle of running process.class
        if sig_2.is_active():
          import pdb; pdb.set_trace()

        mask = mask.unsqueeze(1)
        out_s_b1.attach_mask(mask)
        out_s_b2.attach_mask(mask)
        # lll = out_s_b0.loss * 0.5
        params_b0 = params_b0.sgd_step(
          out_s_b0.loss.mean(), inner_lr, 'no_grad')
        params_b1 = params_b1.sgd_step(
          out_s_b1.loss_masked_mean, inner_lr, 'no_grad')
        params_b2 = params_b2.sgd_step(
          out_s_b2.loss_scaled_mean, inner_lr, 'no_grad')

        with torch.no_grad():
          # test on query set
          out_q_b0 = model(epi.q, params_b0, mask=None)
          out_q_b1 = model(epi.q, params_b1, mask=None)
          out_q_b2 = model(epi.q, params_b2, mask=None)

        # record result
        result_dict.append(
          outer_step=epoch * i, inner_step=k,
          **out_s_b0.as_dict(), **out_s_b1.as_dict(), **out_s_b2.as_dict(),
          **out_q_b0.as_dict(), **out_q_b1.as_dict(), **out_q_b2.as_dict())
        ### end of inner steps (k) ###

        # append to the dataframe
        result_frame = result_frame.append_dict(
          result_dict.index_all(-1).mean_all(-1))

        # logging
        if k % log_steps == 0:
          # print info
          msg = Printer.step_info(
            epoch, mode, i, outer_steps, k, inner_steps, inner_lr)
          msg += Printer.way_shot_query(epi)
          # # print mask
          if not sig_1.is_active() and log_mask:
            msg += Printer.colorized_mask(mask, fmt="3d", vis_num=20)
          # print outputs (loss, acc, etc.)
          msg += Printer.outputs([out_s_b0, out_s_b1, out_s_b2], sig_1.is_active())
          msg += Printer.outputs([out_q_b0, out_q_b1, out_q_b2], sig_1.is_active())
          print(msg)


        ### end of meta minibatch (j) ###

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
        # result_dict['ours_s_mask'].save_fig(f'imgs/masks', save_path, i)
        result_dict.get_items(['b0_s_loss', 'b1_s_loss', 'b2_s_loss'
          'b0_q_loss', 'b1_q_loss', 'b2_q_loss']).save_csv(
            f'classwise/{mode}', save_path, i)

      # distinguishable episodes
      if not i == outer_steps:
        print(f'Path for saving: {save_path}')
        print(f'End_of_episode: {i}')
        import pdb; pdb.set_trace()
      ### end of episode (i) ###


  print(f'End_of_{mode}.')
  # del metadata
  return sampler, result_frame

import os
import pdb
from collections import OrderedDict

import gin
import numpy as np
import torch
import torch.multiprocessing
from loader.loader import EpisodeIterator
from loader.meta_dataset import (MetaDataset, MetaMultiDataset,
                                 PseudoMetaDataset)
from loader.metadata import Metadata
from nn.model import Model
from nn.output import ModelOutput
from nn.sampler2 import MaskMode
from torch.utils.tensorboard import SummaryWriter
from utils import utils
from utils.color import Color
from utils.result import ResultDict, ResultFrame
from utils.utils import Printer

torch.multiprocessing.set_sharing_strategy('file_system')

C = utils.getCudaManager('default')
sig_1 = utils.getSignalCatcher('SIGINT')
sig_2 = utils.getSignalCatcher('SIGTSTP')


@gin.configurable
def loop(mode, data, outer_steps, inner_steps, log_steps, fig_epochs, inner_lr,
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
  # TODO: to gin configuration
  fc_pulling = False  # does not support for now
  samples_per_class = 3
  anneal_outer_steps = 50
  split_method = {1: 'inclusive', 2: 'exclusive'}[1]
  mask_mode = {
    0: MaskMode.SOFT,
    1: MaskMode.DISCRETE,
    2: MaskMode.CONCRETE,
    }[2]

  def inner_step_scheduler(cur_outer_step):
    """when cur_outer_step reaches to anneal_outer_steps, inner_steps """
    assert 0 <= anneal_outer_steps <= outer_steps
    if cur_outer_step < anneal_outer_steps:
      r = cur_outer_step / anneal_outer_steps
      cur_inner_steps = int(inner_steps * r)
    else:
      cur_inner_steps = inner_steps
    if not anneal_outer_steps == 0:
      print('Inner step scheduled : '
            f'{cur_inner_steps}/{inner_steps} ({r*100:5.2f}%)')
    return cur_inner_steps

  # 100 classes in total
  if split_method == 'exclusive':
    # meta_support, remainder = data.split_class(0.1)  # 10 classes
    # meta_query = remainder.sample_class(50)  # 50 classes
    meta_support, meta_query = data.split_class(1 / 5)  # 1(100) : 4(400)
    # meta_support, meta_query = data.split_class(0.3)  # 30 : 70 classes
  elif split_method == 'inclusive':
    # subdata = data.sample_class(10)  # 50 classes
    meta_support, meta_query = data.split_instance(0.5)  # 5:5 instances
  else:
    raise Exception()

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
    result_dict = ResultDict()
    scheduled_inner_steps = inner_step_scheduler(i)

    # initialize sampler
    sampler.initialize()
    # initialize base learner
    model_q = Model(len(meta_support), mode='metric')

    if fc_pulling:
      model_s = Model(len(meta_support), mode='fc')
      params = C(model_s.get_init_params('ours'))
    else:
      model_s = model_q
      params = C(model_s.get_init_params('ours'))

    if not train or force_base:
      """baseline 0: naive single task learning
         baseline 1: single task learning with the same loss scale
      """
      # baseline parameters
      params_b0 = C(params.copy('b0'), 4)
      params_b1 = C(params.copy('b1'), 4)

    # episode iterator
    episode_iterator = EpisodeIterator(
        inner_steps=scheduled_inner_steps,
        support=meta_support.sample_class(10),
        query=meta_query.sample_class(10),
        split_ratio=0.5,
        resample_every_iteration=True,
        samples_per_class=samples_per_class,
        num_workers=4,
        pin_memory=True,
    ).sample_episode()

    for k, (meta_s, meta_q) in enumerate(episode_iterator, 1):
      outs = []
      meta_s = C(meta_s)
      with torch.set_grad_enabled(train):
        # task encoding (very first step and right after a meta-update)
        if (k == 1) or (train and (k - 1) % unroll_steps == 0):
          out_for_sampler = model_s(meta_s, params)
          mask, lr = sampler(
              pairwise_dist=out_for_sampler.pairwise_dist,
              classwise_loss=out_for_sampler.loss,
              classwise_acc=out_for_sampler.acc,
              n_classes=out_for_sampler.n_classes,
              mask_mode=mask_mode,
          )
        # mask = mask_.rsample() if concrete_mask else mask_

      # use learned learning rate if available
      lr = inner_lr if lr is None else lr  # inner_lr: preset / lr: learned

      # train on support set
      params, mask = C([params, mask], 2)
      out_s = model_s(meta_s, params, mask=mask, mask_mode=mask_mode)
      out_s_loss_masked = out_s.loss_masked
      outs.append(out_s)

      # inner gradient step
      out_s_loss_masked_mean, lr = C([out_s.loss_masked_mean, lr], 2)
      params = params.sgd_step(
          out_s_loss_masked_mean, lr, second_order=True, detach_param=True)

      # baseline
      if not train or force_base:
        out_s_b0 = model_s(meta_s, params_b0)
        out_s_b1 = model_s(meta_s, params_b1)
        outs.extend([out_s_b0, out_s_b1])

        # attach mask to get loss_s
        out_s_b1.attach_mask(mask)

        # inner gradient step (baseline)
        params_b0 = params_b0.sgd_step(out_s_b0.loss.mean(), inner_lr)
        params_b1 = params_b1.sgd_step(out_s_b1.loss_scaled_mean, lr)

      del meta_s
      meta_q = C(meta_q)
      # test on query set
      with torch.set_grad_enabled(train):
        params = C(params, 3)
        out_q = model_q(meta_q, params)
        outs.append(out_q)

      # baseline
      if not train or force_base:
        with torch.no_grad():
          # test on query set
          out_q_b0 = model_q(meta_q, params_b0)
          out_q_b1 = model_q(meta_q, params_b1)
          outs.extend([out_q_b0, out_q_b1])

      del meta_q
      # record result
      result_dict.append(
          outer_step=epoch * i,
          inner_step=k,
          **ModelOutput.as_merged_dict(outs))

      # append to the dataframe
      result_frame = result_frame.append_dict(
          result_dict.index_all(-1).mean_all(-1))

      # logging
      if k % log_steps == 0:
        # print info
        msg = Printer.step_info(
            epoch, mode, i, outer_steps, k, scheduled_inner_steps, lr)
        # msg += Printer.way_shot_query(epi)
        # print mask
        if not sig_1.is_active() and log_mask:
          msg += Printer.colorized_mask(mask, fmt="2d", vis_num=20)
        # print outputs (loss, acc, etc.)
        msg += Printer.outputs(outs, sig_1.is_active())
        print(msg)

      # to debug in the middle of running process.
      if k == inner_steps and sig_2.is_active():
        import pdb
        pdb.set_trace()

      # compute outer gradient
      if train and (k % unroll_steps == 0 or k == inner_steps):
        outer_loss += out_q.loss.mean()
        outer_loss.backward()
        outer_loss = 0
        params.detach_().requires_grad_()
        out_s_loss_masked.detach_()
        mask = mask.detach()

      if not train:
        # when params is not leaf node created by user,
        #   requires_grad attribute is False by default.
        params.requires_grad_()

      # meta(outer) learning
      if train and update_steps and k % update_steps == 0:
        # when you have meta_batchsize == 0, update_steps == unroll_steps
        outer_optim.step()
        sampler.zero_grad()
      ### end of inner steps (k) ###

    if train and update_epochs and i % update_epochs == 0:
      # when you have meta_batchsize > 0, update_epochs == meta_batchsize
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
      # meta_s.s.save_fig(f'imgs/meta_support', save_path, i)
      # meta_q.s.save_fig(f'imgs/meta_query', save_path, i)
      result_dict['ours_s_mask'].save_fig(f'imgs/masks', save_path, i)
      result_dict.get_items(
          ['ours_s_mask', 'ours_s_loss', 'ours_s_loss_masked', 'b0_s_loss',
           'b1_s_loss', 'ours_q_loss', 'b0_q_loss', 'b1_q_loss']
      ).save_csv(f'classwise/{mode}', save_path, i)

    # distinguishable episodes
    if not i == outer_steps:
      print(f'Path for saving: {save_path}')
      print(f'End_of_episode: {i}')

    ### end of episode (i) ###

  print(f'End_of_{mode}.')
  # del metadata
  return sampler, result_frame

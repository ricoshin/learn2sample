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

  samples_per_class = 3
  train = True if mode == 'train' else False
  force_base = True
  model_mode = 'metric'
  # split_method = 'inclusive'
  split_method = 'exclusive'
  hard_mask = False

  # 100 classes in total
  if split_method == 'exclusive':
    meta_support, remainder = data.split_class(0.1)  # 10 classes
    meta_query = remainder.sample_class(50)  # 50 classes
    # meta_support, meta_query = data.split_class(0.3)  # 30 : 70 classes
  elif split_method == 'inclusive':
    subdata = data.sample_class(10)  # 50 classes
    meta_support, meta_query = subdata.split_instance(0.5)  # 5:5 instances
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
    # initialize sampler
    sampler.initialize()
    # initialize base learner
    model_fc = Model(len(meta_support), mode='fc')
    model_metric = Model(len(meta_support), mode='metric')
    params = C(model_fc.get_init_params('ours'))

    if not train or force_base:
      """baseline 0: naive single task learning
         baseline 1: single task learning with the same loss scale
      """
      # baseline parameters
      params_b0 = C(params.copy('b0'), 4)
      params_b1 = C(params.copy('b1'), 4)

    result_dict = ResultDict()
    episode_iterator = EpisodeIterator(
        support=meta_support,
        query=meta_query,
        split_ratio=0.5,
        resample_every_iteration=True,
        inner_steps=inner_steps,
        samples_per_class=samples_per_class,
        num_workers=4,
        pin_memory=True,
    )
    episode_iterator.sample_episode()

    for k, (meta_s, meta_q) in enumerate(episode_iterator, 1):
      # task encoding (very first step and right after a meta-update)
      if (k == 1) or (train and (k - 1) % unroll_steps == 0):
        with torch.set_grad_enabled(train):
          out_embed = model_metric(meta_s, params)
          mask, lr = sampler(
            pairwise_dist=out_embed.pairwise_dist,
            classwise_loss=out_embed.loss,
            classwise_acc=out_embed.acc,
            n_classes=out_embed.n_classes,
            )
      # use learned learning rate if available
      #   inner_lr: preset / lr: learned
      lr = inner_lr if lr is None else lr

      # train on support set
      params, mask = C([params, mask], 2)
      out_s = model_fc(meta_s, params, mask=mask, hard_mask=hard_mask)
      out_s_loss_masked = out_s.loss_masked

      # inner gradient step
      out_s_loss_masked_mean, lr = C([out_s.loss_masked_mean, lr], 2)
      params = params.sgd_step(
        out_s_loss_masked_mean, lr, 'second', detach_param=True)

      # test on query set
      with torch.set_grad_enabled(train):
        params = C(params, 3)
        out_q = model_metric(meta_q, params)

      if not train or force_base:
        # feed support set (baseline)
        out_s_b0 = model_fc(meta_s, params_b0)
        out_s_b1 = model_fc(meta_s, params_b1)

        # attach mask to get loss_s
        out_s_b1.attach_mask(mask)

        # inner gradient step (baseline)
        params_b0 = params_b0.sgd_step(
            out_s_b0.loss.mean(), inner_lr, 'no_grad')
        params_b1 = params_b1.sgd_step(
            out_s_b1.loss_scaled_mean, lr, 'no_grad')

        with torch.no_grad():
          # test on query set
          out_q_b0 = model_metric(meta_q, params_b0)
          out_q_b1 = model_metric(meta_q, params_b1)

      outs = [out_s, out_s_b0, out_s_b1, out_q, out_q_b0, out_q_b1]
      # record result
      result_dict.append(
        outer_step=epoch * i, inner_step=k, **ModelOutput.as_merged_dict(outs))

      # append to the dataframe
      result_frame = result_frame.append_dict(
          result_dict.index_all(-1).mean_all(-1))

      # logging
      if k % log_steps == 0:
        # print info
        msg = Printer.step_info(
            epoch, mode, i, outer_steps, k, inner_steps, lr)
        # msg += Printer.way_shot_query(epi)
        # print mask
        if not sig_1.is_active() and log_mask:
          msg += Printer.colorized_mask(mask, fmt="2d", vis_num=20)
        # print outputs (loss, acc, etc.)
        msg += Printer.outputs([out_s, out_q], sig_1.is_active())
        if not train or force_base:
          msg += Printer.outputs(
              [out_s_b0, out_q_b0, out_s_b1, out_q_b1], sig_1.is_active())
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
        params.requires_grad_()

      # meta(outer) learning
      if train and update_steps and k % update_steps == 0:
        outer_optim.step()
        sampler.zero_grad()
      ### end of inner steps (k) ###

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
      meta_s.s.save_fig(f'imgs/meta_support', save_path, i)
      meta_q.s.save_fig(f'imgs/meta_query', save_path, i)
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

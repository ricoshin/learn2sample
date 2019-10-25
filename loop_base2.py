import os
import pdb
from collections import OrderedDict

import gin
import numpy as np
import torch
from loader.loader import EpisodeIterator
from loader.metadata import Metadata
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

  samples_per_class = 5
  train = True if mode == 'train' else False
  force_base = True
  model_mode = 'metric'

  # meta-support: 30 / meta-query: 70 classes
  meta_support, remainder = data.split_class(0.1)  # 10
  meta_query, _ = remainder.split_class(0.5)  # 45


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

    import pdb; pdb.set_trace()
    model = Model(len(meta_support), mode=model_mode)
    # baseline parameters
    params_b0 = C(model.get_init_params('b0'))

    result_dict = ResultDict()
    episode_iterator = EpisodeIterator(
      support=meta_support,
      query=meta_query,
      split_ratio=0.5,
      inner_steps=inner_steps,
      samples_per_class=samples_per_class,
      num_workers=2,
      pin_memory=True,
    )
    episode_iterator.sample_episode()

    for k, (meta_s, meta_q) in enumerate(episode_iterator, 1):
      # import pdb; pdb.set_trace()

      # feed support set (baseline)
      out_s_b0 = model(meta_s, params_b0, None)

      # inner gradient step (baseline)
      params_b0 = params_b0.sgd_step(
        out_s_b0.loss.mean(), inner_lr, 'no_grad')

      with torch.no_grad():
        # test on query set
        out_q_b0 = model(meta_q, params_b0, mask=None)

      # record result
      result_dict.append(
        outer_step=epoch * i, inner_step=k,
        **out_s_b0.as_dict(), **out_q_b0.as_dict())
      ### end of inner steps (k) ###

      # append to the dataframe
      result_frame = result_frame.append_dict(
        result_dict.index_all(-1).mean_all(-1))

      # logging
      if k % log_steps == 0:
        # print info
        msg = Printer.step_info(
          epoch, mode, i, outer_steps, k, inner_steps, inner_lr)
        # msg += Printer.way_shot_query(epi)
        # print mask
        if not sig_1.is_active() and log_mask:
        # print outputs (loss, acc, etc.)
          msg += Printer.outputs(
            [out_s_b0, out_q_b0], sig_1.is_active())
        print(msg)

      # to debug in the middle of running process.
      if k == inner_steps and sig_2.is_active():
        import pdb; pdb.set_trace()

    # # tensorboard
    # if save_path and train:
    #   step = (epoch * (outer_steps - 1)) + i
    #   res = ResultFrame(result_frame[result_frame['outer_step'] == i])
    #   loss = res.get_best_loss().mean()
    #   acc = res.get_best_acc().mean()
    #   writer.add_scalars(
    #       'Loss/train', {n: loss[n] for n in loss.index}, step)
    #   writer.add_scalars('Acc/train', {n: acc[n] for n in acc.index}, step)
    #
    # # dump figures
    # if save_path and i % fig_epochs == 0:
    #   meta_s.s.save_fig(f'imgs/meta_support', save_path, i)
    #   meta_q.s.save_fig(f'imgs/meta_query', save_path, i)
    #   result_dict['ours_s_mask'].save_fig(f'imgs/masks', save_path, i)
    #   result_dict.get_items(['ours_s_mask', 'ours_s_loss',
    #     'ours_s_loss_masked', 'b0_s_loss', 'b1_s_loss',
    #     'ours_q_loss', 'b0_q_loss', 'b1_q_loss']).save_csv(
    #       f'classwise/{mode}', save_path, i)

    # distinguishable episodes
    if not i == outer_steps:
      print(f'Path for saving: {save_path}')
      print(f'End_of_episode: {i}')

    ### end of episode (i) ###

  print(f'End_of_{mode}.')
  # del metadata
  return sampler, result_frame

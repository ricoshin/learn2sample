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
from nn.output import ModelOutput
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

  samples_per_class = 10
  train = True if mode == 'train' else False
  force_base = True
  model_mode = 'metric'
  # split_method = 'inclusive'
  split_method = 'exclusive'

  # 100 classes in total
  if split_method == 'exclusive':
    meta_support, remainder = data.split_class(0.1)  # 10 classes
    meta_query = remainder.sample_class(50)  # 50 classes
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

    fc_lr = 0.01
    metric_lr = 0.01
    import pdb; pdb.set_trace()
    model_fc = Model(len(meta_support), mode='fc')
    model_metric = Model(len(meta_support), mode='metric')

    # baseline parameters
    params_fc = C(model_fc.get_init_params('fc'))
    params_metric = C(model_metric.get_init_params('metric'))

    result_dict = ResultDict()
    episode_iterator = EpisodeIterator(
      support=meta_support,
      query=meta_query,
      split_ratio=0.5,
      resample_every_episode=True,
      inner_steps=inner_steps,
      samples_per_class=samples_per_class,
      num_workers=2,
      pin_memory=True,
    )
    episode_iterator.sample_episode()

    for k, (meta_s, meta_q) in enumerate(episode_iterator, 1):
      # import pdb; pdb.set_trace()

      # feed support set
      # [standard network]
      out_s_fc = model_fc(meta_s, params_fc, None)
      with torch.no_grad():
        # embedding space metric
        params_fc_m = C(params_fc.copy('fc_m'))
        out_s_fc_m = model_metric(meta_s, params_fc_m, mask=None)
      # [prototypical network]
      out_s_metric = model_metric(meta_s, params_metric, None)

      # inner gradient step (baseline)
      params_fc = params_fc.sgd_step(
        out_s_fc.loss.mean(), fc_lr, 'no_grad')
      params_metric = params_metric.sgd_step(
        out_s_metric.loss.mean(), metric_lr, 'no_grad')

      # test on query set
      with torch.no_grad():
        if split_method == 'inclusive':
          out_q_fc = model_fc(meta_q, params_fc, mask=None)
        params_fc_m = C(params_fc.copy('fc_m'))
        out_q_fc_m = model_metric(meta_q, params_fc_m, mask=None)
        out_q_metric = model_metric(meta_q, params_metric, mask=None)

      if split_method == 'inclusive':
        outs = [
          out_s_fc, out_s_fc_m, out_s_metric, out_q_fc, out_q_fc_m, out_q_metric]
      elif split_method == 'exclusive':
        outs = [out_s_fc, out_s_fc_m, out_s_metric, out_q_fc_m, out_q_metric]

      # record result
      result_dict.append(
        outer_step=epoch * i, inner_step=k, **ModelOutput.as_merged_dict(outs))
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
          msg += Printer.outputs(outs, True)
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

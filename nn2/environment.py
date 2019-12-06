import copy
import random

import torch
from dotmap import DotMap
from loader.metadata import Metadata
from nn2.model import Model
from utils import utils
from utils.helpers import InnerStepScheduler

C = utils.getCudaManager('default')


class Environment(object):
  """RL environment(testbed) in which the agent(sampler) can enjoy the
  experiences of teaching the learner(base - model).
    Sampler: A teacher who schedules the learning especially(or only) by
                selecting desirable teaching material.
    Model: A student who learns under the supervision of of the teacher.
    Environment: A class room where the teacher can meet different students and
        subjects so that he / she can also gradually learn how to teach relying
        on the feedback from each intermittent student evaluation.
  """

  def __init__(self, model, metadata, loader_cfg, data_split_method, mask_unit,
               async_stream=True, sync_baseline=True, query_track=False,
               max_action_collapsed=100):
    assert isinstance(metadata, Metadata)
    self.model = model
    self.model_base = model.new()
    self.metadata = metadata
    self.loader_cfg = loader_cfg
    self.data_split_method = data_split_method
    self.mask_unit = mask_unit
    self.async_stream = async_stream
    self.sync_baseline = sync_baseline
    self.query_track = query_track
    self.max_action_collapsed = max_action_collapsed
    # split meta-support and meta-query set
    self.meta_sup, self.meta_que = metadata.split(data_split_method)
    self.meta_sup, self.meta_que = metadata.split(data_split_method)
    self.device = 'cpu'

  def to(self, device, non_blocking=False):
    self.device = device
    self.model = self.model.to(device, non_blocking=non_blocking)
    self.model_base = self.model_base.to(device, non_blocking=non_blocking)
    return self

  def reset(self):
    """Reset environment(the dataloaders and the model).
    """
    self.meta_s_loader = self.meta_sup.episode_loader(self.loader_cfg)
    self.meta_q_loader = self.meta_que.episode_loader(self.loader_cfg)
    self.meta_s = self.meta_s_loader()
    self.model.reset(len(self.meta_sup))
    self.model_base.reset(len(self.meta_sup))
    self.model_base.copy_state_from(self.model)
    self.acc_prev = None
    self.action_collapsed = 0
    # self.meta_q = self.meta_q_loader()
    # utils.ForkablePdb().set_trace()
    return self(action=None)  # return state only

  def get_mask_with_sparsity(self, mask, sparsity):
    n_ones = int(len(mask) * sparsity)
    n_zeros = len(mask) - n_ones
    mask_new = [0] * n_zeros + [1] * n_ones
    random.shuffle(mask_new)
    return torch.tensor(mask_new).to(mask.device)

  def __call__(self, action=None, loop_manager=None):
    """Simulate one action and return reward and next state.
      Args:
        action: instance/class selection mask
      Returns:
        state, reward
    """
    if not hasattr(self, 'meta_s_loader'):
      raise RuntimeError('Do .reset() first before running .step().')

    m = loop_manager
    info = DotMap()

    if action is None:
      # when no action is provided
      meta_s = self.meta_s
      meta_s_base = self.meta_s
    else:
      # mask: ours
      mask = action.mask
      sparsity = (mask == 1.).sum().float() / len(mask)
      info.ours.mask_sp = sparsity
      # mask: baseline
      mask_base = torch.ones(mask.size())  # all-one base
      # mask_base = self.get_mask_with_sparsity(mask, sparsity)
      # TODO: cannot mask out base_mask for mse_loss btn sampler embedding
      sparsity_base = (mask_base == 1.).sum().float() / len(mask_base)
      info.base.mask_sp = sparsity_base
      # instance/class selection
      if self.mask_unit == 'instance':
        mata_s = self.mata_s.masked_select(mask)
        # TODO: instance mask has not implemented yet
      elif self.mask_unit == 'class':
        meta_s = self.meta_s.classwise.masked_select(mask)
        meta_s_base = self.meta_s.classwise.masked_select(mask_base)

    stream = torch.cuda.Stream()
    stream_base = torch.cuda.Stream() if self.async_stream else stream

    with torch.enable_grad():
      # train (model)
      with torch.cuda.stream(stream):
        self.model.train()
        if meta_s is None:
          utils.forkable_pdb().set_trace()
        state = self.model(data=meta_s, debug=m and m.rank == 0)
        # utils.ForkablePdb().set_trace()
        self.model.zero_grad()
        state.loss.backward()
        self.model.optim.step()
        info.ours.s = state

      # train (baseline)
      with torch.cuda.stream(stream_base):
        #   baseline has to be used just for reward generation,
        #   not for state generation as we don't want to keep it at test time.
        self.model_base.train()
        state_base = self.model_base(data=meta_s_base)
        self.model_base.zero_grad()
        state_base.loss.backward()
        self.model_base.optim.step()
        info.base.s = state_base

    if self.query_track:
      with torch.no_grad():
        meta_q = self.meta_q_loader()
        with torch.cuda.stream(stream):
          info.ours.qt = self.model(data=meta_q)
        with torch.cuda.stream(stream_base):
          info.base.qt = self.model_base(data=meta_q)

    # state (one-step-ahead input for sampler)
    self.meta_s = self.meta_s_loader()
    state.meta_s = self.meta_s

    if action is None:
      return state

    # initial reward
    reward = torch.tensor([0.]).to(self.device)
    # Reward: horizon & sparsity reward (dense)
    reward -= 0.1  # long horizon penalty
    # (meaningless if there's no terminal state)

    # at each end of unrolling
    if m and m.end_of_unroll():
      acc, acc_base = [], []
      self.model.eval()
      self.model_base.eval()
      with torch.no_grad():
        # test (model & baseline)
        for _ in range(m.query_steps):  # TODO: as cfg
          meta_q = self.meta_q_loader()
          with torch.cuda.stream(stream):
            acc.append(self.model(data=meta_q).cls_acc.float().mean())
          with torch.cuda.stream(stream_base):
            acc_base.append(self.model_base(data=meta_q).cls_acc.float().mean())
      # Reward: self performance gain
      acc = sum(acc) / len(acc)
      info.ours.q.acc = acc
      if self.acc_prev:
        self_gain = (acc - self.acc_prev) * 1000
        reward += self_gain
        info.r.self_gain = self_gain
        info.ours.q.acc_prev = self.acc_prev
      else:
        info.r.self_gain = 0
        info.ours.q.acc_prev = 0
      self.acc_prev = acc
      # Reward: relative performance gain
      acc_base = sum(acc_base) / len(acc_base)
      rel_gain = (acc - acc_base) * 1000
      info.base.q.acc = acc_base
      info.r.rel_gain = rel_gain  # update rel_gain
      reward += rel_gain
      # Reward: sparsity
      if rel_gain > 0:
        sp_reward = (1 - sparsity**2) * 0.1
        info.r.sparsity = sp_reward
        reward += sp_reward
      else:
        info.r.sparsity = 0
      if self.sync_baseline:
        # synchoronize baseline with model
        # utils.ForkablePdb().set_trace()
        self.model_base.copy_state_from(self.model)

    # terminal state
    terminal = False
    if action.collapsed:
      self.action_collapsed += 1
      # print(self.action_collapsed)
      if self.action_collapsed >= self.max_action_collapsed:
        terminal = True
        # print('terminal!')
    reward = reward.squeeze().detach()
    return state, reward, info, terminal

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
        subjects so that he / she can also gradually learn how to teach relying on
        the feedback based on each intermittent student evaluation.
  """

  def __init__(self, model, metadata, loader_cfg, data_split_method, mask_unit):
    assert isinstance(metadata, Metadata)
    self.model = model
    self.base_model = model.new()
    self.metadata = metadata
    self.loader_cfg = loader_cfg
    self.data_split_method = data_split_method
    self.mask_unit = mask_unit
    # split meta-support and meta-query set
    self.meta_sup, self.meta_que = metadata.split(data_split_method)
    self.meta_sup, self.meta_que = metadata.split(data_split_method)
    self.device = 'cpu'

  def to(self, device, non_blocking=False):
    self.device = device
    self.model = self.model.to(device, non_blocking=non_blocking)
    self.base_model = self.base_model.to(device, non_blocking=non_blocking)
    return self

  def reset(self):
    """Reset environment(the dataloaders and the model).
    """
    self.meta_s_loader = self.meta_sup.episode_loader(self.loader_cfg)
    self.meta_q_loader = self.meta_que.episode_loader(self.loader_cfg)
    self.model.reset()
    self.base_model.copy_state_from(self.model)
    self.meta_s = self.meta_s_loader()
    # self.meta_q = self.meta_q_loader()
    # utils.ForkablePdb().set_trace()
    return self(action=None)[0]  # return state only

  def get_random_action(self, action_instance, sparsity):
    n_ones = int(len(action_instance) * sparsity)
    n_zeros = len(action_instance) - n_ones
    random_action = [0] * n_zeros + [1] * n_ones
    random.shuffle(random_action)
    return torch.tensor(random_action).to(action_instance.device)

  def __call__(self, action=None, loop_manager=None):
    """Simulate one action and return reward and next state.
      Args:
        action: instance/class selection mask
      Returns:
        state, reward
    """
    if not hasattr(self, 'meta_s_loader'):
      raise RuntimeError('Do .reset() first before running .step().')

    if action is not None:
      # sample mask
      n_iter = 0
      max_iter = 1000
      while True:
        n_iter += 1
        if n_iter >= max_iter:
          print('Resampling number exceeded maximum iteration!')
        mask = action.probs.multinomial(1).data
        if mask.sum() > 0:
          break
      # create baseline mask
      mask_base = torch.ones(mask.size())  # all-one base
      # instance/class selection
      if self.mask_unit == 'instance':
        mata_s = self.mata_s.masked_select(mask)
        # TODO: instance mask has not implemented yet
      elif self.mask_unit == 'class':
        meta_s = self.meta_s.classwise.masked_select(mask)
        # action_mask = self.get_random_action(
        #   mask, 0.5 + (100 - loop_manager.outer_step) / 200)
        # meta_s_base = self.meta_s.classwise.masked_select(action_mask)
        meta_s_base = self.meta_s.classwise.masked_select(mask_base)
    else:
      # when no action is provided
      meta_s = self.meta_s
      meta_s_base = self.meta_s

    with torch.enable_grad():
      # train (model)
      if meta_s is None:
        # when all-zero mask
        utils.ForkablePdb().set_trace()
      state = self.model(data=meta_s)
      # utils.ForkablePdb().set_trace()
      self.model.zero_grad()
      state.loss.mean().backward()
      self.model.optim.step()

      # train (baseline)
      #   baseline has to be used just for reward generation,
      #   not for state generation as we don't want to keep it at test time.
      state_base = self.base_model(data=meta_s_base)
      self.base_model.zero_grad()
      state_base.loss.mean().backward()
      self.base_model.optim.step()

    # state (one-step-ahead input for sampler)
    self.meta_s = self.meta_s_loader()
    state.meta_s = self.meta_s
    # embedding for target of MSE loss
    embed = state_base.embed

    if action is None:
      return state, None, None, None

    # horizon & sparsity reward (dense)
    reward = torch.tensor([0.]).to(self.device)
    reward -= 0.1  # long horizon penalty
    sparsity = (mask == 1.).sum().float() / len(mask)
    sparsity_base = (mask_base == 1.).sum().float() / len(mask_base)
    state.sparsity = sparsity
    state_base.sparsity = sparsity_base
    reward += (1 - sparsity**2) * 0.1  # sparsity reward

    info = DotMap(dict(base=state_base, acc_gain=None))

    # at each end of unrolling
    if loop_manager and loop_manager.end_of_unroll():
      acc = []
      acc_base = []
      with torch.no_grad():
        # test (model & baseline)
        for _ in range(10):  # TODO: as cfg
          meta_q = self.meta_q_loader()
          acc.append(self.model(data=meta_q).acc.float().mean())
          acc_base.append(self.base_model(data=meta_q).acc.float().mean())
      # performance gain reward (sparse)
      acc = sum(acc) / len(acc)
      acc_base = sum(acc_base) / len(acc_base)
      acc_gain = acc - acc_base
      info.acc_gain = acc_gain  # update acc_gain
      reward += acc_gain * 100
      # synchoronize baseline with model
      self.base_model.copy_state_from(self.model)

    reward = reward.squeeze().detach()
    return state, reward, info, embed

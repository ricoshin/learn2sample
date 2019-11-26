import copy
import random

import torch
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
    self.meta_s, self.meta_q = metadata.split(data_split_method)
    self.n_episode = -1
    self.device = 'cpu'

  def to(self, device, non_blocking=False):
    self.device = device
    self.model = self.model.to(device, non_blocking=non_blocking)
    self.base_model = self.base_model.to(device, non_blocking=non_blocking)
    return self

  def reset(self):
    """Reset environment(the dataloaders and the model).
    """
    self.meta_s_loader = self.meta_s.episode_loader(self.loader_cfg)
    self.meta_q_loader = self.meta_q.episode_loader(self.loader_cfg)
    self.model.reset()
    self.base_model.copy_state_from(self.model)
    self.meta_s = self.meta_s_loader()
    self.meta_q = self.meta_q_loader()
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

    self.n_episode += 1

    if action is not None:
      # instance/class selection
      if self.mask_unit == 'instance':
        mata_s = self.mata_s.masked_select(action.instance)
      elif self.mask_unit == 'class':
        meta_s = self.meta_s.classwise.masked_select(action.instance)
        action_random = self.get_random_action(
          action.instance, action.sparsity)
        meta_s_base = self.meta_s.classwise.masked_select(action_random)

    else:
      # when no action is provided
      meta_s = self.meta_s
      meta_s_base = self.meta_s

    # train (model)
    try:
      out_s = self.model(data=meta_s)
    except:
      utils.ForkablePdb().set_trace()
    self.model.zero_grad()
    out_s.loss.mean().backward()
    self.model.optim.step()

    # train (baseline)
    try:
      out_s_base = self.base_model(data=meta_s_base)
    except:
      utils.ForkablePdb().set_trace()
    self.base_model.zero_grad()
    out_s_base.loss.mean().backward()
    self.base_model.optim.step()

    # test (model & baseline)
    with torch.no_grad():
      out_q = self.model(data=self.meta_q)
      out_q_base = self.base_model(data=self.meta_q)

    # input for next step
    self.meta_s = self.meta_s_loader()
    self.meta_q = self.meta_q_loader()

    # state
    state = out_q
    #   incorporate next data into the state
    #     for the sampler that has its own encoder.
    state.meta_s = self.meta_s
    state.meta_q = self.meta_q

    # reward
    reward = torch.tensor([0.]).to(self.device)
    #   long horizon penalty
    reward -= 0.1
    # print(f'{loop_manager and loop_manager.end_of_unroll()}')
    if loop_manager and loop_manager.end_of_unroll():
      #   performance gain reward
      acc = out_s_base.acc.float().mean()
      acc_base = out_s.acc.float().mean()
      acc_gain = acc - acc_base
      print(f'[acc_gain] {acc_gain}')
      # sparsity = (action == 1.0).sum().float() / len(action)
      print(f'[sparsity] {action.sparsity}')
      reward += acc_gain
      if acc_gain >= - 0.005:
        # sparsity reward (only when performance does NOT hurt)
        try:
          reward += 1 - action.sparsity**2
        except:
          utils.ForkablePdb().set_trace()
      # synchoronize baseline with model
      self.base_model.copy_state_from(self.model)

    return state, reward.detach()

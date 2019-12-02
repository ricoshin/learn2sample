import os
import torch
import torch.optim as optim
from nn.model import Model
from torch.distributions import Categorical # equal to multinomial
from utils import utils
import numpy as np

C = utils.getCudaManager('default')

class Environment(object):
  def __init__(self):
    pass

  def reset(self):
    pass

  def step(self, outs):
    [out_s, out_s_b0, out_s_b1, out_q, out_q_b0, out_q_b1] = outs
    alpha = 0.1 # TODO learnable
    beta = 0.1 # TODO learnable
    out_q_acc = out_q.acc.mean()
    out_q_b1_acc = out_q_b1.acc.mean()
    reward = out_q.acc.mean() - out_q_b1.acc.mean()
    # print(f'out_q_acc:{out_q_acc}, out_q_b1_acc:{out_q_b1_acc}, reward: {reward}')
    reward_ = - alpha * (out_s.n_classes == out_s_b1.n_classes)
    reward += reward_
    reward -= alpha * (out_s.n_classes == out_s_b1.n_classes)
    # print(f'out_s.n_classes:{out_s.n_classes}, out_s_b1.n_classes:{out_s_b1.n_classes}')
    # print(f'- alpha * out_s.n_classes == out_s_b1.n_classes: {reward_}')
    # print(f'total reward: {reward}')
    return reward


class Policy(object):
  def __init__(self, sampler, gamma=0.99, learning_rate=0.01):
    self.gamma = gamma
    self.learning_rate = learning_rate
    self.loss_hist = []
    self.rw_hist = []
    self.optimizer = optim.Adam(sampler.parameters(), lr=self.learning_rate)
    self.reset()

  def reset(self):
    self.actions = C(torch.Tensor([]))
    self.rewards = []
    self.n_cls_per_action = []

  def predict(self, action_prob, neg_rw=2):
    # if outputs of sampler are softmax
    distribution = Categorical(action_prob)
    # sample an action from the output of a network 
    action = distribution.sample() 
    action = 1 - action
    
    neg_rw = neg_rw if torch.sum(action) == 0 else 0
    # Add log probability of our chosen action to our history
    self.actions = C(torch.cat([
      self.actions,
      distribution.log_prob(action)
    ])) # use log_prob to construct an equivalent loss function
    self.n_cls_per_action.append(len(action))
    return action.type(torch.FloatTensor), neg_rw

  def update(self):
    R = 0
    rewards = []
    # Discount future rewards back to the present using gamma
    for r, n in zip(self.rewards[::-1], self.n_cls_per_action[::-1]):
      R = r + self.gamma * R
      for i in range(n):
        rewards.insert(0, R)

    # Scale rewards
    rewards = C(torch.FloatTensor(rewards))
    # import pdb; pdb.set_trace()
    # rewards = (rewards - rewards.mean()) / \
    #     (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    loss = C(torch.sum(torch.mul(self.actions, rewards).mul(-1), -1))

    # Update network weights
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # Save and intialize episode history counters
    self.loss_hist.append(loss.item())
    self.rw_hist.append(np.sum(self.rewards))
    self.reset()

  def to_text(self, print_action=False):
    if print_action:
      pass
    return '[rw]:{:.2f}'.format(sum(self.rewards) / len(self.rewards))



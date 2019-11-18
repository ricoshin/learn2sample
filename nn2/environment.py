from loader.metadata import Metadata
from nn2.model import Model
from utils.helpers import InnerStepScheduler
from utils import utils

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
    self.metadata = metadata
    self.loader_cfg = loader_cfg
    self.optim = torch.optim.SGD(lr=0.01)
    self.data_split_method = data_split_method
    self.mask_unit = mask_unit
    # split meta-support and meta-query set
    self.meta_s, self.meta_q = metadata.split(data_split_method)
    self.n_episode = -1

  def reset(self):
    """Reset environment(the dataloaders and the model).
    """
    self.meta_s_loader = self.meta_s.episode_loader(self.loader_cfg)
    self.meta_q_loader = self.meta_q.episode_loader(self.loader_cfg)
    self.model.reset()
    return self.step(action=None)

  def step(self, action=None):
    """Simulate one action and return reward and next state.
      Args:
        action: instance/class selection mask
      Returns:
        reward, state
    """
    if not hasattr(self, 'meta_s_loader'):
      raise RuntimeError('Do .reset() first before running .step().')

    self.n_episode += 1
    meta_s = self.meta_s_loader()

    if action:
      if self.mask_unit == 'instance':
        mata_s = mata_s.masked_select(action)
      elif self.mask_unit == 'class':
        meta_s = meta_s.classwise.masked_select(action)

    out_s = self.model(data=meta_s)
    self.model.zero_grad()
    out_s.loss.backward()
    self.model.optim.step()

from loader.metadata import Metadata
from nn2.model import Model
from utils.helpers import InnerStepScheduler


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

  def __init__(self, metadata, data_split_method, loader_cfg, model_mode,
               gpu_id):
    assert isinstance(metadata, Metadata)
    assert data_split_method in ['inclusive', 'exclusive']
    self.data_split_method = data_split_method
    self.loader_cfg = loader_cfg
    # self.inner_step_scheduler = inner_step_scheduler
    self.model_mode = model_mode
    self.gpu_id = gpu_id
    self.meta_s, self.meta_q = metadata.split(data_split_method)
    self.n_episode = -1
    self.reset()

  def reset(self):
    """Reset environment(the dataloaders and the model).
    """
    if self.data_split_method == 'inclusive':
      self.meta_s_loader = self.meta_s.dataset_loader(self.loader_cfg)
      self.meta_q_loader = self.meta_q.dataset_loader(self.loader_cfg)
    elif self.data_split_method == 'exclusive':
      self.meta_s_loader = self.meta_s.episode_loader(self.loader_cfg)
      self.meta_q_loader = self.meta_q.episode_loader(self.loader_cfg)
    else:
      raise Exception()
    # TODO: fix later!
    n_classes = len(self.metadata) if self.model_mode == 'fc' else None
    self.model = Model(n_classes=n_classes, mode=self.model_mode)
    self.model = self.model.to(self.gpu_id if self.gpu_id >=0 else 'cpu')

  def step(self, action):
    """Simulate one action and return reward and next state.
      Args:
        action: instance/class selection mask
      Returns:
        reward, state
    """
    self.n_episode += 1
    meta_s, meta_q = self.meta_s_loader(), self.meta_q_loader()

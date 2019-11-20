import pdb

import torch
import torch.nn as nn
from dotmap import DotMap
from loader.episode import Dataset, Episode
from nn.output import ModelOutput
from nn.sampler2 import MaskDist
from torch.nn import functional as F
from utils import utils
from utils.helpers import weights_init, BaseModule, Flatten
from utils.torchviz import make_dot
from utils.utils import MyDataParallel
from utils.helpers import OptimGetter

C = utils.getCudaManager('default')


class Model(BaseModule):
  """Base-learner Module."""

  def __init__(self, input_dim, embed_dim, channels, kernels,
               distance_type='cosine', last_layer='metric', optim_getter=None,
               n_classes=None, auto_reset=True, preprocess=None):
    super(Model, self).__init__()
    assert last_layer in ['fc', 'metric']
    assert distance_type in ['euclidean', 'cosine']
    if optim_getter is not None:
      assert isinstance(optim_getter, OptimGetter)
    # fc: conventional learning using fully-connected layer
    # metric: metric-based learning using Euclidean distance (Prototypical Net)
    self.input_dim = input_dim
    self.embed_dim = embed_dim
    self.channels = channels
    self.kernels = kernels
    self.distance_type = distance_type
    self.last_layer = last_layer
    self.ch = [3] + channels  # 1st: 3 channels for input image
    self.kn = kernels
    self.optim_getter = optim_getter
    self.n_classes = n_classes
    self.input_dim = input_dim
    self._embed_dim = None
    self.preprocess = preprocess
    if auto_reset:
      self.reset()

  # @property
  # def embed_dim(self):
  #   """compute embedding dimension by feeding a dummy input."""
  #   if not hasattr(self, 'layers'):
  #     self.reset()
  #   if self._embed_dim is None:
  #     x = torch.zeros([1] + self.input_dim)  # dummy x
  #     x = self.preprocess(x)
  #     for layer in self.layers.values():
  #       x = layer(x)
  #     self._embed_dim = x.view(1, -1).size(-1)
  #   return self._embed_dim

  def new(self):
    # deepcopy?
    return Model(self.input_dim, self.embed_dim, self.channels, self.kernels,
                 self.distance_type, self.last_layer, self.optim_getter,
                 self.n_classes).to(self.device)

  def copy_state_from(self, sampler_src):
    self.load_state_dict(sampler_src.state_dict())
    self.to(self.device)
    self.optim = self.optim_getter(self.parameters())

  def reset(self):
    self.layers = self.build().to(self.device)
    if self.optim_getter:
      self.optim  = self.optim_getter(self.parameters())
      # # reset optimizer with new parameters
      # defaults_backup = self.optim.defaults
      # self.optim = self.optim.__class__(self.layers.parameters())
      # self.optim.defaults = defaults_backup
    return self

  def build(self):
    layers = nn.ModuleDict()
    for i in range(len(self.ch) - 1):
      # conv blocks
      layers.update([
          [f'conv_{i}', nn.Conv2d(self.ch[i], self.ch[i + 1], self.kn[i])],
          # [f'norm_{i}', nn.GroupNorm(self.n_group, self.ch[i + 1])],
          [f'bn_{i}', nn.BatchNorm2d(
              self.ch[i + 1], track_running_stats=False)],
          [f'relu_{i}', nn.ReLU(inplace=True)],
          [f'mp_{i}', nn.MaxPool2d(2, 2)],
      ])
    # compute last feature dimension
    x = torch.zeros([1] + self.input_dim)  # dummy x
    if self.preprocess:
      x = self.preprocess(x)
    for layer in layers.values():
      x = layer(x)
    x_dim = x.view(1, -1).size(-1)
    # linear layers
    layers.update({f'flatten': Flatten()})
    layers.update({f'fc_embed': nn.Linear(x_dim, self.embed_dim)})
    if self.last_layer == 'fc':
      layers.update({f'fc_class': nn.Linear(self.embed_dim, self.n_classes)})
    layers.apply(weights_init)
    return layers

  def forward(self, data, debug=False):
    """
    Args:
      data (loader.Dataset or loader.Episode):
        loader.episode.Dataset:
          Support or query set (standard learning / 'fc' mode)
        loader.episode.Epidose:
          Support and query set (meta-learning / 'metric' mode)

    Returns:
      nn.output.ModelOutput
    """
    if not hasattr(self, 'layers'):
      raise RuntimeError('Do .reset() first!')

    # images and labels
    if isinstance(data, Dataset):
      # when using Meta-dataset
      x = data.imgs  # [n_cls*n_ins, 3, 84, 84]
      y = data.labels  # [n_cls*n_ins]
      classwise_fn = data.get_view_classwise_fn()
    elif isinstance(data, Episode):
      # when using customized bi-level dataset
      x = data.concatenated.imgs
      y = data.concatenated.labels
      classwise_fn = data.q.get_view_classwise_fn()
    else:
      raise Exception(f'Unknown type: {type(data)}')

    if self.preprocess:
      x = self.preprocess(x)

    for layer in self.layers.values():
      x = layer(x)

    n_samples = x.size(0)
    x_embed = x.view(n_samples, -1)  # [n_samples, embed_dim(576)]
    dist = self._pairwise_dist(
        support_embed=x_embed[:n_samples // 2],
        query_embed=x_embed[n_samples // 2:],
        labels=data.q.labels,
    )

    if self.last_layer == 'fc':
      # fix later
      logits = x.view(x.size(0), self.n_classes)
    elif self.last_layer == 'metric':
      logits = dist
      y = data.q.labels
    else:
      raise Exception(f'Unknown last_layer: {self.last_layer}')

    loss = F.cross_entropy(logits, y, reduction='none')
    acc = logits.argmax(dim=1) == y

    if debug:
      utils.ForkablePdb().set_trace()

    # returns output
    return DotMap(dict(
        loss=loss,
        acc=acc,
        dist=dist,
        logits=logits,
        labels=y,
    ))

  def _pairwise_dist(self, support_embed, query_embed, labels):
    proto_types = []
    for i in set(labels.tolist()):
      proto_types.append(support_embed[labels == i].mean(0, keepdim=True))
    proto_types = torch.cat(proto_types, dim=0)
    n_classes = proto_types.size(0)  # prototypes: [n_classes, embed_dim]
    n_samples = query_embed.size(0)  # queries: [n_samples, embed_dim]
    proto_types = proto_types.unsqueeze(0).expand(n_samples, n_classes, -1)
    query_embed = query_embed.unsqueeze(1).expand(n_samples, n_classes, -1)
    if self.distance_type == 'euclidean':
      # [n_samples, n_classes, dim]
      pairwise_dist = (proto_types - query_embed)**2
      pairwise_dist = pairwise_dist.sum(dim=2) * (-1)
    elif self.distance_type == 'cosine':
      proto_types = F.normalize(proto_types, p=2, dim=2)
      query_embed = F.normalize(query_embed, p=2, dim=2)
      # [n_samples, n_classes, dim]
      pairwise_dist = proto_types * query_embed
      pairwise_dist = pairwise_dist.sum(dim=2)
    return pairwise_dist  # [n_samples, n_classes]

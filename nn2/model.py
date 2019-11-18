import pdb

import DotMap
import torch
import torch.nn as nn
from loader.episode import Dataset, Episode
from nn.output import ModelOutput
from nn.sampler2 import MaskDist
from torch.nn import functional as F
from utils import utils
from utils.torchviz import make_dot
from utils.utils import MyDataParallel

C = utils.getCudaManager('default')


class Model(nn.Module):
  """Base-learner Module."""

  def __init__(self, last_layer, distance_type, optim, lr, n_classes=None):
    super(Model, self).__init__()
    assert last_layer in ['fc', 'metric']
    assert distance_type in ['euclidean', 'cosine']
    # fc: conventional learning using fully-connected layer
    # metric: metric-based learning using Euclidean distance (Prototypical Net)
    self.last_layer = last_layer
    self.distance_type = distance_type
    self.ch = [3, 64, 64, 64, 64]
    self.kn = [3, 3, 3, 3]
    self.optim_ = optim
    self.lr = lr
    self.n_classes = n_classes

  def reset(self):
    layers = nn.ModuleDict()
    for i in range(len(self.ch) - 1):
      layers.update([
          [f'conv_{i}', nn.Conv2d(self.ch[i], self.ch[i + 1], self.kn[i])],
          # [f'norm_{i}', nn.GroupNorm(self.n_group, self.ch[i + 1])],
          [f'bn_{i}', nn.BatchNorm2d(
              self.ch[i + 1], track_running_stats=False)],
          [f'relu_{i}', nn.ReLU(inplace=True)],
          [f'mp_{i}', nn.MaxPool2d(2, 2)],
      ])
    if self.last_layer == 'fc':
      layers.update({f'last_conv': nn.Conv2d(self.ch[-1], self.n_classes, 3)})
    self.layers = C(layers)
    self.optim = getattr(torch.optim, self.optim_)(
        self.layers.parameters(), lr=self.lr)

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

  def forward(self, data, debug=False):
    """
    Args:
      data (loader.Dataset or loader.Episode):
        loader.episode.Dataset:
          Support or query set (standard learning / 'fc' mode)
        loader.episode.Epidose:
          Support and query set (meta-learning / 'metric' mode)
      params (nn.model.Params):
        model parameters that can be updated outside of the module.
      mask (torch.FloatTensor):
        Classwise weighting overlay mask.
        Controls the effects from each classes.
        torch.Size([n_cls*n_ins, 1])

                               |   input drop    |    softmax drop
                               | (hard_mask=True)|  (hard_mask=False)
    ----------------------------------------------------------------
    Gradient-based  |    fc    |        x        |         o
       Learning     |  metric  |        x        |         o
    ----------------------------------------------------------------
    Reinforcement   |    fc    |        o        |         o
       Learning     |  metric  |        o        |         x

    *In case of the metric-based learning, softmax units are automatically
    reduced, whereas additional treatments are needed for the FC layer.

    Returns:
      nn.output.ModelOutput
    """
    if hasattr(self, layers):
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
      raise Exception()

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
      logits = x.view(x.size(0), self.n_classes)
    elif self.last_layer == 'metric':
      logits = dist
      y = data.q.labels
    else:
      raise Exception()

    loss = F.cross_entropy(logits, y)
    acc = logits.argmax(dim=1) == y

    if debug:
      utils.ForkablePdb().set_trace()

    # returns output
    return DotMap(dict(
        loss=loss,
        acc=acc,
        dist=dist,
        logits=logits,
    ))

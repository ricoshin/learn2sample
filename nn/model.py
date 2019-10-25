import pdb

import torch
import torch.nn as nn
from nn.output import ModelOutput
from torch.nn import functional as F
from utils.torchviz import make_dot
from utils.utils import MyDataParallel
from loader.episode import Dataset, Episode


class Params(object):
  """Parameters for Model(base-learner)."""

  def __init__(self, params_dict, name):
    super(Params, self).__init__()
    assert isinstance(params_dict, dict)
    assert isinstance(name, str)
    self._dict = params_dict
    self.name = name

  def __repr__(self):
    return f"Params({repr([p for p in self._dict.keys()])})"

  def __getitem__(self, value):
    if isinstance(value, int):
      item = self.param_names()[value]
    elif isinstance(value, str):
      item = self._dict[value]
    else:
      raise KeyError(f'Wrong type!: {type(value)}')
    return item

  @classmethod
  def from_module(cls, module, name):
    assert isinstance(module, nn.Module)
    return cls({k: v for k, v in module.named_parameters()}, name)

  def cuda(self, device=None, debug=False):
    dict_ = {k: v.cuda(device) for k, v in self._dict.items()}
    if debug:
      import pdb
      pdb.set_trace()
    return Params(dict_, self.name)

  def copy(self, name):
    dict_ = {k: v.clone() for k, v in self._dict.items()}
    return Params(dict_, name).detach_().requires_grad_()

  def with_name(self, name):
    return Params(self._dict, name)

  def requires_grad_(self):
    for k in self._dict.keys():
      self._dict[k].requires_grad_(True)
    return self

  def detach_(self):
    for k in self._dict.keys():
      self._dict[k].detach_()
    return self

  def param_names(self):
    return list(self._dict.keys())

  def sgd_step(self, loss, lr, grad_mode="no_grad", detach_param=True,
               debug=False):
    assert grad_mode in ['no_grad', 'first', 'second']
    params = self._dict
    with torch.set_grad_enabled(grad_mode != 'no_grad'):
      grads = torch.autograd.grad(
          loss, params.values(),
          create_graph=(grad_mode == 'second'),
          allow_unused=True,
          )
      if debug:
        pdb.set_trace()
      new_params = {}
      for (name, param), grad in zip(params.items(), grads):
        if detach_param:
          param = param.detach()
        new_params[name] = param - lr * grad
    return Params(new_params, self.name).requires_grad_()


class Model(nn.Module):
  """Base-learner Module. Its parameters cannot be updated with the graph
  connected from outside world. To bypass this issue, parameters should not be
  registered as normal parameters but keep their modularity independent to the
  model. The forward pass has to be dealt with functionals by taking over those
  parameters as a funtion argument."""

  def __init__(self, n_classes, mode):
    super(Model, self).__init__()
    assert mode in ['fc', 'metric']
    # fc: conventional learning using fully-connected layer
    # metric: metric-based learning using Euclidean distance (Prototypical Net)
    self.mode = mode
    self.ch = [3, 64, 64, 64, 128]
    self.kn = [5, 5, 3, 3]
    self.n_classes = n_classes
    self.nll_loss = nn.NLLLoss(reduction='none')
    self.n_group = 8
    # if self.n_classes is not None:
    # self.classifier = nn.Conv2d(ch[-1], self.n_classes, 3)

  def get_init_params(self, name='ex'):
    assert isinstance(name, str)
    layers = nn.ModuleDict()
    for i in range(len(self.ch) - 1):
      layers.update([
          [f'conv_{i}', nn.Conv2d(self.ch[i], self.ch[i + 1], self.kn[i])],
          [f'norm_{i}', nn.GroupNorm(self.n_group, self.ch[i + 1])],
          [f'relu_{i}', nn.ReLU(inplace=True)],
          [f'mp_{i}', nn.MaxPool2d(2, 2)],
      ])
    if self.mode == 'fc':
      layers.update({f'last_conv': nn.Conv2d(self.ch[-1], self.n_classes, 3)})
    return Params.from_module(layers, name)

  def _forward(self, x, p):
    assert isinstance(p, (Params, MyDataParallel))
    p = p._dict
    for i in range(len(self.ch) - 1):
      x = F.conv2d(x, p[f'conv_{i}.weight'], p[f'conv_{i}.bias'])
      x = F.group_norm(x, self.n_group, weight=p[f'norm_{i}.weight'],
                       bias=p[f'norm_{i}.bias'])
      x = F.relu(x, inplace=True)
      x = F.max_pool2d(x, 2, 2)
    if self.mode == 'fc':
      x = F.conv2d(x, p[f'last_conv.weight'], p[f'last_conv.bias'], 3)
    return x

  def _euclidean_dist(self, proto_types, query_embed):
    n_classes = proto_types.size(0)  # [n_classes, embed_dim]
    n_samples = query_embed.size(0)  # [n_samples, embed_dim]
    proto_types = proto_types.unsqueeze(0).expand(n_samples, n_classes, -1)
    query_embed = query_embed.unsqueeze(1).expand(n_samples, n_classes, -1)
    logit = -((proto_types - query_embed)**2).sum(dim=2)
    return logit  # [n_samples, n_classes]

  def _metric_based_logit(self, support_embed, query_data, params):
    # reshaping functions
    classwise = query_data.get_view_classwise_fn()
    flatten = lambda x: x.view(x.size(0), -1)
    # prototypes: [n_classes, embed_dim]
    proto_types = classwise(flatten(support_embed)).mean(dim=1)
    # query embedings: [n_samples, embed_dim]
    query_embed = flatten(self._forward(query_data.imgs, params).squeeze())
    return self._euclidean_dist(proto_types, query_embed)

  def forward(self, data, params, mask=None, debug=False):
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

    Returns:
      nn.output.ModelOutput
    """
    assert isinstance(params, (Params, MyDataParallel))

    # images and labels
    if self.mode == 'fc':
      assert isinstance(data, Dataset)
      x = data.imgs  # [n_cls*n_ins, 3, 84, 84]
      y = data.labels  # [n_cls*n_ins]
    elif self.mode == 'metric':
      assert isinstance(data, Episode)
      x = data.s.imgs
      y = data.s.labels

    # funtional forward
    x = self._forward(x, params).squeeze()  # [n_cls*n_ins, n_cls, 1, 1]

    if self.mode == 'metric':
      x = self._metric_based_logit(x, data.q, params)
      y = data.q.labels

    # import pdb; pdb.set_trace()

    if debug:
      pdb.set_trace()

    # loss
    x = F.log_softmax(x, dim=1)  # [n_cls*n_ins, n_cls]
    loss = self.nll_loss(x, y)  # [n_cls*n_ins]

    return ModelOutput(
        params_name=params.name,
        dataset_name=data.name,
        n_classes=data.n_classes,
        loss_sample=loss,
        log_softmax=x,
        label=y,
        mask=mask)

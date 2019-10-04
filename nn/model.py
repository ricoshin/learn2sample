import torch
import torch.nn as nn
from nn.output import ModelOutput
from torch.nn import functional as F
from utils.torchviz import make_dot


class Params(nn.Module):
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

  def cuda(self, device):
    self._dict = {k: v.cuda(device) for k, v in self._dict.items()}
    return self

  def clone(self):
    dict_ = {k: v.clone() for k, v in self._dict.items()}
    return Params(dict_, self.name)

  def copy(self):
    return self.clone().detach_().requires_grad_()

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

  def detach_requiresd_grad(self):
    dict_ = {k: v.detach().requires_grad_(True)
             for k, v in self._dict.items()}
    return Params(dict_, self.name)

  def param_names(self):
    return list(self._dict.keys())

  def sgd_step(self, loss, lr, grad_mode="no_grad", debug=False):
    assert grad_mode in ['no_grad', 'first', 'second']
    params = self._dict
    with torch.set_grad_enabled(grad_mode != 'no_grad'):
      grads = torch.autograd.grad(
          loss, params.values(), create_graph=(grad_mode == 'second'),
          allow_unused=True)
      if debug:
        import pdb; pdb.set_trace()
      new_params = {}
      for (name, param), grad in zip(params.items(), grads):
        new_params[name] = param - lr * grad
    return Params(new_params, self.name).requires_grad_()


class Model(nn.Module):
  """Base-learner Module. Its parameters cannot be updated with the graph
  connected from outside world. To bypass this issue, parameters should not be
  registered as normal parameters but keep their modularity independent to the
  model. The forward pass has to be dealt with functionals by taking over those
  parameters as a funtion argument."""

  def __init__(self, n_classes):
    super(Model, self).__init__()
    self.ch = [3, 64, 64, 64, 128]
    self.kn = [5, 5, 3, 3]
    self.n_classes = n_classes
    self.nll_loss = nn.NLLLoss(reduction='none')
    self.n_group = 1
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
    layers.update({f'last_conv': nn.Conv2d(self.ch[-1], self.n_classes, 3)})
    return Params.from_module(layers, name)

  def _forward(self, x, p):
    assert isinstance(p, Params)
    p = p._dict
    for i in range(len(self.ch) - 1):
      x = F.conv2d(x, p[f'conv_{i}.weight'], p[f'conv_{i}.bias'])
      x = F.group_norm(x, self.n_group, weight=p[f'norm_{i}.weight'],
                       bias=p[f'norm_{i}.bias'])
      x = F.relu(x, inplace=True)
      x = F.max_pool2d(x, 2, 2)
    x = F.conv2d(x, p[f'last_conv.weight'], p[f'last_conv.bias'], 3)
    return x

  def forward(self, dataset, params, mask=None, debug=False):
    """
    Args:
      dataset (loader.Dataset):
        Support or query set(imgs/labels/ids).
      params (nn.model.Params)
      mask (torch.FloatTensor):
        Classwise weighting overlay mask.
        Controls the effects from each classes.
        torch.Size([n_cls*n_ins, 1])

    Returns:
      loss (torch.FloatTensor): cross entropy loss.
      acc (torch.FloatTensor): accuracy of classification.
    """
    assert isinstance(params, Params)
    if self.n_classes is None:
      raise RuntimeError(
          'forward() with Model.n_class=None is only meant to be called in '
          'get_init_parameters() just for getting initial parameters.')

    x = dataset.imgs  # [n_cls*n_ins, 3, 84, 84]
    y = dataset.labels  # [n_cls*n_ins]

    # funtional forward
    x = self._forward(x, params)  # [n_cls*n_ins, n_cls, 1, 1]
    if debug:
      import pdb
      pdb.set_trace()

    # loss
    x = F.log_softmax(x.squeeze(), dim=1)  # [n_cls*n_ins, n_cls]
    loss = self.nll_loss(x, y)  # [n_cls*n_ins]

    return ModelOutput(
        params_name=params.name, dataset_name=dataset.name,
        element_loss=loss, log_softmax=x, label=y, mask=mask)

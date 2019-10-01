import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.torchviz import make_dot


class Params(nn.Module):
  """Parameters for Model(base-learner)."""
  def __init__(self, params_dict):
    super(Params, self).__init__()
    assert isinstance(params_dict, dict)
    self._dict = params_dict

  def __repr__(self):
    return f"Params({repr([p for p in self._dict.keys()])})"

  def __getitem__(self, value):
    if isinstance(value, int):
      item = self.names()[value]
    elif isinstance(value, str):
      item = self._dict[value]
    else:
      raise KeyError(f'Wrong type!: {type(value)}')
    return item

  @classmethod
  def from_module(cls, module):
    assert isinstance(module, nn.Module)
    return cls({k: v for k, v in module.named_parameters()})

  def cuda(self, device):
    self._dict = {k: v.cuda(device) for k, v in self._dict.items()}
    return self

  def clone(self):
    dict_ = {k: v.clone() for k, v in self._dict.items()}
    return Params(dict_)

  def detach_(self):
    for k in self._dict.keys():
      self._dict[k].detach_().requires_grad_(True)

  def detach(self):
    dict_ = {k: v.detach().requires_grad_(True)
             for k, v in self._dict.items()}
    return Params(dict_)

  def names(self):
    return list(self._dict.keys())

  def sgd_step(self, loss, lr, detach_p=False, detach_g=False,
               second_order=False):
    params = self._dict
    if detach_p is True and detach_g is True:
      raise Exception('parameters and gradients should not be detached both.')
    create_graph = not detach_g and second_order
    grads = torch.autograd.grad(
      loss, params.values(), create_graph=create_graph)
    new_params = {}
    for (name, param), grad in zip(params.items(), grads):
      if detach_p:
        param = param.detach()
      if detach_g:
        grad = grad.detach()
      new_params[name] = param - lr * grad
    return Params(new_params)


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

  def get_init_params(self):
    layers = nn.ModuleDict()
    for i in range(len(self.ch) - 1):
      layers.update([
        [f'conv_{i}', nn.Conv2d(self.ch[i], self.ch[i + 1], self.kn[i])],
        [f'norm_{i}', nn.GroupNorm(self.n_group, self.ch[i + 1])],
        [f'relu_{i}', nn.ReLU(inplace=True)],
        [f'mp_{i}', nn.MaxPool2d(2, 2)],
      ])
    layers.update({f'last_conv': nn.Conv2d(self.ch[-1], self.n_classes, 3)})
    return Params.from_module(layers)

  def _forward(self, x, p):
    assert isinstance(p, Params)
    p = p._dict
    for i in range(len(self.ch) - 1):
      x = F.conv2d(x, p[f'conv_{i}.weight'], p[f'conv_{i}.bias'])
      x = F.group_norm(
          x, self.n_group, weight=p[f'norm_{i}.weight'], bias=p[f'norm_{i}.bias'],
        )
      x = F.relu(x, inplace=True)
      x = F.max_pool2d(x, 2, 2)
    x = F.conv2d(x, p[f'last_conv.weight'], p[f'last_conv.bias'], 3)
    return x

  def forward(self, dataset, params, mask=None, debug=False):
    """
    Args:
      dataset (loader.Dataset):
        Support or query set(imgs/labels/ids).
      params (networks.model.Params)
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
    n_samples = dataset.n_samples  # n_ins
    view_classwise = dataset.get_view_classwise_fn()

    # funtional forward
    x = self._forward(x, params)  # [n_cls*n_ins, n_cls, 1, 1]
    if debug:
      import pdb; pdb.set_trace()

    # loss and accuracy
    x = F.log_softmax(x.squeeze(), dim=1)  # [n_cls*n_ins, n_cls]
    loss = self.nll_loss(x, y)  # [n_cls*n_ins]
    acc = (x.argmax(dim=1) == y).float()

    if mask is None:
      conf = x.exp().max(dim=1)[0]  # confidence
      conf_pp = conf.gather(0, y).mean()  # for prediceted positive
      id_tp = x.argmax(dim=1) == y
      id_fp = x.argmax(dim=1) != y

      if all(id_tp):
        conf_fp = torch.tensor(0.0)
        conf_tp = conf[id_tp].mean()
      elif all(id_fp):
        conf_tp = torch.tensor(0.0)
        conf_fp = conf[id_fp].mean()
      else:
        conf_tp = conf[id_tp].mean()  # for true positive
        conf_fp = conf[id_fp].mean()  # for false positive

      if torch.isnan(conf_tp):
        print('nan!')
        import pdb; pdb.set_trace()

      return loss.mean(), acc.mean(), [conf_pp, conf_tp, conf_fp]
    # weighted average by mask
    else:
      loss = view_classwise(loss).mean(1, keepdim=True)
      acc = view_classwise(acc).mean(1, keepdim=True)
      mask_sum = mask.sum().detach()
      loss_w = (loss * mask).sum() / mask_sum  # weighted averaged loss
      acc_w = (acc * mask).sum() / mask_sum  # weighted averaged acc
      return loss, acc, loss_w, acc_w

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.torchviz import make_dot


class Params(nn.Module):
  def __init__(self, params_dict):
    super(Params, self).__init__()
    assert isinstance(params_dict, dict)
    self._dict = params_dict

  @classmethod
  def from_module(cls, module):
    assert isinstance(module, nn.Module)
    return cls({k: v for k, v in module.named_parameters()})

  def cuda(self):
    self._dict = {k: v.cuda() for k, v in self._dict.items()}
    return self

  def clone(self):
    dict_ = {k: v.clone() for k, v in self._dict.items()}
    return Params(dict_)

  def detach_(self):
    for k in self._dict.keys():
      self._dict[k].detach_().requires_grad_(True)

  def detach(self):
    dict_ = {k: v.detach().requires_grad_(True) for k, v in self._dict.items()}
    return Params(dict_)

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
  def __init__(self, n_classes):
    super(Model, self).__init__()
    self.ch = [3, 100, 150, 200, 250]
    self.kn = [5, 5, 3, 3]
    self.n_classes = n_classes
    self.nll_loss = nn.NLLLoss(reduction='none')
    # if self.n_classes is not None:
      # self.classifier = nn.Conv2d(ch[-1], self.n_classes, 3)

  def get_init_params(self):
    layers = nn.ModuleDict()
    for i in range(len(self.ch) - 1):
      layers.update([
        [f'conv_{i}', nn.Conv2d(self.ch[i], self.ch[i + 1], self.kn[i])],
        [f'bn_{i}', nn.BatchNorm2d(self.ch[i + 1], track_running_stats=False)],
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
      x = F.batch_norm(
          x, running_mean=None, running_var=None, training=True,
          weight=p[f'bn_{i}.weight'], bias=p[f'bn_{i}.bias'],
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
      mask (torch.FloatTensor):
        Classwise weighting overlay mask.
        Controls the effects from each classes.
        torch.Size([n_cls*n_ins, 1])
      detach_params (bool):
        This flag can be used to prevent from computing second derivative of
        model parameters, while stil keeping those piplelines open heading to
        the sampler parameters.

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
      return loss.mean(), acc.mean()
    # weighted average by mask
    else:
      # to match dimension
      mask = mask.repeat([1, n_samples])  # [n_cls, 1]
      loss = view_classwise(loss) * mask # [n_cls, n_ins]
      acc = view_classwise(acc) * mask
      # weighted loss by sampler mask
      loss_w = (loss * mask).sum() / mask.sum().detach()
      acc_w = (acc * mask).sum() / mask.sum().detach()
      return loss, acc, loss_w, acc_w

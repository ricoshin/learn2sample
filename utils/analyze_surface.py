import random
import torch
import torch.nn.functional as F
from models.model_helpers import ParamsFlattener
from nn.maskgen_topk import MaskGenerator as manual_mask_gen
from utils import utils

C = utils.getCudaManager('default')


# def eval_mask_by_lipschitz(params, mask):
#   assert all([isinstance(in, ParamsFlattener) for in in [params, mask]])
#

# def eval_step_direction(model_clas, data, params, n_grid=200, std=1e-4):

def eval_gauss_var(model_cls, data, params, n_sample=200, std=1e-4):
  assert isinstance(params, ParamsFlattener)
  params = params.detach()
  losses = []
  for _ in range(n_sample):
    p = {}
    for k, v in params.unflat.items():
      p[k] = v + C(torch.zeros(v.size()).normal_(0, std))
    params_perturbed = ParamsFlattener(p)
    model = C(model_cls(params=params_perturbed))
    losses.append(model(*data['in_train'].load()))
  return torch.stack(losses).var()

def inversed_masked_params(params, mask, step, r, thres=0.5):
  inv_mask = {k: (m < thres) for k, m in mask.unflat.items()}
  # import pdb; pdb.set_trace()
  num_ones = {k: v.sum() for k, v in inv_mask.items()}
  for k, v in inv_mask.items():
    diff = num_ones[k] - mask.t_size(0)[k] * r[f"sparse_{k.split('_')[1]}"]
    if diff >= 0:
      m = inv_mask[k]
      nonzero_ids = m.nonzero().squeeze().tolist()
      drop_ids = random.sample(nonzero_ids, diff.tolist())
      m[drop_ids] = torch.tensor(0)
    else:
      m = inv_mask[k]
      zero_ids = (m == 0).nonzero().squeeze().tolist()
      drop_ids = random.sample(zero_ids, -diff.tolist())
      m[drop_ids] = torch.tensor(1)
    inv_mask[k] = inv_mask[k].float()
  inv_mask = ParamsFlattener(inv_mask)
  mask_layout = inv_mask.expand_as(params)
  step_sparse = step * mask_layout
  params_sparse = params + step_sparse
  params_pruned = params_sparse.prune(inv_mask)
  return params_pruned, params_sparse


def random_masked_params(params, mask, step, r, thres=0.5):
  assert isinstance(params, ParamsFlattener)
  # assert all([isinstance(arg, ParamsFlattener) for arg in (params, mask)])
  params = params.detach()
  step = step.detach()
  r = {'layer_' + k.split('_')[1]: v for k, v in r.items()}
  mask_rand = manual_mask_gen.randk(set_size=mask.tsize(0), topk=r)
  mask_layout = mask_rand.expand_as(params)
  step_sparse = step * mask_layout
  params_sparse = params + step_sparse
  params_pruned = params_sparse.prune(mask_rand > thres)
  return params_pruned, params_sparse


def eval_lipschitz(params, i):
  assert isinstance(params, ParamsFlattener)
  params = params.unflat
  for k, v in params.items():
    params[k] = v.clone().detach().requires_grad_(False)
  input_size = params[f'mat_{i}'].size(0)
  def affine_fun(x):
    return torch.matmul(x, params[f'mat_{i}']) + params[f'bias_{i}']
  return generic_power_method(affine_fun, [1, input_size])[0]


def generic_power_method(affine_fun, input_size, eps=1e-8,
                         max_iter=1000, use_cuda=False):
  """ Return the highest singular value of the linear part of
  `affine_fun` and it's associated left / right singular vectors.

  INPUT:
      * `affine_fun`: an affine function
      * `input_size`: size of the input
      * `eps`: stop condition for power iteration
      * `max_iter`: maximum number of iterations
      * `use_cuda`: set to True if CUDA is present

  OUTPUT:
      * `eigenvalue`: maximum singular value of `affine_fun`
      * `v`: the associated left singular vector
      * `u`: the associated right singular vector

  NOTE:
      This algorithm is not deterministic, depending of the random
      initialisation, the returned eigenvectors are defined up to the sign.

      If affine_fun is a PyTorch model, beware of setting to `False` all
      parameters.requires_grad.

  TEST::
      >>> conv = nn.Conv2d(3, 8, 5)
      >>> for p in conv.parameters(): p.requires_grad = False
      >>> s, u, v = generic_power_method(conv, [1, 3, 28, 28])
      >>> bias = conv(torch.zeros([1, 3, 28, 28]))
      >>> linear_fun = lambda x: conv(x) - bias
      >>> torch.norm(linear_fun(v) - s * u) # should be very small

  TODO: more tests with CUDA
  """
  zeros = C(torch.zeros(input_size))
  bias = affine_fun(zeros)

  def linear_fun(x): return affine_fun(x) - bias
  # bias removed affine function (= gradient of affine_fun)
  #   f(x) - f(0) = M (where f(x) = Mx +b)

  def norm(x, p=2):
    """ Norm for each batch

    FIXME: avoid it?
    """
    norms = C(torch.zeros(x.shape[0]))
    for i in range(x.shape[0]):
      norms[i] = x[i].norm(p=p)
    return norms

  # Initialise with random values
  v = C(torch.randn(input_size))
  v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_size)

  stop_criterion = False
  it = 0

  while not stop_criterion:
    previous = v
    v = _norm_gradient_sq(linear_fun, v)  # M'Mv
    v = F.normalize(v.view(v.shape[0], -1), p=2, dim=1).view(input_size)
    stop_criterion = (torch.norm(v - previous) < eps) or (it > max_iter)
    it += 1
  # Compute Rayleigh product to get eivenvalue
  u = linear_fun(v)  # unormalized left singular vector
  eigenvalue = norm(u)  # ||Mv||_2: approx. Lipshitz constant for affine_fun
  u = u.div(eigenvalue)
  return eigenvalue, u, v


def _norm_gradient_sq(linear_fun, v):
  v = v.clone().detach().requires_grad_(True)
  loss = torch.norm(linear_fun(v))**2
  # g(v) = ||f(v) - f(0)||_2^2  = ||M||_2^2
  loss.backward()
  # partial_wrt_v(g(v)) = M'Mv
  return v.grad.data  # detached


def test():
  # conv = C(torch.nn.Conv2d(3, 8, 5))
  # for p in conv.parameters(): p.requires_grad = False
  bias = C(torch.randn(10))
  small = C(torch.randn(10, 10))
  large = C(torch.randn(10, 10) * 10)

  def small_fn(x): return x.matmul(small) + bias

  def large_fn(x): return x.matmul(large) + bias

  s_small, _, _ = generic_power_method(small_fn, [1, 10])
  s_large, _, _ = generic_power_method(large_fn, [1, 10])

  import pdb
  pdb.set_trace()

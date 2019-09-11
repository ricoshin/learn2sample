#import sys
#sys.path.append('/home/user/Personal/Baek/CS671/loss-landscape')
#from plot_surface import *
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from nn.maskgen_topk import MaskGenerator as MaskGenerator2
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import torch
from tqdm import tqdm
from models.model_helpers import ParamsIndexTracker, ParamsFlattener
from utils import utils
C = utils.getCudaManager('default')
#debug_sigint = utils.getDebugger('SIGINT')
#debug_sigstp = utils.getDebugger('SIGTSTP')
def analyzing_mask(mask_gen, layer_size, mode, iteration, iter_interval, text_dir):
    sparse_r = {}
    #if mode == 'test' and (iteration % iter_interval == 0):
    if (iteration % iter_interval == 0):
      mask = mask_gen.sample_mask()
      mask = ParamsFlattener(mask)
      for k, v in layer_size.items():
        r = (mask > 0.5).sum().unflat[k].tolist() / v
        sparse_r[f"sparse_{k.split('_')[1]}"] = r
      alpha = mask_gen.a
      beta = mask_gen.b
      expectation = alpha / (alpha + beta)
      #variance =  (alpha * beta) / ((alpha + beta) * (alpha + beta) * (alpha + beta +1))
      bern_n = 1
      #import pdb; pdb.set_trace()
      variance =  bern_n * (alpha * beta) * (alpha + beta + bern_n) / ((alpha + beta) * (alpha + beta) * (alpha + beta +1))
      sorted_layer_0 = np.sort(tensor2numpy(expectation[0:layer_size['layer_0']]))[::-1]
      mask_thr_layer_0 = sorted_layer_0[int(sparse_r['sparse_0'] * layer_size['layer_0'])]
      sorted_layer_1 = np.sort(tensor2numpy(expectation[layer_size['layer_0']: layer_size['layer_0']+layer_size['layer_1']]))[::-1]
      mask_thr_layer_1 = sorted_layer_1[int(sparse_r['sparse_1'] * layer_size['layer_1'])]
      drop = tensor2numpy(expectation) < tensor2numpy(expectation.median())
      sparse_r_drop = np.concatenate(((tensor2numpy(expectation) < mask_thr_layer_0)[0:layer_size['layer_0']], (tensor2numpy(expectation) < mask_thr_layer_1)[layer_size['layer_0']: layer_size['layer_0']+layer_size['layer_1']]), axis=0)
      for index, drop in enumerate([drop, sparse_r_drop]):
        drop = np.array(drop, dtype=int)
        retain = 1 - drop
        certain = tensor2numpy(variance < variance.median())
        uncertain = 1 - certain
        certain_drop = certain * drop
        certain_retain = certain * retain
        uncertain_drop = uncertain * drop
        uncertain_retain = uncertain * retain
        if index == 0:
          print("\niteration {} (median) certain drop : {} certain retain : {} uncertain drop : {} uncertain_retain : {}\n".format(iteration, certain_drop.sum(), certain_retain.sum(), uncertain_drop.sum(), uncertain_retain.sum()))
        else:
          print("\niteration {} (sparse) certain drop : {} certain retain : {} uncertain drop : {} uncertain_retain : {}\n".format(iteration, certain_drop.sum(), certain_retain.sum(), uncertain_drop.sum(), uncertain_retain.sum()))
    

def sampling_mask(mask_gen, layer_size, model_train, params, sample_num=10000, mode='test', iteration=0, iter_interval=10, result_dir='result/mask_compare'):
    topk = True
    sparse_r = {}
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    mask_result = None
    #import pdb; pdb.set_trace()
    #if mask=='test' and (iteration % iter_interval == 0):
    if (iteration % iter_interval == 0):
        for i in range(sample_num):
            mask = mask_gen.sample_mask()
            mask = ParamsFlattener(mask)
            mask_flat = (mask.flat>0.5).float()
            mask.unflat
            if i == 0:
                mask_cat = mask_flat.unsqueeze(dim=0)
                mask_sum = mask
            else:
                mask_cat = torch.cat((mask_cat, mask_flat.unsqueeze(dim=0)), dim=0)
                mask_sum += mask
        #import pdb; pdb.set_trace()
        mask_mean, mask_var = mask_cat.mean(dim=0).squeeze(), mask_cat.var(dim=0).squeeze()
        
        mask_sum /= sample_num
        mask = mask>0.5

        grad = model_train.params.grad.detach()
        act = model_train.activations.detach()
        for k, v in layer_size.items():
            r = (mask > 0.5).sum().unflat[k].tolist() / v
            sparse_r[f"sparse_{k.split('_')[1]}"] = r

        topk_mask_gen = MaskGenerator2.topk if topk else MaskGenerator2.randk
        layer_0_topk = topk_mask_gen(grad=grad, set_size=layer_size, topk=sparse_r['sparse_0'])._unflat['layer_0'].view(-1)
        layer_0_topk = layer_0_topk>0.5
        layer_1_topk = topk_mask_gen(grad=grad, set_size=layer_size, topk=sparse_r['sparse_1'])._unflat['layer_1'].view(-1)
        layer_1_topk = layer_1_topk>0.5

        layer_0_prefer_topk = topk_mask_gen(grad=mask_sum.expand_as(params), set_size=layer_size, topk=sparse_r['sparse_0'])._unflat['layer_0'].view(-1)
        layer_1_prefer_topk = topk_mask_gen(grad=mask_sum.expand_as(params), set_size=layer_size, topk=sparse_r['sparse_1'])._unflat['layer_1'].view(-1)
        layer_0_prefer_topk = layer_0_prefer_topk>0.5
        layer_1_prefer_topk = layer_1_prefer_topk>0.5
        layer_0 = torch.cat([grad.unflat['mat_0'],
        grad.unflat['bias_0'].unsqueeze(0)], dim=0).abs().sum(0)
        layer_1 = torch.cat([grad.unflat['mat_1'],
        grad.unflat['bias_1'].unsqueeze(0)], dim=0).abs().sum(0)
        layer_0_abs = ParamsFlattener({'layer_0': layer_0})
        layer_1_abs = ParamsFlattener({'layer_1': layer_1})

        hist_mean, bins_mean  = np.histogram(tensor2numpy(mask_mean), bins=20)
        hist_var, bins_var = np.histogram(tensor2numpy(mask_var), bins=20)

        sorted_layer_0 = np.sort(tensor2numpy(mask_mean[0:layer_size['layer_0']]))[::-1]
        mask_thr_layer_0 = sorted_layer_0[int(sparse_r['sparse_0'] * layer_size['layer_0'])]
        sorted_layer_1 = np.sort(tensor2numpy(mask_mean[layer_size['layer_0']: layer_size['layer_0']+layer_size['layer_1']]))[::-1]
        mask_thr_layer_1 = sorted_layer_1[int(sparse_r['sparse_1'] * layer_size['layer_1'])]
        drop = tensor2numpy(mask_mean) < tensor2numpy(mask_mean.median())
        sparse_r_drop = np.concatenate(((tensor2numpy(mask_mean) < mask_thr_layer_0)[0:layer_size['layer_0']], (tensor2numpy(mask_mean) < mask_thr_layer_1)[layer_size['layer_0']: layer_size['layer_0']+layer_size['layer_1']]), axis=0)
        for index, drop in enumerate([drop, sparse_r_drop]):
            drop = np.array(drop, dtype=int)
            retain = 1 - drop
            certain = tensor2numpy(mask_var < mask_var.median())
            uncertain = 1 - certain
            certain_drop = certain * drop
            certain_retain = certain * retain
            uncertain_drop = uncertain * drop
            uncertain_retain = uncertain * retain
            if index == 0:
                print("\niteration {} (median) certain drop : {} certain retain : {} uncertain drop : {} uncertain_retain : {}\n".format(iteration, certain_drop.sum(), certain_retain.sum(), uncertain_drop.sum(), uncertain_retain.sum()))
            else:
                print("\niteration {} (sparse) certain drop : {} certain retain : {} uncertain drop : {} uncertain_retain : {}\n".format(iteration, certain_drop.sum(), certain_retain.sum(), uncertain_drop.sum(), uncertain_retain.sum()))
        #import pdb; pdb.set_trace()

        sparse_r, overlap_mask_ratio_0, overlap_mask_ratio_1, overlap_prefer_ratio_0, overlap_prefer_ratio_1 = plot_masks(mask, layer_0_topk, layer_1_topk, mask_sum, layer_0_prefer_topk, layer_1_prefer_topk, 
                        layer_0, layer_1, result_dir, iteration, sparse_r, mask_mean, mask_var)
        mask_result =  dict(
                    sparse_0 = sparse_r['sparse_0'],
                    sparse_1 = sparse_r['sparse_1'],
                    overlap_mask_ratio_0 = overlap_mask_ratio_0.tolist(),
                    overlap_mask_ratio_1 = overlap_mask_ratio_1.tolist(),
                    overlap_prefer_ratio_0 = overlap_prefer_ratio_0.tolist(),
                    overlap_prefer_ratio_1 = overlap_prefer_ratio_1.tolist(),
                    certain_drop=certain_drop,
                    certain_retain=certain_retain,
                    uncertain_drop=uncertain_drop,
                    uncertain_retain=uncertain_retain
                    )

    return mask_result

def plot_masks(mask, layer_0_topk, layer_1_topk, mask_sum, layer_0_prefer_topk, layer_1_prefer_topk, layer_0, layer_1, result_dir, iteration, sparse_r, mask_mean, mask_var):
    plot_mask(tensor2numpy(mask._unflat['layer_0']), os.path.join(result_dir,'iter{}_mask_layer0.png'.format(iteration)), 'sparse ratio - layer 0 = {}'.format(sparse_r['sparse_0']))
    plot_mask(tensor2numpy(mask._unflat['layer_1']), os.path.join(result_dir,'iter{}_mask_layer1.png'.format(iteration)), 'sparse ratio - layer 1 = {}'.format(sparse_r['sparse_1']))

    #import pdb; pdb.set_trace()
    plot_mask(tensor2numpy(layer_0_topk), os.path.join(result_dir,'iter{}_topk_layer0.png'.format(iteration)), 'topk - layer 0 ({})'.format(sparse_r['sparse_0']))
    plot_mask(tensor2numpy(layer_1_topk), os.path.join(result_dir,'iter{}_topk_layer1.png'.format(iteration)), 'topk - layer 1 ({})'.format(sparse_r['sparse_1']))

    plot_mask(tensor2numpy(layer_0_prefer_topk), os.path.join(result_dir,'iter{}_prefer_topk_layer0.png'.format(iteration)), 'prefer topk - layer 0 ({})'.format(sparse_r['sparse_0']))
    plot_mask(tensor2numpy(layer_1_prefer_topk), os.path.join(result_dir,'iter{}_prefer_topk_layer1.png'.format(iteration)), 'prefer topk - layer 1 ({})'.format(sparse_r['sparse_1']))

    plot_mask(tensor2numpy(mask_sum._unflat['layer_0']), os.path.join(result_dir, 'iter{}_mask_sum_layer0.png'.format(iteration)), 'prefer - layer 0 ({})'.format(sparse_r['sparse_0']))
    plot_mask(tensor2numpy(mask_sum._unflat['layer_1']), os.path.join(result_dir, 'iter{}_mask_sum_layer1.png'.format(iteration)), 'prefer - layer 1 ({})'.format(sparse_r['sparse_1']))

    plot_mask(tensor2numpy(layer_0), os.path.join(result_dir,'iter{}_grad_layer0.png'.format(iteration)), 'grad - layer 0 ({})'.format(sparse_r['sparse_0']))
    plot_mask(tensor2numpy(layer_1), os.path.join(result_dir,'iter{}_grad_layer1.png'.format(iteration)), 'grad - layer 1 ({})'.format(sparse_r['sparse_1']))

    #import pdb; pdb.set_trace()
    overlap_mask_0 = layer_0_topk * mask._unflat['layer_0']
    overlap_mask_1 = layer_1_topk * mask._unflat['layer_1']

    overlap_mask_ratio_0 = overlap_mask_0.sum().float()/(len(overlap_mask_0) * sparse_r['sparse_0'])
    overlap_mask_ratio_1 = overlap_mask_1.sum().float()/(len(overlap_mask_1) * sparse_r['sparse_1'])

    plot_mask(tensor2numpy(overlap_mask_0), os.path.join(result_dir,'iter{}_overlap_mask_layer0.png'.format(iteration)), 
                '{}/{}(overlap/topk) overlap ratio({:.2f})'.format(overlap_mask_0.sum(),int(sparse_r['sparse_0'] * len(overlap_mask_0)), overlap_mask_ratio_0))
    plot_mask(tensor2numpy(overlap_mask_1), os.path.join(result_dir,'iter{}_overlap_mask_layer1.png'.format(iteration)), 
                '{}/{}(overlap/topk) overlap ratio({:.2f})'.format(overlap_mask_1.sum(),int(sparse_r['sparse_1'] * len(overlap_mask_1)), overlap_mask_ratio_1))

    #import pdb; pdb.set_trace()
    overlap_prefer_0 = layer_0_topk * layer_0_prefer_topk
    overlap_prefer_1 = layer_1_topk * layer_1_prefer_topk

    overlap_prefer_ratio_0 = overlap_prefer_0.sum().float()/(len(overlap_prefer_0) * sparse_r['sparse_0'])
    overlap_prefer_ratio_1 = overlap_prefer_1.sum().float()/(len(overlap_prefer_1) * sparse_r['sparse_1']) 
    #import pdb; pdb.set_trace()
    plot_mask(tensor2numpy(overlap_prefer_0), os.path.join(result_dir,'iter{}_overlap_prefer_layer0.png'.format(iteration)), 
                '{}/{}(overlap/topk) overlap ratio({:.2f})'.format(overlap_prefer_0.sum(),int(sparse_r['sparse_0'] * len(overlap_prefer_0)), overlap_prefer_ratio_0))
    plot_mask(tensor2numpy(overlap_prefer_1), os.path.join(result_dir,'iter{}_overlap_prefer_layer1.png'.format(iteration)), 
                '{}/{}(overlap/topk) overlap ratio({:.2f})'.format(overlap_prefer_1.sum(),int(sparse_r['sparse_1'] * len(overlap_prefer_1)), overlap_prefer_ratio_1))
    
    plot_dist(tensor2numpy(mask_var), os.path.join(result_dir, 'iter{}_dist_var.png'.format(iteration)), 'distribution of mask uncertainty')
    plot_dist(tensor2numpy(mask_mean), os.path.join(result_dir, 'iter{}_dist_mean.png'.format(iteration)),'distribution of mask retain prob')
    plot_scatter(tensor2numpy(mask_mean), tensor2numpy(mask_var), os.path.join(result_dir, 'iter{}_mean-var.png'.format(iteration)), 'distribution of mask mean-var')
    return sparse_r, overlap_mask_ratio_0, overlap_mask_ratio_1, overlap_prefer_ratio_0, overlap_prefer_ratio_1

def plot_mask_result(mask_result, result_dir):
    plot_1D(mask_result['sparse_0'], os.path.join(result_dir,'sparse_0.png'), 'sparse ratio of layer 0')
    plot_1D(mask_result['sparse_1'], os.path.join(result_dir,'sparse_1.png'), 'sparse ratio of layer 1')
    plot_1D(mask_result['overlap_mask_ratio_0'], os.path.join(result_dir,'overlap_mask_ratio_0.png'), 'overlap ratio between topk/mask of layer 0')
    plot_1D(mask_result['overlap_mask_ratio_1'], os.path.join(result_dir,'overlap_mask_ratio_1.png'), 'overlap ratio between topk/mask of layer 1')
    plot_1D(mask_result['overlap_prefer_ratio_0'], os.path.join(result_dir,'overlap_prefer_ratio_0.png'), 'overlap ratio between topk/prefer of layer 0')
    plot_1D(mask_result['overlap_prefer_ratio_1'], os.path.join(result_dir,'overlap_prefer_ratio_1.png'), 'overlap ratio between topk/prefer of layer 1')

def plot_loss(model_cls, model, params, input_data, dataset, feature_gen, mask_gen, step_gen, scale_way, xmin=-2.0, xmax=0.5, num_x=20, mode='test', iteration=0, iter_interval=10, loss_dir='result/draw_loss'):
    #if mode == 'test' and ((iteration-1) % iter_interval == 0) and (iteration >1):
    if ((iteration-1) % iter_interval == 0) and (iteration >1):
        X = np.linspace(xmin, xmax, num_x)
        Y = np.linspace(xmin, xmax, num_x)

        model_train = C(model_cls(params=params.detach()))
        #step_data = data['in_train'].load()
        step_data = input_data
        train_nll, train_acc = model_train(*step_data)
        train_nll.backward()

        g = model_train.params.grad.flat.detach()
        w = model_train.params.flat.detach()

        feature, v_sqrt = feature_gen(g)

        size = params.size().unflat()
        kld = mask_gen(feature, size)
        # step & mask genration
        mask = mask_gen.sample_mask()
        mask = ParamsFlattener(mask)
        mask_layout = mask.expand_as(params)
        step_X = step_gen(feature, v_sqrt)
        step_X = params.new_from_flat(step_X[0]) * mask_layout
        step_X = step_X.flat.view(-1)
        step_Y = model_train.params.grad.flat.view(-1) 
        step_X_ = params.new_from_flat(-1.0 * step_X)
        step_X2_ = params.new_from_flat(-10.0 * step_X)
        step_Y_ = params.new_from_flat(1.0 * step_Y)
        step_Y2_ = params.new_from_flat(0.1 * step_Y)
        #import pdb; pdb.set_trace()
        #step_Y = step_Y * step_X.abs().sum() / step_Y.abs().sum()
        L2_X = (step_X * step_X).sum()
        L2_Y = (step_Y * step_Y).sum()

        layer_settings = [['mat_0', 'bias_0', 'mat_1', 'bias_1'], ['mat_0', 'bias_0'], ['mat_1', 'bias_1']]
        normalize_way = 'filter_norm'
        #result_dirs = ['loss_all_scale_{}'.format(scale_way), 'loss_layer0_scale_{}'.format(scale_way), 'loss_layer1_scale_{}'.format(scale_way)]
        result_dirs = ['loss_all_scale_{}'.format(scale_way)]
        for layer_set, result_dir in zip(layer_settings, result_dirs):
            grad_dir = os.path.join(loss_dir, normalize_way, result_dir, 'gradient')
            if not os.path.exists(grad_dir):
                os.makedirs(grad_dir)
            step_dir = os.path.join(loss_dir, normalize_way, result_dir, 'step')
            if not os.path.exists(step_dir):
                os.makedirs(step_dir)
            step_X_ = params.new_from_flat(-1.0 * step_X)
            step_Y_ = params.new_from_flat(1.0 * step_Y)
            #step_X2_ = params.new_from_flat(-10.0 * step_X)
            #step_Y2_ = params.new_from_flat(0.1 * step_Y)

            if normalize_way is not None:
              for step in (step_X_, step_Y_):
                for matrix in ['mat_0', 'bias_0', 'mat_1', 'bias_1']:
                  di = step.unflat[matrix]
                  norm_di = torch.norm(di,2, dim=0)
                  thetai = params.unflat[matrix]
                  if normalize_way == 'filter_norm':
                    norm_thetai = torch.norm(di,2, dim=0) ## TODO Division by zero bug fix
                    normalize_di = di * norm_thetai / (norm_di+1e-5)
                  elif normalize_way == 'weight_norm':
                    normalize_di = di * thetai / (di + 1e-5)
                  step.unflat[matrix] = normalize_di

            abs_X = step_X_.abs().sum()
            abs_Y = step_Y_.abs().sum()
            L2_X = (step_X_ * step_X_).sum()
            L2_Y = (step_Y_ * step_Y_).sum()
            scale_X, scale_Y = 0, 0
                
            #import pdb; pdb.set_trace()
            for layer in ['mat_1', 'bias_1']:
                scale_X += abs_X.unflat[layer].item()
                scale_Y += abs_Y.unflat[layer].item()
            
            scale_g_s = scale_X/scale_Y
            scale_s_g = scale_Y/scale_X

            for layer in layer_set:
                scale_X += abs_X.unflat[layer].item()
                scale_Y += abs_Y.unflat[layer].item()
            
            #import pdb; pdb.set_trace()
            #step_Y_ = step_Y_ * (step_X_.abs().sum() /  step_Y_.abs().sum())
            if scale_way == 's':
                scale_s = scale_s_g
                scale_g = 1.0
            elif scale_way == 'g':
                scale_s = 1.0
                scale_g = scale_g_s
            else:
                scale_s = 1.0
                scale_g = 1.0       
            Z_X = get_1D_Loss(X, step_X_, scale_s, step_X.size(), layer_set, dataset, model_cls, params)
            Z_Y = get_1D_Loss(Y, step_Y_, scale_g, step_Y.size(), layer_set, dataset, model_cls, params)
            #Z_X2 = get_1D_Loss(X, step_X2_, scale, step_X.size(), layer_set, data['in_train'], model_cls, params)
            #Z_Y2 = get_1D_Loss(Y, step_Y2_, scale,step_Y.size(), layer_set,data['in_train'], model_cls, params)
            plot_2d(X, Z_X, os.path.join(step_dir, 'iter_{:04d}_STEPxMASK_1dLoss.png'.format(iteration)), 'L1 norm = {:.2f}'.format(scale_X*scale_s))
            plot_2d(Y, Z_Y, os.path.join(grad_dir, 'iter_{:04d}_1.0xGradient_1dLoss.png'.format(iteration)),'L1 norm = {:.2f}'.format(scale_Y*scale_g))
            #plot_2d(Y, Z_X2, os.path.join(result_dir,'iter_{}_10xStepxMASK_1dLoss.png'.format(iteration)))
            #plot_2d(Y, Z_Y2, os.path.join(result_dir,'iter_{}_0.1xGradient_1dLoss.png'.format(iteration)))
    #import pdb; pdb.set_trace()

def get_1D_Loss(X, step_X_, scale, flat_size, layers, dataset, model_cls, params):
  Z = np.zeros(len(X))
  temp = np.zeros(len(X))
  for axis_x in tqdm(range(len(X))):
    #import pdb; pdb.set_trace()
    direction = params.new_from_flat(torch.zeros(flat_size).cuda())
    for layer in layers:
      direction.unflat[layer] = step_X_.unflat[layer]
    X_scale = params.new_from_flat(scale * X[axis_x] * torch.ones(flat_size).cuda())
    step_X__ =  X_scale * direction
    
    params_z = params + step_X__
    model_train = C(model_cls(params=params_z.detach()))
    for index in range(len(dataset)):
      input_data = dataset.load()
      train_nll, _ = model_train(*input_data)
      temp[axis_x] = train_nll.item()/input_data[0].size(0)
      if index==0:
        Z[axis_x] = temp[axis_x]
      else:
        Z[axis_x] += temp[axis_x]
  return Z
def plot_1D(data, savefig='mask.png', title=None):
  sns.set()
  x = np.linspace(1, len(data), len(data))
  fig = plt.figure()
  plt.plot(x,data)
  plt.title(title)
  if savefig is not None:
    plt.savefig(savefig)
    plt.close()
  return fig
def plot_3D(X=None, Y=None, Z=None, savefig='test.png'):
  """
  ======================
  3D surface (color map)
  ======================

  Demonstrates plotting a 3D surface colored with the coolwarm color map.
  The surface is made opaque by using antialiased=False.

  Also demonstrates using the LinearLocator and custom formatting for the
  z axis tick labels.
  """
  sns.set()
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  # Make data.
  if Z is None:
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    #R = np.sqrt(X**2 + Y**2)
    #Z = np.sin(R)
    Z = np.sqrt(X**2 + Y**2)
  # Plot the surface.
  surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

  # Customize the z axis.
  ax.set_zlim(Z.reshape(-1, 1).min(), Z.reshape(-1, 1).max())
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.xlabel('Step Generator')
  plt.ylabel('Gradient')
  plt.savefig(savefig)
  plt.close()
  #plt.show()

def plot_contour(X=None, Y=None, Z=None, savefig='test.png'):
  """
  fig = plt.figure()
  plt.title("Contour plots")
  plt.contour(X, Y, Z, alpha=.75, cmap='jet')
  """
  sns.set()
  fig, ax = plt.subplots()
  CS = ax.contour(X, Y, Z)
  ax.clabel(CS, inline=1, fontsize=10)
  #ax.set_title('Simplest default with labels')
  plt.xlabel('Step Generator')
  plt.ylabel('Gradient')
  plt.savefig(savefig)
  plt.close()

def plot_mask(mask, savefig='mask.png', title=None):
  sns.set()
  x = np.linspace(1,len(mask),len(mask))
  y = mask
  extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
  #y = mask/mask.mean()
  
  fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)

  ax.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
  ax.set_yticks([])
  ax.set_xlim(extent[0], extent[1])
  plt.title(title)

  ax2.plot(x,y)
  #plt.tight_layout()
  
  """
  fig = plt.figure()
  plt.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent)
  #plt.set_yticks([])
  #plt.set_xlim(extent[0], extent[1])
  """
  plt.title(title)
  plt.savefig(savefig)
  plt.close()
  #plt.show()
  
def plot_2d(x, y, savefig='mask.png', title=None):
  sns.set()
  fig = plt.figure()
  plt.title(title)
  plt.plot(x,y)
  plt.savefig(savefig)
  plt.close()
  #plt.show()

def plot_2d2(x, y, savefig='mask.png', title=None):
  axis = np.linspace(1, len(x), len(x))
  sns.set()
  fig = plt.figure()
  plt.title(title)
  plt.plot(axis, x, label='mean')
  plt.plot(axis, y, label='var')
  plt.legend(loc='upper left')
  plt.savefig(savefig)
  plt.close()
def plot_dist(x, savefig='mask.png', title=None):
  sns.set()
  fig = plt.figure()
  sns.distplot(x)
  plt.title(title)
  plt.savefig(savefig)
  plt.close()
def plot_scatter(x,y, savefig='boxplot.png', title=None):
  sns.set()
  fig = plt.figure()
  #tips = sns.load_dataset("tips")
  channel_dim = len(x)
  #x = np.linspace(1, channel_dim, channel_dim)
  for i in range(channel_dim):
    #import pdb; pdb.set_trace()
    #plt.plot(x, data[i, :])
    plt.scatter(x[i], y[i], marker='o', color='blue')
  
  #sns.catplot(x="channel", y="retain", kind="swarm", data=array(data))
  plt.xlabel('mean')
  plt.ylabel('var')
  plt.title(title)
  plt.savefig(savefig)
  plt.close()
  #import pdb; pdb.set_trace()

def tensor2numpy(x):
  return np.array(x.detach().cpu())

def filter_norm(direction):
  return direction
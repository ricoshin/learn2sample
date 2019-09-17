import numpy as np
import torch
from utils import utils
from loader import MetaDataset, PseudoMetaDataset
from networks.model import Model
from networks.sampler import EncoderClass, EncoderInstance, Sampler

C = utils.getCudaManager('default')
sig_1 = utils.getSignalCatcher('SIGINT')
sig_2 = utils.getSignalCatcher('SIGTSTP')

def loop(train=True):

  metadata = MetaDataset(split='train')
  # metadata = PseudoMetaDataset()

  outer_steps = 1000
  inner_steps = 150
  unroll_steps = 10
  # update_steps = inner_steps * 2
  update_steps = unroll_steps
  log_steps = 5
  inner_lr = 1.0
  outer_lr = 0.001

  sampler = C(Sampler())
  sampler.train()
  # outer_optim = torch.optim.SGD(sampler.parameters(), lr=outer_lr)
  outer_optim = torch.optim.Adam(sampler.parameters(), lr=outer_lr)

  for i in range(1, outer_steps):
    outer_loss = 0

    for j, epi in enumerate(metadata.loader(n_batches=1), 1):
      try:
        epi.s = C(epi.s)
        epi.q = C(epi.q)
      except:  # OOM
        continue
      view_classwise = epi.s.get_view_classwise_fn()
      # view_elementwise = epi.s.get_view_elementwise_fn()

      xs = sampler.enc_ins(epi.s.imgs)
      xs = view_classwise(xs)
      xs = sampler.enc_cls(xs)
      # state = C(sampler.mask_gen.init_state(epi.n_classes))
      mask = loss_s = C(sampler.mask_gen.init_mask(epi.n_classes))

      model = Model(epi.n_classes)
      params = C(model.get_init_params())
      params2 = params.clone().detach()
      # import pdb; pdb.set_trace()

      for j in range(1, inner_steps):
        debug_1 = sig_1.is_active()
        debug_2 = sig_2.is_active(inner_steps % 10 ==0)

        mask = sampler.mask_gen(xs, mask, loss_s)  # class mask
        # mask = C(torch.ones(epi.n_classes, 1))

        loss_s, acc_s, loss_s_w, acc_s_w = model(
          epi.s, params, mask, debug_1)
        loss_s2, acc_s2 = model(epi.s, params2, None)
        loss_s2 = loss_s2 * mask.mean().detach()

        if debug_2:
          import pdb; pdb.set_trace()

        params = params.sgd_step(
          loss_s_w, inner_lr, second_order=True)
        params2 = params2.sgd_step(
          loss_s2, inner_lr, second_order=False).detach()

        loss_q_m, acc_q_m = model(epi.q, params, mask=None)
        loss_q2, acc_q2 = model(epi.q, params2, mask=None)

        if j % log_steps == 0:
          print(
            f'[out:{i:4d}/{outer_steps}|in:{j:4d}/{inner_steps}]'
            f'ways:{(mask > 0.5).sum().tolist():2d}/{epi.n_classes:2d}|'
            f'[{"|".join([f"{m:4.2f}" for m in mask.squeeze().tolist()])}]|'
            f'S/Q:{epi.s.n_samples:2d}/{epi.q.n_samples:2d}|'
            f'S:{loss_s.mean().tolist():5.2f}(w.{loss_s_w.tolist():4.2f})/'
            f'{acc_s.mean().tolist()*100:5.2f}(w.{acc_s_w.tolist()*100:4.1f})%|'
            f'Q:{loss_q_m.tolist():6.3f}(m.{np.log(epi.n_classes):3.1f})/'
            f'{acc_q_m.mean().tolist()*100:4.1f}%|'
            f'[Base]S:{loss_s2:5.2f}/{acc_s2*100:4.1f}%|'
            f'Q:{loss_q2:6.3f}/{acc_q2*100:4.1f}%|'
          )
        if train and j % unroll_steps == 0:
          outer_loss += loss_q_m
          outer_loss.backward(retain_graph=True)
          outer_loss = 0
          params.detach_()
          sampler.detach_()
        if train and j % update_steps == 0:
          outer_optim.step()
          sampler.zero_grad()
      print()
  import pdb; pdb.set_trace()
  print('end_of_loop')

# def meta_train

if __name__ == '__main__':
  loop(train=True)

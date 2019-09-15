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
  inner_steps = 500
  unroll_steps = 3
  inner_lr = 0.1
  outer_lr = 0.05

  sampler = C(Sampler())
  sampler.train()
  outer_optim = torch.optim.SGD(sampler.parameters(), lr=outer_lr)

  for i in range(outer_steps):
    outer_loss = 0

    for j, epi in enumerate(metadata.loader(n_batches=1)):
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

      sampler.mask_gen.initialze(epi.n_classes)
      model = Model(epi.n_classes)
      params = C(model.get_init_params())
      # import pdb; pdb.set_trace()

      for j in range(inner_steps):
        debug_1 = sig_1.is_active()
        debug_2 = sig_2.is_active()

        mask = sampler.mask_gen(xs)  # class mask
        loss_s_m, acc_s_m, loss_s_w, acc_s_w = model(
          epi.s, params, mask, debug_1)

        if debug_2:
          import pdb; pdb.set_trace()

        params = params.sgd_step(
          loss_s_w, inner_lr, second_order=True)

        if train and (j + 1) % unroll_steps == 0:
          loss_q_m, acc_q_m = model(epi.q, params, mask=None)
          outer_loss += loss_q_m
          print(
            f'[outer:{i:4d}/{outer_steps}|inner:{j:4d}/{inner_steps}]'
            f'ways:{(mask > 0.5).squeeze().sum().tolist()}/{epi.n_classes:2d}|'
            f'S/Q:{epi.s.n_samples:2d}/{epi.q.n_samples:2d}|'
            f'loss_s:{loss_s_m.tolist():6.3f}({loss_s_w.tolist():6.3f})|'
            f'acc_s:{acc_s_m.tolist():6.3f}({acc_s_w.tolist():6.3f})|'
            f'loss_q:{loss_q_m.tolist():6.3f}/{np.log(epi.n_classes):6.3f}|'
            f'acc_q:{acc_q_m.tolist():6.3f}/100.0'
          )
          sampler.zero_grad()
          outer_loss.backward()
          outer_optim.step()
          outer_loss = 0
          params.detach_()
          sampler.mask_gen.detach_()
  import pdb; pdb.set_trace()
  print('end_of_loop')

# def meta_train

if __name__ == '__main__':
  loop(train=True)

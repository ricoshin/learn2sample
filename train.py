import torch
from loader import MetaDataset, PseudoMetaDataset
from networks.sampler import EncoderInstance, EncoderClass, Sampler
from networks.model import Model

if __name__ == '__main__':

  metadata = MetaDataset(split='train')
  # metadata = PseudoMetaDataset()


  # init_model = Model().cuda()
  # init_params = init_model.get_params()
  outer_steps = 1000
  inner_steps = 1000
  unroll_steps = 3
  log_step = 5
  inner_lr = 0.1
  outer_lr = 0.05

  sampler = Sampler().cuda()
  sampler.train()
  outer_optim = torch.optim.SGD(sampler.parameters(), lr=outer_lr)


  for i in range(outer_steps):
    outer_loss = 0


    for j, epi in enumerate(metadata.loader(n_batches=1)):
      epi.s = epi.s.cuda()
      epi.q = epi.q.cuda()
      view_classwise = epi.s.get_view_classwise_fn()
      # view_elementwise = epi.s.get_view_elementwise_fn()

      xs = sampler.enc_ins(epi.s.imgs)
      xs = view_classwise(xs)
      xs = sampler.enc_cls(xs)

      sampler.mask_gen.initialze(epi.n_classes)
      model = Model(epi.n_classes)
      params = model.get_init_params().cuda()
      # import pdb; pdb.set_trace()

      for j in range(inner_steps):
        mask = sampler.mask_gen(xs)  # class mask
        # xs = view_classwise(epi.s.imgs)  # entire support set
        # xs = xs * m  # masking operation
        # xs = view_elementwise(xs)
        inner_loss_s, acc_s = model(epi.s, params, mask)
        # import pdb; pdb.set_trace()
        try:
          params = params.sgd_step(
            inner_loss_s, inner_lr, second_order=True)
        except:
          import pdb; pdb.set_trace()


        if (j + 1) % unroll_steps == 0:
          inner_loss_q, acc_q = model(epi.q, params)  # without mask
          outer_loss += inner_loss_q
          print(
            f'[iteration {j:4d}] '
            f'loss_s:{inner_loss_s.tolist():3.4f}|acc_s:{acc_s.tolist():3.4f}|'
            f'loss_q:{inner_loss_q.tolist():3.4f}|acc_q:{acc_q.tolist():3.4f}'
          )

          sampler.zero_grad()
          outer_loss.backward()
          outer_optim.step()
          outer_loss = 0
          params.detach_()
          sampler.mask_gen.detach_()
          # import pdb; pdb.set_trace()

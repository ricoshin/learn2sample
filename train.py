from loader import MetaDataset, PseudoMetaDataset
from networks.sampler import EncoderInstance, EncoderClass, Sampler
from networks.model import Model

if __name__ == '__main__':

  metadata = MetaDataset(split='train')
  # metadata = PseudoMetaDataset()

  sampler = Sampler().cuda()
  sampler.train()

  init_params = Model().cuda().get_params()
  outer_steps = 1000
  inner_steps = 5
  inner_lr = 0.1
  outer_lr = 0.1

  for i in range(outer_steps):
    outer_loss = 0


    for j, epi in enumerate(metadata.loader(n_batches=1)):
      epi.s.imgs = epi.s.imgs.cuda()
      epi.s.labels = epi.s.labels.cuda()
      view_classwise = epi.s.get_view_classwise_fn()
      # view_elementwise = epi.s.get_view_elementwise_fn()

      xs = sampler.enc_ins(epi.s.imgs)
      xs = view_classwise(xs)
      xs = sampler.enc_cls(xs)

      sampler.mask_gen.initialze(epi.n_classes)
      model = Model(epi.n_classes).init_with(init_params).cuda()
      model.train()

      for j in range(inner_steps):
        mask = sampler.mask_gen(xs)  # class mask
        # xs = view_classwise(epi.s.imgs)  # entire support set
        # xs = xs * m  # masking operation
        # xs = view_elementwise(xs)
        inner_loss_s, acc = model(epi.s, mask)
        # grad = model.compute_grad(inner_loss, init_params)
        model.sgd_step(inner_loss_s, inner_lr, second_order=True)

      learned_params = model.get_params()
      model = Model(epi.n_classes).init_with(learned_params).cuda()
      inner_loss_q, acc = model(epi.q)
      outer_loss += inner_loss_q

    sampler.sgd_step(outer_loss, outer_lr)

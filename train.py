from loader import MetaDataset, PseudoMetaDataset
from networks.sampler import EncoderInstance, EncoderClass, Sampler
from networks.model import Model

if __name__ == '__main__':

  metadata = MetaDataset(split='train')
  # metadata = PseudoMetaDataset()

  # TODO: integrate all these into single modules
  enc_ins = EncoderInstance().cuda()
  enc_cls = EncoderClass().cuda()
  sampler = Sampler().cuda()
  enc_ins.train()
  enc_cls.train()
  sampler.train()

  init_params = Model.cuda().get_init()
  outer_steps = 1000
  inner_steps = 100

  for i in range(outer_steps):
    outer_loss = 0


    for j, epi in enumerate(metadata.loader(n_batches=1)):
      epi.s.imgs = epi.s.imgs.cuda()
      epi.s.labels = epi.s.labels.cuda()
      view_classwise = epi.s.get_view_classwise_fn()
      view_elementwise = epi.s.get_view_elementwise_fn()

      xs = enc_ins(epi.s.imgs)
      xs = view_classwise(xs)
      xs = enc_cls(xs)

      sampler.initialze(epi.n_classes)
      model = Model(epi.n_classes).cuda().init_with(init_params)
      model.train()

      for j in range(inner_steps):
        mask = sampler(xs)  # class mask
        # xs = view_classwise(epi.s.imgs)  # entire support set
        # xs = xs * m  # masking operation
        # xs = view_elementwise(xs)
        inner_loss, acc = model(epi.s, mask, detach_params=False)
        grad = model.compute_grad(inner_loss, init_params)
        import pdb; pdb.set_trace()

import argparse
from nn2.model import Model
from nn2.sampler2 import Sampler

parser = argparse.ArgumentParser(description='Learning to sample evaluation.')
parser.add_argument('--load_dir', type=str, help='path to saved sampler.')

if __name__ == '__main__':
  print('Start_of_program')

  # loader configuration
  if cfg.loader.class_balanced:
    # class balanced sampling
    loader_cfg = LoaderConfig(
        class_size=cfg.loader.class_size,
        sample_size=cfg.loader.sample_size,
        num_workers=2,
    ).to(device, non_blocking=True)
  else:
    # typical uniform sampling
    #   (at least 2 samples per class)
    loader_cfg = LoaderConfig(
        batch_size=cfg.loader.batch_size,
        num_workers=2,
    ).to(device, non_blocking=True)

  # agent
  # utils.ForkablePdb().set_trace()
  sampler = shared_sampler.new().to(device)
  # model (belongs to enviroment)
  model = Model(
      input_dim=cfg.loader.input_dim,
      embed_dim=cfg.model.embed_dim,
      channels=cfg.model.channels,
      kernels=cfg.model.kernels,
      distance_type=cfg.model.distance_type,
      last_layer=cfg.model.last_layer,
      optim_getter=OptimGetter(cfg.model.optim, lr=cfg.model.lr),
      n_classes=len(metadata),
  ).to(device, non_blocking=True)
  # environment
  env = Environment(
      model=model,
      metadata=metadata,
      loader_cfg=loader_cfg,
      data_split_method=cfg.loader.split_method,
      mask_unit=cfg.sampler.mask_unit,
  ).to(device, non_blocking=True)

  # loop manager
  m = LoopMananger(train, rank, ready, ready_step, done, cfg.steps)
  state = env.reset()

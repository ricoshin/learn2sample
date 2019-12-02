from loader.loader import LoaderConfig
from nn2.environment import Environment
from nn2.model import Model
from nn2.sampler2 import Sampler
from utils import utils
from utils.helpers import LoopMananger, OptimGetter
from utils.result import ResultDict, ResultFrame, ResultList


def test(cfg, metadata, sampler):
  utils.random_seed(cfg.args.seed)
  device = 'cuda:0'
  sampler = sampler.load(cfg.dirs.save.params).to(device)
  model, env = initialize(cfg, metadata, device)
  state = env.reset()
  result_dict = ResultDict()
  result_frame = ResultFrame()

  # loop manager
  m = LoopMananger(
      train=False,
      outer_steps=cfg.steps.test.outer,
      inner_steps=cfg.steps.test.inner,
      log_steps=cfg.steps.test.log,
  )
  
  print(f'[Class] S: {len(env.meta_sup)} | Q: {len(env.meta_que)} | '
        f'class_balanced: {cfg.loader.class_balanced}')
  print(f'[Loader] split_method: {cfg.loader.split_method} | '
        f'batch_size: {cfg.loader.batch_size}\n')

  for outer_step, inner_step in m:
    action, value, embed_sampler = sampler(state, eps=0.0)
    # action = action_.sample()
    state, reward, info, embed_model = env(action, loop_manager=m)
    result_dict.append(
        outer_step=outer_step,
        inner_step=inner_step,
        ours_actual_loss=state.loss,
        ours_metric_loss=state.cls_loss,
        ours_metric_acc=state.cls_acc.float(),
        ours_mask_probs=action.probs[:, 1],
        ours_mask_instance=action.mask,
        base_actual_loss=info.base.loss,
        base_metric_loss=info.base.cls_loss,
        base_metric_acc=info.base.cls_acc.float(),
    )
    result_frame = result_frame.append_dict(
      result_dict.index_all(-1).mean_all(-1))

    if m.log_step():
      print(f'\n[Dir] {cfg.dirs.save.top}')
      print(f'[Step] outer: {outer_step:4d} | inner: {inner_step:4d}')
      # Note that loss is actual loss that was used for training
      #           acc is metric acc and does not care about fc layer.
      print(f'[Model] loss: {state.loss:6.3f} | '
            f'acc: {state.cls_acc.float().mean():6.3f} | '
            f'sparsity: {state.sparsity:6.3f}')
      print(f'[Base] loss: {info.base.loss:6.3f} | '
            f'acc: {info.base.cls_acc.float().mean():6.3f} | '
            f'sparsity: {info.base.sparsity:6.3f}')
      print(f'[RL] reward: {reward:6.3f} | value: {value:6.3f}\n')

    if m.end_of_episode():
      result_dict.save_csv('classwise/', cfg.dirs.save.test, outer_step)
      result_dict = ResultDict()
      state = env.reset()
      sampler.zero_states()

  result_frame.save_csv('overall', cfg.dirs.save.test)
  result_frame.save_final_lineplot('actual_loss', cfg.dirs.save.test)
  result_frame.save_final_lineplot('metric_loss', cfg.dirs.save.test)
  result_frame.save_final_lineplot('metric_acc', cfg.dirs.save.test)
  result_frame.save_final_lineplot('mask_probs', cfg.dirs.save.test)

  loss_actual = result_frame.get_best('actual_loss', 'min')
  loss_metric = result_frame.get_best('metric_loss', 'min')
  acc_metric = result_frame.get_best('metric_acc', 'max')

  print(f'\nFinal result:\n')
  loss_actual.save_mean_std('[Loss_actual]', cfg.dirs.save.test)
  loss_metric.save_mean_std('[Loss_metric]', cfg.dirs.save.test)
  acc_metric.save_mean_std('[Accuracy]', cfg.dirs.save.test)


def test_on_query(model):
  pass


def initialize(cfg, metadata, device):
  # loader configuration
  if cfg.loader.class_balanced:
    # class balanced sampling
    loader_cfg = LoaderConfig(
        class_size=cfg.loader.class_size,
        sample_size=cfg.loader.sample_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
    ).to(device, non_blocking=True)
    batch_size = cfg.loader.class_size * cfg.loader.sample_size
  else:
    # typical uniform sampling
    #   (at least 2 samples per class)
    loader_cfg = LoaderConfig(
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
    ).to(device, non_blocking=True)
    batch_size = cfg.loader.batch_size

  # model (belongs to enviroment)
  model = Model(
      input_dim=cfg.loader.input_dim,
      embed_dim=cfg.model.embed_dim,
      channels=cfg.model.channels,
      kernels=cfg.model.kernels,
      distance_type=cfg.model.distance_type,
      fc_pulling=cfg.model.fc_pulling,
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
      async_stream=cfg.control.async_stream,
      sync_baseline=False,
  ).to(device, non_blocking=True)

  return model, env

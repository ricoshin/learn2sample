import torch
from loader.episode import Episode
from nn.output import ModelOutput


class InnerStepScheduler():
  def __init__(self, outer_steps, inner_steps, anneal_outer_steps):
    assert 0 <= anneal_outer_steps <= outer_steps
    self.outer_steps = outer_steps
    self.inner_steps = inner_steps
    self.anneal_outer_steps = anneal_outer_steps

  def __call__(self, cur_outer_step, verbose=False):
    """linearly increase cur_inner_steps from 0 to 1.
      when cur_outer_step == anneal_outer_steps,
      cur_outer_step reaches to inner_steps.
    """
    if cur_outer_step < self.anneal_outer_steps:
      r = cur_outer_step / self.anneal_outer_steps
      cur_inner_steps = int(self.inner_steps * r)
    else:
      cur_inner_steps = self.inner_steps
    if verbose and not self.anneal_outer_steps == 0:
      print('Inner step scheduled : '
            f'{cur_inner_steps}/{self.inner_steps} ({r*100:5.2f}%)')
    return cur_inner_steps


class Printer():
  """Collection of printing functions."""
  @staticmethod
  def step_info(epoch, mode, out_step_cur, out_step_max, in_step_cur,
                in_step_max, lr):
    lr = lr.tolist()[0] if isinstance(lr, torch.Tensor) else lr
    return (f'[{mode}|epoch:{epoch:2d}]'
            f'[out:{out_step_cur:3d}/{out_step_max}|'
            f'in:{in_step_cur:4d}/{in_step_max}][{lr:4.3f}]')

  @staticmethod
  def way_shot_query(dataset):
    assert isinstance(episode, Episode)
    return (f'W/S/Q:{episode.n_classes:2d}/{episode.s.n_samples:2d}/'
            f'{episode.q.n_samples:2d}|')

  @staticmethod
  def colorized_mask(mask, min=0.0, max=1.0, multi=100, fmt='3d', vis_num=20,
                     colors=[160, 166, 172, 178, 184, 190]):
    out_str = []
    reset = "\033[0m"
    masks = mask.squeeze().tolist()
    if len(masks) > vis_num:
      id_offset = len(masks) / vis_num
    else:
      id_offset = 1
      vis_num = len(masks)
    for i in range(vis_num):
      m = masks[int(i * id_offset)] * multi
      color_offset = (max - min) * multi / len(colors)
      color = colors[int(m // color_offset) if int(m // color_offset)
                     < len(colors) else -1]
      out_str.append(f"\033[38;5;{str(color)}m" +
                     "%%%s" % fmt % int(m) + reset)
      # import pdb; pdb.set_trace()
    return f'[{"|".join(out_str)}]'

  @staticmethod
  def outputs(outputs, print_conf):
    assert isinstance(outputs, (list, tuple))
    assert(all([isinstance(out, ModelOutput) for out in outputs]))
    return "".join([out.to_text(print_conf) for out in outputs])

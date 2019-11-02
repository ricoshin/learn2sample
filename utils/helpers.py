import torch
from loader.episode import Episode
from loader.loader import MetaEpisodeIterator
from loader.metadata import Metadata
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


class Logger():
  """Collection of printing functions."""

  def __init__(self):
    self._log = ""

  def flush(self):
    print(self._log)
    self._log = ""

  def step_info(self, epoch, mode, out_step_cur, out_step_max, in_step_cur,
                in_step_max, lr):
    lr = lr.tolist()[0] if isinstance(lr, torch.Tensor) else lr
    self._log += (f'[{mode}|epoch:{epoch:2d}]'
                  f'[out:{out_step_cur:3d}/{out_step_max}|'
                  f'in:{in_step_cur:4d}/{in_step_max}][{lr:5.4f}]')

  def split_info(self, meta_support, meta_query, meta_episode_iterator):
    assert isinstance(meta_support, Metadata)
    assert isinstance(meta_query, Metadata)
    assert isinstance(meta_episode_iterator, MetaEpisodeIterator)
    # entire dataset
    s = meta_support
    q = meta_query
    # sampled inner dataset
    epi_s = meta_episode_iterator.support
    epi_q = meta_episode_iterator.query
    n_samples = meta_episode_iterator.samples_per_class
    self._log += (
        f'S({len(s)}):{len(epi_s)}w-{n_samples}s|'
        f'Q({len(q)}):{len(epi_q)}w-{n_samples}s|'
    )

  def colorized_mask(self, mask, min=0.0, max=1.0, multi=100, fmt='3d',
                     vis_num=20, colors=[160, 166, 172, 178, 184, 190],
                     cond=True):
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
    self._log += f'[{"|".join(out_str)}]'

  def outputs(self, outputs, print_conf):
    assert isinstance(outputs, (list, tuple))
    ###########################################################
    from nn.reinforcement import Policy
    assert(all([isinstance(out, (ModelOutput, Policy)) for out in outputs]))
    ###################################################################
    self._log += "".join([out.to_text(print_conf) for out in outputs])

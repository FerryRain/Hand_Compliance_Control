from mjlab.tasks.leaphand.leaphand_env_cfg import (
    leaphand_contact_env_cfg,
    LeapHandControlCfg,
)

from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Leaphand-Contact-Relocation",
  env_cfg=leaphand_contact_env_cfg(),
  play_env_cfg=leaphand_contact_env_cfg(play=True),
  rl_cfg=LeapHandControlCfg(amplitude=0.8), 
)
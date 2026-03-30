from mjlab.tasks.leaphand.leaphand_finger_env_cfg import (
    leaphand_contact_env_cfg,
    LeapHandControlCfg,
)

from mjlab.tasks.leaphand.leaphand_palm_env_cfg import (
    leaphand_palm_contact_env_cfg,
    LeapHandPalmControlCfg,
)

from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Leaphand-Finger-Compliance-Control",
  env_cfg=leaphand_contact_env_cfg(),
  play_env_cfg=leaphand_contact_env_cfg(play=True),
  rl_cfg=LeapHandControlCfg(amplitude=0.8), 
)

register_mjlab_task(
  task_id="Leaphand-Palm-Compliance-Control",
  env_cfg=leaphand_palm_contact_env_cfg(),
  play_env_cfg=leaphand_palm_contact_env_cfg(play=True),
  rl_cfg=LeapHandPalmControlCfg(amplitude=0.8),
)
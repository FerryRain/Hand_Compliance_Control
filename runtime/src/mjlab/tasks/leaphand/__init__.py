from mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg import (
    full_hand_mcc_env_cfg,
    FullHandMCCControlCfg,
)

from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
  task_id="Leaphand-Full-Hand-MCC-Control",
  env_cfg=full_hand_mcc_env_cfg(),
  play_env_cfg=full_hand_mcc_env_cfg(play=True),
  rl_cfg=FullHandMCCControlCfg(amplitude=0.5),
)

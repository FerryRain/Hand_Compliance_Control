from mjlab.tasks.leaphand.leaphand_finger_env_cfg import (
    leaphand_contact_env_cfg,
    LeapHandControlCfg,
)

from mjlab.tasks.leaphand.leaphand_finger_adhesion_env_cfg import (
    leaphand_adhesion_env_cfg,
    LeapHandAdhesionControlCfg,
)

from mjlab.tasks.leaphand.leaphand_palm_env_cfg import (
    leaphand_palm_contact_env_cfg,
    LeapHandPalmControlCfg,
)

from mjlab.tasks.leaphand.leaphand_palm_mcc_env_cfg import (
    mcc_palm_contact_env_cfg,
    MCCPalmControlCfg,
)

from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import (
    mcc_finger_contact_env_cfg,
    MCCLeapHandPositionControlCfg,
)

from mjlab.tasks.leaphand.leaphand_full_hand_mcc_env_cfg import (
    full_hand_mcc_env_cfg,
    FullHandMCCControlCfg,
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

register_mjlab_task(
  task_id="Leaphand-Palm-MCC-Compliance-Control",
  env_cfg=mcc_palm_contact_env_cfg(),
  play_env_cfg=mcc_palm_contact_env_cfg(play=True),
  rl_cfg=MCCPalmControlCfg(amplitude=0.8),
)

register_mjlab_task(
  task_id="Leaphand-Finger-Adhesion-Control",
  env_cfg=leaphand_adhesion_env_cfg(),
  play_env_cfg=leaphand_adhesion_env_cfg(play=True),
  rl_cfg=LeapHandAdhesionControlCfg(amplitude=0.8),
)

register_mjlab_task(
  task_id="Leaphand-Finger-MCC-Position-Control",
  env_cfg=mcc_finger_contact_env_cfg(),
  play_env_cfg=mcc_finger_contact_env_cfg(play=True),
  rl_cfg=MCCLeapHandPositionControlCfg(amplitude=0.5),
)

register_mjlab_task(
  task_id="Leaphand-Full-Hand-MCC-Control",
  env_cfg=full_hand_mcc_env_cfg(),
  play_env_cfg=full_hand_mcc_env_cfg(play=True),
  rl_cfg=FullHandMCCControlCfg(amplitude=0.5),
)

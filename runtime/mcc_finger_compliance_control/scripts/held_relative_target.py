"""Hold an MCC absolute joint target throughout one decimated control tick.

The external action remains a relative delta from the measured tick-start q.
Only its conversion to an actuator target is cached, so later physics steps
do not repeatedly add the same delta to the newly moving joint state.
"""


def process_held_target(term, actions, original_process):
    original_process(term, actions)
    position = term._entity.data.joint_pos[:, term._target_ids]
    bias = term._entity.data.encoder_bias[:, term._target_ids]
    target = position + bias + term._processed_actions
    if getattr(term, "_held_tick_target", None) is None:
        term._held_tick_target = target.clone()
    else:
        term._held_tick_target.copy_(target)


def apply_held_target(term, original_apply):
    target = getattr(term, "_held_tick_target", None)
    if target is None:
        return original_apply(term)
    term._entity.set_joint_position_target(target, joint_ids=term._target_ids)


def install_held_relative_targets():
    from mjlab.envs.mdp.actions.actions import JointRelativePositionAction
    cls = JointRelativePositionAction
    old_process, old_apply, old_reset = cls.process_actions, cls.apply_actions, cls.reset

    def process(term, actions):
        process_held_target(term, actions, old_process)

    def apply(term):
        return apply_held_target(term, old_apply)

    def reset(term, env_ids=None):
        old_reset(term, env_ids)
        target = getattr(term, "_held_tick_target", None)
        if target is not None:
            position = term._entity.data.joint_pos[:, term._target_ids]
            bias = term._entity.data.encoder_bias[:, term._target_ids]
            if env_ids is None:
                target.copy_(position+bias)
            else:
                target[env_ids] = (position+bias)[env_ids]

    cls.process_actions, cls.apply_actions, cls.reset = process, apply, reset

    def restore():
        cls.process_actions, cls.apply_actions, cls.reset = old_process, old_apply, old_reset

    return restore

"""Affine normalization is a numerical coordinate system, not a safety limit."""

MODES = ('legacy_unit', 'none')


def resolve_action_clip(requested=None, checkpoint_config=None):
    mode = requested
    if mode is None:
        mode = (checkpoint_config or {}).get('action_sample_clip') or 'legacy_unit'
    if mode not in MODES:
        raise ValueError(f'Unknown DP action sampling clip mode: {mode}')
    return mode


def clips_unit_range(args):
    return resolve_action_clip(getattr(args, 'action_sample_clip', None)) == 'legacy_unit'


def enforce_target_range(max_abs, mode):
    import math
    if not math.isfinite(max_abs):
        raise ValueError('Nonfinite normalized action labels')
    if resolve_action_clip(mode) == 'legacy_unit' and max_abs > 1.00001:
        raise ValueError('Teacher targets exceed legacy sampler range; use an explicit unbounded action contract instead of clipping labels')

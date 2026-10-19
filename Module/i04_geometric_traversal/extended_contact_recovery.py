"""Explicit development configuration for one anchored pivot up to 12 degrees.

The original small-pivot configuration and planner remain unchanged. Only the
candidate angle range and pad-center displacement allowance are extended.
Actual ellipsoid support, inward motion, secondary anchors, collision checks,
force guard and path speeds keep their original limits. A planned pivot is
geometry, not evidence of physical contact or recovery.
"""
from dataclasses import dataclass

import numpy as np

from .contact_recovery_planner import ContactRecoveryConfig


@dataclass(frozen=True)
class ExtendedContactRecoveryConfig(ContactRecoveryConfig):
    angles_deg: tuple = (3., 6., 12.)
    anchor_center_drift_m: float = .003

    def __post_init__(self):
        if (not self.angles_deg or any(not np.isfinite(angle) or not 0. < angle <= 12.
                                       for angle in self.angles_deg)):
            raise ValueError('Extended recovery angles must be finite, positive and at most 12 degrees')
        for name in self.__dataclass_fields__:
            if name != 'angles_deg' and (not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.):
                raise ValueError(f'{name} must be finite and positive')
        original = ContactRecoveryConfig()
        # The new center allowance accommodates rotation of an ellipsoid about
        # its unchanged surface support; it does not relax support penetration.
        if self.anchor_center_drift_m > .003:
            raise ValueError('Extended anchor center displacement must not exceed 3 mm')
        unchanged_upper_limits = (
            'angular_waypoint_deg', 'anchor_support_drift_m',
            'secondary_anchor_drift_m', 'maximum_anchor_inward_m',
            'wrist_speed_m_s', 'rotation_speed_rad_s', 'joint_speed_rad_s',
            'force_pause_n', 'position_tolerance_m', 'rotation_tolerance_rad',
        )
        for name in unchanged_upper_limits:
            if getattr(self, name) > getattr(original, name):
                raise ValueError(f'{name} must retain or tighten the original small-pivot limit')
        if self.minimum_gap_improvement_m < original.minimum_gap_improvement_m:
            raise ValueError('Minimum useful gap improvement must retain the original threshold')

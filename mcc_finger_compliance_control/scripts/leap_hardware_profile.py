"""Explicit LEAP v1 Full/5V simulation candidates, not hardware identification.

The manufacturer estimates 0.186Nm continuous output and lists 0.93Nm
momentary stall torque. The peak candidate uses a conservative 0.8Nm cap;
it is NOT a thermal model and is NOT certified for sustained operation.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LeapActuatorProfile:
    kp_nm_per_rad: float
    kd_nm_s_per_rad: float
    effort_limit_nm: float
    interpretation: str


PROFILES = {
    "legacy": LeapActuatorProfile(35.0, 2.5, 35.0, "Previous uncalibrated collection baseline"),
    "continuous": LeapActuatorProfile(3.0, 0.1, 0.186,
        "Conservative fixed cap at manufacturer-estimated continuous torque"),
    "peak": LeapActuatorProfile(3.0, 0.1, 0.8,
        "Peak-capability diagnostic; permits sustained >0.186Nm and is not continuous-rated"),
}


def profile_metadata(name):
    return dict(profile=name, **asdict(PROFILES[name]),
                hardware_assumption="LEAP Hand v1 Full, XC330-M288-T, 5V, direct joint gearing",
                continuous_estimate_nm=0.186, stall_nm_at_5v=0.93,
                sources=["https://v1.leaphand.com/parts",
                         "https://robotis.us/products/dynamixel-xc330-m288-t",
                         "https://github.com/leap-hand/LEAP_Hand_Sim/issues/9"],
                limitations="PD gains are a simulation starting point, not identified real servo dynamics; no thermal or torque-speed model")

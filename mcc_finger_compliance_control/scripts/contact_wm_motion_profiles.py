"""Seeded object-motion diversity; no measured-state recovery logic."""
import numpy as np

NORMAL_BEHAVIORS={'slide','rotate','ellipse','slide_release','normal_cycle'}


def assign_motion_diversity(jobs,enabled=False):
    for job in jobs:
        if enabled:
            rng=np.random.default_rng(job['seed']+81031)
            angle=float(rng.uniform(-np.pi,np.pi))
            scale=float(rng.uniform(.7,1.3))
            release=float(rng.uniform(.008,.012))
            press=float(rng.uniform(.0002,.0005))
        else:angle,scale,release,press=0.,1.,.009,.0003
        job.update(motion_direction_rad=angle,motion_amplitude_scale=scale,
            release_distance_m=release,press_distance_m=press,
            motion_randomized=bool(enabled))


def smoothstep5(u):
    u=np.clip(u,0.,1.)
    return u**3*(10-15*u+6*u*u)


def plateau_pulse(phase,start,rise_end,fall_start,end):
    return smoothstep5((phase-start)/(rise_end-start))*(1-smoothstep5((phase-fall_start)/(end-fall_start)))


def normal_excursion_m(job,physical_frame,base_period_s):
    """Positive means object retreats along the initial downward approach axis.

    The cycle is preissued; loss of contact never changes timing/direction.
    This modifies the object path AFTER its geometric height compensation,
    otherwise a surface follower would cancel the intended release.
    """
    if job['behavior'] not in ('slide_release','normal_cycle'):return 0.
    phase=(physical_frame*.01*job['motion_rate']/base_period_s)%1.
    return (job['release_distance_m']*plateau_pulse(phase,.25,.35,.5,.65)
            -job['press_distance_m']*plateau_pulse(phase,.05,.10,.15,.20))

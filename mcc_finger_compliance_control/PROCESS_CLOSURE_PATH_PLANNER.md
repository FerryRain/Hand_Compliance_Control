# Closure-path fingertip teacher collection process

## Goal

Keep the existing palm planner unchanged. For every current palm pose, each
finger follows a fixed joint-space path from `open_grasp_q` to the object's
fixed grasp posture. The first open-to-grasp sample that crosses the privileged
mesh surface becomes that finger's MCC target. The target is moved slightly
inside the surface and its pad frame is aligned with the oracle normal.

If the path stays outside the mesh, the finger uses a conservative partial
closure (`0.70` by default, configurable to `0.60` or `0.80`). If the open pose
already has a small mesh penetration, it is not treated as an intersection at
fraction zero; the path is evaluated to the end and the fixed grasp posture is
used when no clean crossing occurs.

## Code changes

- `scripts/surface_mcc_finger.py`
  - Added `DEFAULT_OPEN_Q` and `open_grasp_q`.
  - Added per-finger pad-normal differential correction.
  - Orientation correction cannot reverse the open-to-grasp joint direction.
- `scripts/collect_trajectories.py`
  - Added `_closure_surface_targets()`.
  - Added `--closure-path-fallback-fraction` and `--closure-path-samples`.
  - FullHandMCC uses the closure-path posture as `nominal_posture_q`.
  - Palm trajectory generation remains unchanged.

## Planner files generated

```text
data/plans/current_eval/mustard_latlon45_len0.h5
data/plans/current_eval/mustard_latlon90_len0.h5
data/plans/current_eval/mustard_latlon45_len50.h5
data/plans/current_eval/mustard_latlon45_clear30.h5
data/plans/current_eval/mustard_latlon90_clear30.h5
```

The original 60 mm palm-outline clearance leaves the ring-finger closure path
outside the mustard mesh (about 13.4 mm at full closure). The 30 mm clearance
version has geometric intersections for all four fingers at the initial pose.

## Verified commands

The machine currently has no working NVIDIA driver, so these tests used
`--device cpu`; on a CUDA machine replace it with `--device cuda:0`.

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/generate_manifold_palm_plan.py \
  --object-id ycb_mustard \
  --output mcc_finger_compliance_control/data/plans/current_eval/mustard_latlon45_clear30.h5 \
  --frames 500 --angle-deg 45 --path-length-m 0 \
  --path-mode mesh_latlon --direction 1 \
  --palm-outline-clearance-m 0.030 --smoothing-sigma-frames 8
```

For collection, the total length must include preparation:

```text
trajectory_length >= motion_start + max_prep_wait_steps + motion_length
```

Example:

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/collect_trajectories.py \
  --viewer headless --device cuda:0 --num-envs 1 \
  --trajectory-length 700 --max-trajectories 1 --max-attempts 1 \
  --motion-start 100 --record-start-step 100 --motion-length 350 \
  --max-prep-wait-steps 250 --planner-settle-steps 30 \
  --object-id ycb_mustard --teacher-controller fullhand_mcc \
  --motion-mode planner_inverse \
  --planner-file mcc_finger_compliance_control/data/plans/current_eval/mustard_latlon45_clear30.h5 \
  --initial-orientation-mode fixed --no-contact-gate \
  --closure-path-fallback-fraction 0.70 --closure-path-samples 25 \
  --filename mustard_latlon45_clear30_eval
```

## Current evidence

The first 60 mm-clearance path produced:

```text
index 87.9%, middle 94.7%, ring 28.9%, thumb 75.8%
all four 18.9%, >=3 fingers 72.1%
```

With 30 mm clearance and differential QP disabled, before the latest
initial-penetration fix:

```text
index 99.2%, middle 100.0%, ring 95.3%, thumb 59.4%
all four 54.7%, >=3 fingers 99.2%
```

With 30 mm clearance and QP enabled (partial recording due prep timing):

```text
index 100.0%, middle 99.2%, ring 88.3%, thumb 68.0%
all four 56.2%, >=3 fingers 99.2%
```

The earlier partial runs do **not** represent the final controller. After the
final direction-mask fix (normal alignment is prevented from reversing the
open-to-grasp closure direction), two fresh full-length runs were completed:

| palm plan | recorded frames | index | middle | ring | thumb | all four | at least three |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mustard_latlon45_clear30` | 570 | 100.0% | 100.0% | 97.2% | 100.0% | 97.2% | 100.0% |
| `mustard_latlon90_clear30` | 570 | 99.6% | 99.6% | 84.4% | 96.1% | 80.2% | 99.6% |

The 45-degree mesh-projected path passes the current 95% all-four criterion.
The original 90-degree result exposed a ring-finger limitation and motivated
the path-authority and ellipse changes below.

For the same mustard-object family, the stored fixed-pregrasp baseline
`data/trajectories/ycb_mustard_fixedpregrasp_20260818.h5` has 46.5% all-four
contact (94.6%, 96.6%, 67.6%, 78.1% per finger). Therefore the new 45-degree
closure-path controller exceeds that baseline by 50.7 percentage points and
passes the 95% target. The 90-degree run also exceeds the baseline, but does
not pass the stricter 95% all-four target because of the ring finger.

## 2026-08-24: outward recovery and analytic ellipse

Visual inspection showed that the bottle rolled over the ring finger while
the old persistent-recovery state continued pulling it toward the previous
material contact and the fully closed grasp. The corrected controller now:

- treats the current open-to-grasp intersection as authoritative every frame;
- selects the open endpoint if that endpoint is already inside the object;
- holds the 70% FK endpoint without preload when no intersection exists;
- uses the current closure-path posture, rather than the full grasp, as the
  persistent-recovery null-space reference;
- never lets a stale material anchor override a newly computed intersection.

The ring distal joint then opened from a previous minimum of 0.715 rad to
0.389 rad, proving that outward recovery became active. On the old projected
90-degree palm path this alone reduced all-four contact from 80.2% to 70.2%,
showing that the remaining problem was palm-path geometry rather than failure
of the opening command.

`generate_manifold_palm_plan.py` therefore adds `ellipse_clearance`. It fits a
smooth analytic ellipse to the mesh cross-section, uses quintic endpoint time
scaling and uniform arc-length resampling, and solves one temporally filtered
normal offset so the **mean distance of the complete palm outline** equals a
requested threshold. Example:

```bash
python mcc_finger_compliance_control/scripts/generate_manifold_palm_plan.py \
  --object-id ycb_mustard \
  --output mcc_finger_compliance_control/data/plans/current_eval/mustard_ellipse90_mean30.h5 \
  --frames 500 --angle-deg 90 --path-length-m 0 \
  --path-mode ellipse_clearance --direction 1 \
  --palm-mean-clearance-m 0.030 --smoothing-sigma-frames 8
```

The generated path has 29.90--30.07 mm mean outline clearance and almost
constant translational step length (0.4880 mm mean, 0.0007 mm standard
deviation). With the original CoACD-64 collision asset it achieved 93.9%
all-four contact; all ring-finger frames remained in contact.

## V-HACD-256 collision A/B

Khadivar et al., *Robotics and Autonomous Systems* 166 (2023) 104461, state
that their concave MuJoCo objects were divided into 256 convex pieces with
V-HACD. An initial A/B using raw-scale OBJ decomposition produced:

| collision | index | middle | ring | thumb | all four | at least three |
|---|---:|---:|---:|---:|---:|---:|
| CoACD-64 | 99.8% | 94.4% | 100.0% | 99.6% | 93.9% | 100.0% |
| V-HACD-256 | 100.0% | 99.8% | 100.0% | 96.3% | **96.1%** | 100.0% |

The 256-part run took about 143 s, comparable to the previous approximately
146 s CPU run. Its manifest p95 of 0.584 mm was measured before the old 2.8x
runtime scale, so the physical p95 was approximately 1.64 mm.

The final asset follows the requested preprocessing order exactly:

```text
original nontextured.ply
  -> subtract raw bounds centre
  -> bake uniform 2.8x physical scale
  -> serialize and reload visual_scaled.obj
  -> V-HACD into exactly 256 collision_part_*.obj files
  -> load visual + parts in MuJoCo with runtime scale=1 and geom-local pos=0
```

```bash
python mcc_finger_compliance_control/scripts/decompose_collision_mesh.py \
  assets_external/ycb/models/006_mustard_bottle/google_16k/nontextured.ply \
  --output-dir assets_external/ycb/collision/006_mustard_bottle/vhacd_256_scaled2p8_objstage \
  --backend vhacd --max-hulls 256 --max-vertices 64 \
  --pre-center -0.015339 -0.023499 0.092498 --pre-scale 2.8 \
  --vhacd-resolution 1000000 \
  --vhacd-volume-error-percent 0.1 --vhacd-recursion-depth 16
```

At final physical scale its visual-to-collision p95 is 1.688 mm. The matching
90-degree ellipse plan achieved 100.0%, 99.1%, 100.0%, and 97.2% per-finger
contact, **96.3% all-four**, and 100.0% at-least-three contact. The mustard
YAML selects this baked-scale asset with `size_scale_range: [1, 1]` and
`geoms[].pos: [0, 0, 0]`; its world `body.initial_pos` remains unchanged. The
older collision directories remain available for reproducible A/B tests.

## Hybrid stable-grasp surface compliance

Contact count alone hid an unacceptable IK branch: the old six-dimensional
tip task could satisfy pad orientation by folding one flexion joint underneath
the object while opening the other two. It also recomputed a complete 3-D
target during healthy contact, so a stationary object could still produce
tangential sliding and contact-point hopping.

The collection teacher now separates responsibilities:

- the open-to-grasp closure path supplies a natural nominal hand shape;
- each pad is turned toward the source-mesh normal using only that finger's
  side/opposition joint;
- healthy contact uses the measured pad-point Jacobian only along the surface
  normal, while the two tangent directions remain owned by the stable grasp;
- the normal correction is parameterized by one closure coordinate, so the
  three flexion joints move by the same normalized open-to-grasp fraction;
- while the object moves, the nominal grasp approaches the new planned shape
  slowly; while it is stationary, the healthy reference is frozen and only
  normal pressure regulation remains active;
- only persistent physical contact loss invokes full 3-D surface recovery.

For mesh objects, force directions are queried from the original undecomposed
visual mesh at the measured 3-D contact point. V-HACD parts remain collision
geometry only and their seam normals are not control inputs.

On the first 312 recorded frames of the 90-degree mustard path, removing the
erroneous "shape error implies full 3-D recovery" transition improved all-four
contact from 20.2% to 89.7%, with at-least-three contact at 100%. Relative to
the older high-contact but folded controller, ring distal motion range fell
from 1.318 rad to 0.264 rad. The remaining visual check is therefore posture
quality and stationary hold, not merely the contact bit.

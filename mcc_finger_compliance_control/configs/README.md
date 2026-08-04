# Contact-object configuration

This directory is the source of truth for objects used by the MCC fingertip
data pipeline. Object configurations are resolved in three layers:

```text
objects/base.yaml
  <- families/<family>.yaml
  <- objects/<object_id>.yaml
```

`base.yaml` contains contact and recording defaults. A family controls broad
motion and quality limits. An object file contains its MuJoCo geoms and the
translation/rotation ranges that are safe for that particular shape.

The initial catalog primarily uses MuJoCo primitives. They provide stable
normals and inexpensive collision detection, making them suitable for testing
cross-shape generalization before importing irregular meshes. A compound
object is represented by several geoms on one mocap body; it does not need a
separate OBJ file. `rounded_box` is the exception: the catalog generates one
watertight convex mesh in memory so rounded edges do not create overlapping
primitive contacts.

Validate all configurations without a window:

```bash
python mcc_finger_compliance_control/scripts/view_object_gallery.py \
  --viewer headless
```

View the complete catalog:

```bash
python mcc_finger_compliance_control/scripts/view_object_gallery.py \
  --viewer native
```

View selected objects:

```bash
python mcc_finger_compliance_control/scripts/view_object_gallery.py \
  --object capsule_medium \
  --object cross_capsule \
  --viewer native
```

Generate a static preview in a headless EGL session:

```bash
MUJOCO_GL=egl \
python mcc_finger_compliance_control/scripts/view_object_gallery.py \
  --viewer image \
  --output mcc_finger_compliance_control/outputs/object_gallery.png
```

Visualize the exact collection environment (same object builder, controller,
motion generator, contact material, and sensors as headless collection):

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/collect_trajectories.py \
  --viewer native \
  --device cuda:0 \
  --num-envs 1 \
  --object-id rounded_box_medium \
  --trajectory-length 2500 \
  --motion-start 350 \
  --motion-length 1800 \
  --initial-orientation-mode uniform
```

The live markers use green for a collision carrying at least the configured
force threshold, orange for a weak geometric contact, and red for a lost
fingertip. Use `--viewer viser` if a native GLX window is unavailable.
Viewer mode runs one finite trajectory and does not write an H5 file.

Run a small rotation-only baseline over the complete catalog. Stratified axis
sampling cycles through each object's configured principal/random axes across
the four parallel environments:

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/batch_collect.py \
  --device cuda:0 \
  --num-envs 4 \
  --max-trajectories 4 \
  --trajectory-length 2500 \
  --axis-sampling stratified \
  --seed 20260806
```

Analyze the resulting raw files with family-specific three-tip, four-tip,
continuous-loss, and force limits:

```bash
python mcc_finger_compliance_control/scripts/analyze_quality.py \
  mcc_finger_compliance_control/data/trajectories/*_TIMESTAMP.h5 \
  --report mcc_finger_compliance_control/data/trajectories/baseline_quality.csv
```

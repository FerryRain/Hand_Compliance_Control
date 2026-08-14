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

## Irregular visual meshes and collision meshes

Do not attach one non-convex OBJ/STL as a normal MuJoCo ``mesh`` geom: its
collision shape is one convex hull.  Preprocess irregular YCB-style meshes
into several convex parts while retaining the original mesh as a
non-colliding visual geom.  The repository provides a deterministic V-HACD
pipeline with an optional CoACD backend:

```bash
python mcc_finger_compliance_control/scripts/decompose_collision_mesh.py \
  assets_external/ycb/models/024_bowl/google_16k/textured.obj \
  --output-dir assets_external/ycb_collision/024_bowl \
  --backend vhacd \
  --max-hulls 64 \
  --max-vertices 64 \
  --accept-p95-mm 1.0 \
  --accept-max-parts 64
```

The output contains ``collision_part_*.obj``, ``manifest.json`` with sampled
surface-fit metrics, and ``preview.xml``.  The preview renders the original
mesh translucently and overlays the collision hulls.  Open it with MuJoCo's
viewer before accepting an asset.  In addition to visual inspection, require
the expected cavities/holes to remain contact-free under geometric probes.

V-HACD is available through the existing ``trimesh[easy]`` dependency and is
the reliable default for non-watertight scans.  For optional CoACD trials:

```bash
python -m pip install coacd
python mcc_finger_compliance_control/scripts/decompose_collision_mesh.py \
  INPUT.obj --output-dir OUTPUT --backend coacd \
  --coacd-threshold-m 0.001
```

CoACD ``--coacd-merge`` is deliberately opt-in: aggressive merging in CoACD
1.0.11 is not robust for every repaired open scan.  Fine error thresholds can
also produce hundreds of hulls, which is unsuitable for large GPU batches.
Tune both fit error and hull count rather than accepting an asset by error
alone.  The acceptance flags still write the manifest and preview before
returning a nonzero exit status, so rejected objects remain diagnosable.
Generated third-party collision assets remain under
``assets_external/`` and are intentionally not committed.

The official YCB Google scans are available at 16k, 64k, and 512k polygon
tiers. Download selected high-resolution reference meshes without RGB/RGB-D
recordings as follows:

```bash
python mcc_finger_compliance_control/scripts/download_ycb_models.py \
  --output assets_external/ycb_highres \
  --object-ids 024_bowl 035_power_drill 051_large_clamp \
  --google-resolution 64k \
  --workers 3 \
  --no-berkeley-fallback
```

Use 64k as the normal source for collision preprocessing and 512k only as a
reference surface for validation. In representative YCB objects, 16k and 64k
were already within roughly 0.15--0.23 mm (p95/p99) of the 512k surface. A
multi-millimeter collision error therefore usually comes from bounded convex
decomposition, not the Google mesh resolution; increasing polygon count alone
does not fix it.

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
  --motion-start 1000 \
  --record-start-step 1000 \
  --max-prep-wait-steps 1000 \
  --motion-length 1400 \
  --motion-mode rotation \
  --initial-orientation-mode fixed
```

`motion-start` is the earliest allowed start, not a forced cutoff.  The object
remains stationary until the arm/palm approach is complete and FullHandMCC has
settled all four fingertips (up to `max-prep-wait-steps` extra simulator
steps).  When `record-start-step` equals `motion-start`, object motion and H5
recording then start together on that actual ready frame.  All preceding prep
frames are excluded, and every trajectory still contains the same number of
saved frames. `uniform` samples the full SO(3), but a fixed palm
cannot grasp every object orientation; use it only together with candidate
rejection/resampling. Use a verified `fixed` pose for baseline comparisons.

The live markers use green for a collision carrying at least the configured
force threshold, orange for a weak geometric contact, and red for a lost
fingertip. Use `--viewer viser` if a native GLX window is unavailable.
Viewer mode runs one finite trajectory and does not write an H5 file.

Run a small three-mode baseline over selected objects. `rotation` and
`translation` are deliberately available separately; `combined` is the harder
coupled excitation. Stratified axis sampling cycles through each object's
configured principal/random axes across the four parallel environments:

```bash
MPLCONFIGDIR=/tmp/matplotlib WARP_CACHE_PATH=/tmp/warp \
python mcc_finger_compliance_control/scripts/batch_collect.py \
  --objects capsule_medium ellipsoid_medium sphere_medium \
  --motion-modes rotation translation combined \
  --device cuda:0 \
  --num-envs 4 \
  --max-trajectories 4 \
  --trajectory-length 2500 \
  --motion-start 1000 \
  --record-start-step 1000 \
  --max-prep-wait-steps 1000 \
  --motion-length 1400 \
  --initial-orientation-mode fixed \
  --axis-sampling stratified \
  --seed 20260806
```

Analyze the resulting raw files. The default `strict99` profile requires at
least 99% simultaneous four-tip contact, at least 99% contact for every tip,
and no four-tip loss run longer than five frames:

```bash
python mcc_finger_compliance_control/scripts/analyze_quality.py \
  mcc_finger_compliance_control/data/trajectories/*_TIMESTAMP.h5 \
  --report mcc_finger_compliance_control/data/trajectories/baseline_quality.csv
```

Use `--quality-profile family` only for controller diagnostics. Its looser
three-tip/family thresholds must not be used to select the final DP teacher
dataset.

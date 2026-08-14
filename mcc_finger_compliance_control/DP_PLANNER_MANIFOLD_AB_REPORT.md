# DP Planner / Surface-Manifold A/B Report

## 1. Scope and controlled setup

This experiment deliberately uses the current single-object capsule dataset. The
goal is to establish a working end-to-end demo before collecting multiple-object
generalization data.

- Source: `so3_uniform_v2_255x2500_relaxed999_inverted.h5`
- Episodes: 255, each with 2150 recorded frames
- Four-fingertip contact coverage: approximately 99.999%
- DP sampling stride: 5 simulation frames (20 Hz)
- Observation horizon: 16 DP samples (0.8 s)
- Prediction horizon: 32 DP samples (1.6 s)
- Action: 16-dimensional absolute hand joint position
- Model/training settings are held fixed across each A/B pair.
- Offline A/B uses the same 26 validation episodes, 512 windows, and diffusion
  seeds 101, 202, and 303.
- Live A/B uses the same `fullhand_mcc` execution layer, oracle normal source,
  seed 20260717, 10 diffusion inference steps, chunk execution, and a 20-frame
  replanning interval.

## 2. Input variants

### A0: contact geometry only (50 dimensions per history sample)

```text
q                                  16
contact points in palm frame       12
contact normals in palm frame      12
contact masks                       4
palm linear/angular velocity        6
total                              50
```

### A1: contact geometry + local planner (56 dimensions)

The planner adds one nearby, causal 6D command. It is the palm pose change over
the next 20 raw frames (0.2 s), expressed in the current palm frame:

```text
local palm translation delta        3
local palm rotation-vector delta    3
total planner command               6
```

This is intentionally a nearby command rather than a long future trajectory, so
the learned policy can later accept commands from an external planner.

### A2: A1 + surface-manifold embedding (88 dimensions)

For each fingertip, the past 16 valid contact samples are fit by an independent,
causal local height GP. Eight equal-area query points are placed on a 6 mm disk.
Each query contains:

```text
query position                       3
predicted normal                     3
GP mean (mu)                         1
GP standard deviation (delta)        1
local support                        1
valid mask                           1
per-point total                     10
```

A shared PointNet encodes `[4 fingers, 8 points, 10 features]` into a 32D latent.
The PointNet uses only current and past measurements. Future contact movement and
normal are used only as its supervised pretraining targets.

## 3. A/B 1: does the local planner help?

### Paired offline diffusion evaluation

| Model | Horizon MAE | Horizon P95 | Final-step MAE | Final-step P95 |
|---|---:|---:|---:|---:|
| A0: no planner | 0.001809 rad | 0.007640 rad | 0.003040 rad | 0.014003 rad |
| A1: local planner | **0.001705 rad** | **0.007442 rad** | **0.002793 rad** | **0.012602 rad** |
| Relative change | **-5.77%** | **-2.58%** | **-8.11%** | **-10.00%** |

The local planner improves every paired offline metric. The strongest gain is at
the end of the prediction horizon, where future motion intent matters most.

### Strict live deployment comparison

Statistics below exclude the first 75 bootstrap frames.

| Episode / model | q MAE | q P95 | >=3 fingers | 4 fingers | Force P95 |
|---|---:|---:|---:|---:|---:|
| ep32 A0: no planner | 0.01221 | 0.02037 | 98.3% | 84.2% | 7.13 N |
| ep32 A1: planner | **0.00581** | **0.01099** | 97.0% | **88.5%** | 11.67 N |
| ep68 A0: no planner | **0.00255** | **0.00336** | **98.9%** | **89.7%** | 5.60 N |
| ep68 A1: planner | 0.00465 | 0.00675 | 98.0% | 89.1% | 5.66 N |

Interpretation:

- On difficult ep32, planner conditioning halves joint drift and raises
  four-finger contact by 4.3 percentage points.
- On already-easy ep68, the no-planner model remains slightly better. Planner
  conditioning is therefore useful directional information, not a universal
  low-level stabilization mechanism.
- The ep32 planner run has larger force peaks. This does not invalidate the
  q-only learning result, but force limiting remains necessary before a hardware
  demo.

Overall A/B 1 decision: **keep the local planner input**. Its paired validation
gain is consistent, and it helps the difficult live trajectory where intent is
most useful.

## 4. PointNet pretraining result

The surface encoder is not random or degenerate. On its validation set:

| Predictor | Future contact-point delta MAE | Future normal error |
|---|---:|---:|
| Simple baseline | 1.180 mm | 0.923 deg (hold current normal) |
| PointNet + GP | **0.206 mm** | 1.332 deg |

Ablating its point input confirms that the latent uses geometry:

| Point input | Delta MAE | Normal error |
|---|---:|---:|
| True GP points | 0.207 mm | 1.34 deg |
| Shuffled GP points | 1.054 mm | 20.07 deg |
| Zero GP points | 0.892 mm | 15.14 deg |

The encoder strongly predicts local contact displacement. Its normal head is not
yet better than simply retaining the current normal, so a later version should
predict a residual normal or omit that head.

## 5. A/B 2: does direct manifold-latent concatenation help DP?

### Paired offline diffusion evaluation

| Model | Horizon MAE | Horizon P95 | Final-step MAE | Final-step P95 |
|---|---:|---:|---:|---:|
| A1: planner only | **0.001705** | **0.007442** | **0.002793** | **0.012602** |
| A2: planner + 32D manifold | 0.001885 | 0.008363 | 0.003016 | 0.013480 |
| Relative change | +10.54% | +12.37% | +7.97% | +6.97% |

### ep32 deployment diagnosis

| Mode | q MAE | q P95 | >=3 fingers | 4 fingers | Force P95 |
|---|---:|---:|---:|---:|---:|
| A2 teacher history | 0.00492 | 0.00750 | 97.7% | **91.6%** | 4.20 N |
| A2 live history | 0.03101 | 0.12480 | 72.0% | 59.0% | 39.33 N |

Teacher-history execution works, which verifies the coordinate conversion,
checkpoint loading, online GP implementation, and DP output path. Live execution
fails after contact deviations feed a novel GP latent into DP; that changes q,
which changes contact geometry again and creates a positive-feedback loop.

Overall A/B 2 decision: **do not use the current raw-concatenation manifold model
for the demo**. The geometry representation contains useful information, but its
fusion and robustness training are not yet adequate for closed-loop use.

## 6. Recommended demo pipeline

Use A1 for the first viable single-object demo:

```text
nearby palm motion command
        +
joint/contact history
        -> local-planner-conditioned DP
        -> absolute q reference chunk
        -> FullHandMCC contact/recovery layer
        -> hand
```

The current recommended checkpoint is:

```text
mcc_finger_compliance_control/data/models/
so3_uniform_v2_palm_geometry_local_planner_absolute_q_dp_25k/best.pt
```

The GP/PointNet branch should remain experimental. Before its next A/B:

1. Inject only the latest surface latent, not 16 highly overlapping latents.
2. Reduce the latent to 8-16 dimensions and use a gated residual/FiLM fusion
   whose gate starts near zero, instead of direct state concatenation.
3. Add contact-point/normal noise, latent dropout, and short simulated contact
   loss during DP training.
4. Fine-tune with live/DAgger histories so the encoder sees recovery states.
5. Predict normal change relative to the current normal, or retain the current
   normal directly when it is already the stronger baseline.

Only after the planner-only demo is repeatable should the data collection scope
expand to multiple objects and irregular geometry.

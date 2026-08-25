# I01 Bunny physics evidence — 2026-08-24

## Decision

- Module: `I01-PHY-BUNNY-v1`
- Execution status: `EVALUATED`
- Performance: `MET`
- Gate G2: `GO`
- Scope: Geometry-Oracle + Explicit MCC baseline only; Finger DP/DPRef disabled.

This evidence answers the frozen question: in the specified MuJoCo Bunny scene, the hand can move
while retaining a measured nonempty Bunny contact set when an audited `4 -> 3 -> 4` contact-mode
handover is allowed. Strict four-contact execution cannot complete the same path.

## Frozen protocol and provenance

- Protocol: [`../I01_BUNNY_PROTOCOL.md`](../I01_BUNNY_PROTOCOL.md)
- Protocol SHA-256: `d79d163364199aff4abf72800d8646c8e7b5a4bb01596cfdb1b5f41516a21348`
- Git HEAD at formal run: `f1f92501d96910842533788e734eab285f4c8419`
- Worktree at formal run: dirty; generated summary records hashes for the protocol and 13 implementation
  sources, so this result must not be represented as a clean-commit artifact.
- Environment: Python `3.10.0`, MuJoCo `3.6.0`, NumPy `2.2.6`, SciPy `1.15.3`, trimesh `4.11.5`.
- Host: Intel Core i7-13700, Linux `5.15.0-107-generic-x86_64`.
- Formal benchmark wall time: `52.327 s` for six paired episodes.
- Source stability: `PASS`; all 14 recorded source hashes were identical at benchmark start and end.
- Final `summary.json` SHA-256:
  `e94de1cef3ba857f92418b5030a818eff70347b1c27c69417fe643117311088b`.

Input Bunny is [`../assets/stanford_bunny.ply`](../assets/stanford_bunny.ply), SHA-256
`7fb5395ff0bdfcab05a61e03748db28556cff2484d2fd6b3c81845a29b8886ef`. The canonical side-laid
mesh has 34,834 vertices, 69,451 faces and extents
`[0.300000, 0.297369, 0.232513] m`.

## Reproduction

Run from the repository root:

```bash
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python

$PY -m unittest Module.tests.test_i01_bunny_physics -v
$PY -m Module.i01_bunny_physics.benchmark
MUJOCO_GL=osmesa XDG_CACHE_HOME=/tmp/handcomp-i01-cache \
  $PY -m Module.i01_bunny_physics.visual_demo --reuse
```

The benchmark writes numerical results and full per-tick traces. The visual command reads those frozen
results and does not recompute acceptance metrics.

Post-run verification:

- targeted I01 + M06–M12 regression: 15 tests in `3.861 s`, `OK`;
- full `Module/tests` regression: 98 tests in `44.492 s`, `OK`;
- all three MP4 files decode as H.264, 12 fps, 12.000 s;
- generated HTML local links, six-row CSV, JSON hashes and 6000-tick seed-7 traces passed integrity
  checks; `git diff --check` passed.

## Formal results

Paired seeds are `7, 11, 19`; acquisition is 3 s and the scored traversal window is 9 s with a shared
60 mm target path.

| metric | fixed `|A|=4` | variable `4->3->4` |
| --- | ---: | ---: |
| actual progress values | `19.781 / 20.195 / 19.885 mm` | `58.430 / 58.423 / 58.429 mm` |
| actual progress median | `19.885 mm` | `58.429 mm` |
| mean bootstrap 95% CI | `[19.781, 20.195] mm` | `[58.423, 58.430] mm` |
| nonempty-contact fraction | `100% / 100% / 100%` | `99.956% / 99.933% / 99.956%` |
| maximum all-contact-loss gap | `0 ms` | `2 ms` |
| worst valid fingertip force | `4.736 N` | `5.475 N` |
| primary pass | `0/3` | `3/3` |
| measured handover mechanism pass | n.a. | `3/3` |
| M10 certificates / M06 barriers | `0 / 0` | `9 / 9` |
| authority violations | `0` | `0` |

All six episodes have zero over-force tick and zero non-tip collision tick. The maximum per-episode
controller P95 is `0.852 ms`; maximum physics-step P95 is `0.316 ms`. Variable-mode M10 swept-audit
P95 is `0.408–0.433 s`; these audits run during a planned plateau and are not claimed as a real-time
planner rate.

Fixed episodes stop with `FIXED_MODE_LOST` after the four-contact requirement is broken for more than
40 ms. They retain at least one valid Bunny contact, so the stop is specifically the frozen strict-mode
constraint rather than total contact loss. The historical `47.97 mm` planning failure is a different
scene/result and was not reproduced here; `19.885 mm` must not be called a physical reach limit.

The G2 margin is:

```text
median(L_variable) - median(L_fixed) = 38.544 mm >= 10 mm
```

Variable also passes primary and measured-handover requirements in 3/3 episodes. Prediction suffixes
never enter the command path. Therefore the frozen evaluator returns `G2=GO`.

## Geometry and interpretation limits

- Exact transformed triangle mesh is used for rendering and post-contact residual checks.
- MuJoCo collision uses a deterministic `181 x 181` vertical-ray upper-envelope hfield generated from
  the same mesh. A contact only counts when its exact-mesh residual is at most `2.5 mm`.
- Across variable episodes the exact-mesh contact-residual P95 is `0.094–0.099 mm`; approximately
  `0.12–0.21%` of raw collision candidates are rejected by the exact-mesh filter.
- Bunny is fixed and gravity is off to isolate contact control. This is not evidence for complete
  nonconvex triangle collision, gravity-on manipulation, hardware, or sim-to-real.
- This I01 result did not itself decide G1a/G3. Later independent evaluations report
  `G1a=PASS` and `G3=NO_GO`; E05 strategy G1b has been retired in favor of descriptive metrics.
  DPRef/Main and GPIS remain outside this I01 run.

## Generated artifacts

- Review page: [`../generated/i01_bunny_physics/index.html`](../generated/i01_bunny_physics/index.html)
- Dashboard: [`../generated/i01_bunny_physics/i01_bunny_dashboard.png`](../generated/i01_bunny_physics/i01_bunny_dashboard.png)
- Synchronized video: [`../generated/i01_bunny_physics/i01_fixed_vs_variable.mp4`](../generated/i01_bunny_physics/i01_fixed_vs_variable.mp4)
- Machine summary: [`../generated/i01_bunny_physics/summary.json`](../generated/i01_bunny_physics/summary.json)
- Episode table: [`../generated/i01_bunny_physics/episodes.csv`](../generated/i01_bunny_physics/episodes.csv)
- Full traces: `Module/generated/i01_bunny_physics/traces.npz` and six per-episode NPZ files.

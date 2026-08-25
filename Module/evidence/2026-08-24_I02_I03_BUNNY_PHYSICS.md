# I02/I03 Bunny Physics Evidence — 2026-08-24

## Scope and provenance

- Frozen protocol: `Module/I02_I03_BUNNY_PROTOCOL.md`
- Module IDs: `I02-PHY-BUNNY-v1`, `I03-PHY-BUNNY-v1`
- Formal command:
  `/home/ferry/data/Anaconda/envs/handcomp/bin/python -m Module.i02_i03_bunny_physics.benchmark`
- Output: `Module/generated/i02_i03_bunny_physics/`
- Formal matrix: cells `i02_long/i02_short/i03_beam/i03_shadow`, seeds `7/11/19`, 20 s,
  timestep 0.002 s, gravity 0, paired initialization.
- Environment: Python 3.10.0, NumPy 2.2.6, MuJoCo 3.6.0, SciPy 1.15.3,
  trimesh 4.11.5; Intel i7-13700, 24 logical CPUs.
- Git HEAD at run: `f1f92501d96910842533788e734eab285f4c8419`; worktree dirty because the
  authorized modules were not committed. Source hashes were captured before and after the 12 episodes;
  `source_stability=PASS`.
- DP, DPRef, GPIS, prediction-suffix commands and Shadow execution authority were disabled.

## Shared physical task

The fixed Bunny visual/audit surface is the exact canonical triangle mesh; its deterministic `181 x 181`
upper envelope is the MuJoCo collision surface. The path moves 0→10 mm, holds for the planning plateau,
moves 10→60 mm, returns 60→10 mm, and holds; scheduled cumulative distance is 110 mm. A scored contact
must have measured fingertip--Bunny force and exact-mesh residual at most 2.5 mm.

All executing prefixes start from a measured nonempty contact set, are optimized by M09, certified by M10,
executed by M06, and terminate at a measured micro barrier. M11/M12 never issue commands. The decision
profile retains the I01 LEAP servo (`kp=22`); after the 4–7 s decision plateau, every cell uses the same
long-traversal compliance profile (`kp=16`) with state-preserving MCC handoff.

## I02 result — EVALUATED / NOT_MET

| metric | LONG | SHORT |
| --- | ---: | ---: |
| common task pass | 3/3 | 3/3 |
| measured 4→3→4 | 3/3 | 3/3 |
| supported traversal median | 101.767 mm | 101.808 mm |
| terminal prediction error median | 1.476 mm | 1.467 mm |
| worst peak force | 5.637 N | 4.825 N |
| certificate / barrier total | 9 / 9 | 15 / 15 |
| over-force / non-tip / authority / suffix command | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

Every successful SHORT episode has exactly three REPOSITION certificates and three fresh measured roots.
The task-pass count and traversal non-inferiority conditions pass. However, neither frozen robustness branch
passes:

- execution failure count is 0 for both cells;
- frozen error threshold is `0.80 * 1.476 + 0.250 = 1.431 mm`, while SHORT is 1.467 mm.

Therefore the short-prefix mechanism is valid and safe, but this experiment does not prove the required
closed-loop robustness improvement. I02 must remain `NOT_MET`.

## I03 result — EVALUATED / MET

| metric | BEAM | SHADOW |
| --- | ---: | ---: |
| selected edge | SLIDE(3), 3/3 | SLIDE(1), 3/3 |
| actual terminal viability | NONVIABLE, 3/3 | VIABLE, 3/3 |
| dead ends | 3 | 0 |
| common task pass | 0/3 | 3/3 |
| measured handover | 0/3 | 3/3 |
| supported traversal median | 7.111 mm | 101.125 mm |
| actual terminal margin | 0.0168–0.0205 rad | 0.0478–0.0480 rad |
| worst peak force | 7.422 N | 7.010 N |
| certificate / barrier total | 3 / 3 | 12 / 12 |

ShadowSucc reduces paired dead ends by 3 and improves median supported traversal by 94.014 mm, exceeding the
frozen 30 mm threshold. Every SHADOW terminal has at least one distinct successor, `M12.execution_authority`
is false, and suffix-command count is zero. I03 is `MET`.

## Timing boundary

Timing is CPU wall time inside the named component; it excludes rendering and file I/O. Across recommended
cells:

- controller P95: at most 0.809 ms/tick (I02-SHORT), 0.777 ms/tick (I03-SHADOW);
- MuJoCo step P95: at most 0.314 ms/tick;
- selected M09 P95: at most 2.399 ms (I02-SHORT), 1.720 ms (I03-SHADOW);
- M11 search: 11.07–11.19 ms;
- M12 predicted/actual: 0.362–0.370 ms / 0.475–0.483 ms;
- M10 exact Bunny audit P95: 0.393–0.459 s for I02-SHORT and 0.358–0.426 s for I03-SHADOW.

M10 is deliberately an exact, non-realtime prefix audit in this integration and dominates planning latency.
This result demonstrates safety/authority semantics, not a realtime planning claim.

## Gate and boundary

G3 requires I02 and I03 both to be MET. Since I02 is NOT_MET, `G3=NO_GO`; GPIS integration remains locked.
The result is limited to fixed known Bunny, gravity-off Geometry-Oracle + Explicit MCC. It is not evidence for
unknown objects, GPIS, full non-convex mesh collision dynamics, gravity-on, hardware, or sim-to-real.

Review artifacts:

- `Module/generated/i02_i03_bunny_physics/index.html`
- `Module/generated/i02_i03_bunny_physics/i02_i03_bunny_dashboard.png`
- `Module/generated/i02_i03_bunny_physics/i02_long_vs_short.mp4`
- `Module/generated/i02_i03_bunny_physics/i03_beam_vs_shadow.mp4`

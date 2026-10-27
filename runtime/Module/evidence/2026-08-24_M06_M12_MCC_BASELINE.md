# M06–M12 Oracle + MCC baseline module evidence

Date: 2026-08-24
Environment: `/home/ferry/data/Anaconda/envs/handcomp/bin/python`
Protocol: `Module/M06_M12_MCC_BASELINE_PROTOCOL.md`
Scope: Geometry-Oracle + Explicit Fingertip MCC baseline only; no DP; G1 unchanged.

## Reproduction

```bash
cd /home/ferry/data/Code2/Research/hand_comliance_control
PY=/home/ferry/data/Anaconda/envs/handcomp/bin/python
$PY -m unittest Module.tests.test_m06_m12_planner -v
$PY -m Module.m06_m12_benchmark
$PY -m Module.m06_m12_visual_demo --reuse-benchmark
```

Full regression after implementation: `88 tests in 39.480 s`, `OK`.

## Frozen local result

Machine: 13th Gen Intel Core i7-13700, 24 logical CPUs, Linux x86_64.

| Module | Effect | P95 latency | Verdict |
| --- | --- | ---: | --- |
| M06 | 5/5 semantic scenarios; 0 authority violations | 0.168 ms/step | `MET` |
| M07 | 15 modes; 131 legal edges; deterministic | 1.157 us/legality | `MET` |
| M08 | 4096 candidates; FN rate 0 | 3.626 us/screen | `MET` |
| M09 | 160 cases; success 1.0; target-error P95 2.97e-9 m | 1.159 ms/optimize | `MET` |
| M10 | 65 swept samples; 6/6 adversaries rejected | 3.031 ms/audit | `MET` |
| M11 | H2/H3 exhaustive optimum retained; score gap 0 | 50.97/122.69 ms/search | `MET` |
| M12 | 1024 states; viable/dead-end cases correct | 0.203 ms/state | `MET` |

M11 H3 optimized 96 edges versus exhaustive 246. The certified integration smoke committed
`MAKE(3)`, changed measured contact set `{1,2}->{1,2,3}`, preserved contact continuity 1.0, and
closed the micro-barrier with a real-state snapshot. Its two later prefixes remained prediction-only.

## Provenance and boundary

- protocol SHA-256: `a382f9a119543832ffe128a3cf0679da715f77c415c4760517d7fbffd348ba37`;
- implementation/benchmark code SHA-256:
  `70ad90375e9ceaee6584030a80b303c15185455ce75a57ce1d25029db8c0413b`;
- machine-readable evidence: `Module/generated/m06_m12_mcc_baseline/summary.json`;
- visual review: `Module/generated/m06_m12_mcc_baseline/index.html`;
- timing uses `time.perf_counter_ns`; exact boundaries are embedded in the summary;
- numeric acceptance uses an analytic plane and deterministic linearized validation backend;
- Stanford Bunny is visualization only;
- this is not FR3 nonlinear IK, long-horizon MuJoCo traversal, I01, G2/G3, gravity-on, hardware,
  GPIS, or DP/Main evidence.

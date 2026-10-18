# RAMP resubmission — scripts & reproduction

Everything the reviewer's `RAMP_experimental_tasks.md` asks for. Outputs go to
`resubmission/out/` (JSON + human-readable `.txt`); figures to `figures_paper/`.
Filled placeholders are aggregated in `out/ALL_PLACEHOLDERS.md`; a narrative
summary is in `RESULTS.md`.

## Environment
- Python 3.14.3, PyTorch **2.11.0+cu128** (CUDA 12.8), NumPy 2.4.2, SciPy 1.17.1.
- GPU: RTX 5070 Ti (Blackwell sm_120, 17 GB). CPU: i9-14900K, 64 GB RAM.
- The default optimiser config (M=16, L=64, B/mc=32, Np=200, G=300) is unchanged.

## What reuses existing data vs. re-runs
| Script | Task(s) | Compute |
|---|---|---|
| `task1_stats.py` | 1 (HV stats) | reprocess `results/experiment.pkl` (25 seeds) |
| `task5_similarity.py` | 5 | shared-batch eval only (no optimisation) |
| `task13_figures.py` | 13 | reprocess pickle -> Fig 1/2/3 |
| `rerun_fronts.py` | 1b + 2 | **re-run** 6 methods × 15 seeds (fronts, timing, mem) |
| `task1b_igd.py` | 1 (IGD) | analysis of `rerun_fronts.pkl` |
| `task2_runtime.py` | 2 | analysis of `rerun_fronts.pkl` |
| `task4_rmp_sweep.py` | 4 | **re-run** fixed RMP {0.1,0.3,0.5} × 15 seeds |
| `sensitivity.py` | 6,7,8 | **re-run** RAMP at B/L/M sweeps × 15 seeds |
| `task9_power.py` | 9 | **re-run** 3 methods, inv-gain power × 15 seeds |
| `task10_online.py` | 10 | **re-run** online per-realisation reference + timing |
| `task3_cpu_gpu.py` | 3 | **re-run** matched RAMP on CPU and GPU |
| `task13_sensfigs.py` | 13 | sensitivity/RMP figures |
| `aggregate_results.py` | — | collect all placeholders |
| `run_reruns.py` | — | orchestrates every dependent re-run in sequence |

## Reproduce from scratch
```bash
# fast, no optimiser re-run
python resubmission/task1_stats.py
python resubmission/task5_similarity.py
python resubmission/task13_figures.py

# the re-run foundation (GPU, ~1.5 h), then everything that depends on it
python resubmission/rerun_fronts.py --seeds 15 --device cuda
python resubmission/run_reruns.py --seeds 15 --device cuda
```

## Core code changes (minimal, additive, backward-compatible)
- `mfea.py` — `MFEAResult` now also records per-generation transfer/intra-task
  survival counts (used by the RMP sweep). No change to the optimisation itself.
- `config.py`, `comm.py`, `sensing.py`, `objectives.py`, `baselines.py` — added a
  `power_policy` knob (`"equal"` default reproduces the spec model exactly;
  `"inv_gain"` is the fairness-oriented ablation for Task 9). The `"equal"` path is
  the original scalar computation, unchanged.

## Key honest findings (do not overclaim)
- RAMP significantly beats Pooled & MOPSO (large, p<1e-6) and MO-DE (p_Holm≈0.03) on
  both HV and IGD; RAMP vs No-transfer is a **statistical tie** (as framing-C states).
- Regimes are **moderately** related (Spearman ρ≈0.55): transfer is justified, not
  redundant.
- Peak GPU memory is dominated by the **one-off archive cross-regime re-evaluation**,
  not the evolutionary search.

# EvoComm-MOMGRAP — Experiment Protocol

`scripts_marl/run_ablation.py` + `momgrap.marl.evaluation.statistics`.

## Evaluation grid (axes)

```
regime      : LoS / Rayleigh / Rician              (dynamics.regime_sampling)
CSI error   : tau = 0.00 / 0.05 / 0.10 / 0.20 / 0.30
actor delay : 0 / 1 / 3 / 5                          (csi.actor_delay_steps)
comm budget : 0 / 1 / 2 / full                       (communication.max_messages_per_step)
comm method : none / full / random / learned / EA-graph
initialization : random / uspa_zero_phase / ramp_knee
memory      : recurrent (feed-forward not implemented — see limitations)
```

`configs_marl/eval.yaml` defines a small default subset; expand the axes for a
paper-scale sweep.

## Raw metrics (per cell)

min-user rate, sum-rate, sensing illumination (dB), outage, requested / delivered /
dropped messages, reconfiguration cost, constraint violation, episode return, runtime,
GPU memory.

## Statistical protocol

- **Independent training seeds** for policies; **paired evaluation seeds** across all
  cells (same seed → same channel realisation, common random numbers).
- `mean_ci` — mean ± std + 95% CI (Student-t).
- `paired_t_test` (`scipy.stats.ttest_rel`) + `cohens_d_paired` effect size.
- `holm_correction` — Holm-Bonferroni step-down when comparing many methods.
- Do **not** claim significance from too few runs.

## Pareto reporting

Objectives: maximize min-user rate, maximize sensing illumination, minimize messages.

- `pareto_front` — raw non-dominated points (m-objective).
- `pareto_coverage` — C-metric C(A,B).
- `epsilon_indicator` — additive epsilon indicator.
- `pairwise_2d_projections` — 2D projections for plotting.
- **No 3D hypervolume**: `momgrap.metrics.hypervolume_2d` is 2-objective only and is not
  mis-adapted; add a verified HV library if 3D HV is required.

## Outputs

```
outputs_marl/ablation/<run>/
  cells.csv            per-cell mean ± CI + runtime + GPU mem
  per_episode.csv      raw per-(cell,seed) metrics
  comparisons.json     paired tests vs 'none' + Holm + Cohen's d
  pareto.json          non-dominated comm methods per (regime,tau,...)
  config_resolved.yaml
  metadata.json        git commit/dirty, Python/Torch/CUDA/GPU, seeds
```

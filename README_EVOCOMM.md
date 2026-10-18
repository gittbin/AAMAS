# EvoComm-MOMGRAP

Multi-Agent Learning + Evolutionary Communication extension built **additively** on
top of the `momgrap` (MO-MGRAP-ISAC) codebase. It reuses the existing physics, GPU
kernels, MFEA, NSGA-II and robust archive unchanged, and adds a sequential
multi-agent environment under imperfect/delayed CSI with selective communication, a
recurrent MAPPO trainer, and an evolutionary search over the communication graph.

All new code lives under `momgrap/marl/`, `configs_marl/`, `scripts_marl/`,
`tests_marl/`. The legacy `momgrap` public APIs are preserved and `python smoke_test.py`
still passes.

---

## 1. What is inherited from `momgrap`

- **Physics / channels** — `channels.py` (`sample_environment`, steering, path loss,
  `build_comm_channels`), `comm.py` (`bd_precoding`, `power_allocation`, `evm_sinr_rate`),
  `sensing.py` (`effective_steering`, `beampattern` = sensing *illumination*),
  `genotype.py` (`decode`, `repair_positions`).
- **Optimization** — `nsga.py` (`fast_nondominated_sort`, `crowding_distance`,
  `environmental_selection`, m-objective), `operators.py`, `archive.py`, `mfea.py`
  (`run_mfea`, `unified_robust_front`).
- The physical model: 1 BS with M movable antennas, 1 RIS with L elements, K users,
  Q sensing targets, 3 propagation regimes (LoS / Rayleigh / Rician).

## 2. What is new (the MARL extension)

| Area | Module |
| --- | --- |
| Config | `momgrap/marl/config.py` (`EvoCommConfig` + `load_evocomm_config`) |
| Channel components | `physics/components.py` (direct/reflected/aggregate) |
| Imperfect CSI + delay | `physics/csi.py` |
| Temporal dynamics | `physics/dynamics.py` (mobility + Gauss–Markov fading) |
| Action codebooks | `actions/codebooks.py`, `actions/decoder.py` |
| Safety shield | `actions/safety.py` |
| One-interval backend | `physics/backend.py` |
| Communication | `communication/{graph,messages,budget,modes}.py` |
| Observations | `observations/{features,builder}.py`, `normalization.py` |
| Environment | `envs/ramp_parallel_env.py` (PettingZoo `ParallelEnv`) |
| Baselines + eval | `policies/baselines.py`, `evaluation/{rollout,metrics,comm_metrics,statistics}.py` |
| RAMP initialization | `physics/initialization.py` |
| RMAPPO | `algorithms/rmappo/{networks,buffer,trainer,checkpoint}.py` |
| Graph evolution | `evolution/{graph_encoding,graph_fitness,graph_search}.py` |

See `docs/EVOCOMM_ENVIRONMENT_SPEC.md`, `EVOCOMM_OBSERVATIONS.md`,
`EVOCOMM_ACTIONS.md`, `EVOCOMM_REWARD.md`, `EVOCOMM_EXPERIMENT_PROTOCOL.md`.

## 3. Install

```bash
pip install -r requirements-marl.txt   # inherits requirements.txt + gymnasium/pettingzoo/PyYAML/tensorboard/scipy
```

## 4. Legacy smoke (unchanged momgrap)

```bash
python smoke_test.py
```

> Stage C regenerates `figures/fig1..3.png`. Those hold offline experiment artifacts — back
> them up first (or run only Stage A/B) if you need to preserve them.

## 5. MARL smoke

```bash
python scripts_marl/smoke_backend.py     # physics forward chain, shapes
python scripts_marl/smoke_env.py         # PettingZoo API + random rollout
pytest -q tests_marl -m "not slow and not cuda"
```

## 6. Baseline evaluation

```bash
python scripts_marl/evaluate_baseline.py --policy heuristic_full_comm \
    --config configs_marl/smoke.yaml --seeds 0 1 2
```

## 7. Training (recurrent MAPPO)

```bash
python scripts_marl/train_rmappo.py --config configs_marl/smoke.yaml
# paper scale:
python scripts_marl/train_rmappo.py --config configs_marl/mvp.yaml --device cuda
```

## 8. Checkpoint evaluation

```bash
python scripts_marl/evaluate_checkpoint.py --checkpoint <run>/checkpoint.pt \
    --config configs_marl/smoke.yaml --seeds 0 1 2
```

## 9. RAMP initialization export

```bash
python scripts_marl/export_ramp_initializations.py --config configs_marl/smoke.yaml --seed 0
```

## 10. Evolutionary communication-graph search

```bash
python scripts_marl/evolve_comm_graph.py --config configs_marl/smoke.yaml \
    --policy heuristic_full_comm --pop-size 12 --max-gen 8
# or with a trained policy:
python scripts_marl/evolve_comm_graph.py --checkpoint <run>/checkpoint.pt --config <cfg>
```

## 11. Scientific ablation

```bash
python scripts_marl/run_ablation.py --eval-config configs_marl/eval.yaml
```

## 12. Output folders

Everything is written under `outputs_marl/` (never `results/` or `figures/`):

```
outputs_marl/eval/<run>/             baseline evaluation bundle
outputs_marl/train/<run>/            checkpoint + train.csv + TensorBoard
outputs_marl/initializations/<name>  RAMP init .pt/.json
outputs_marl/graph_search/<run>/     pareto graphs + baselines + cache
outputs_marl/ablation/<run>/         cells/per-episode/comparisons/pareto
```

## 13. Reproducibility

- Every config is resolved and saved (`config_resolved.yaml`); metadata records git
  commit + dirty status, Python/Torch/CUDA/GPU, and the seed list.
- Checkpoints store optimizer/scheduler/normalization/projection + Python/NumPy/
  Torch(CPU+CUDA) RNG for exact resume/eval.
- Graph fitness uses common random numbers (same seed/tau grid for every graph) and a
  fitness cache keyed by policy + config + seeds + taus + graph bitstring.

## 14. Limitations

- One BS, one RIS; no multi-cell interference.
- Codebook (discrete) actions; the actor does **not** synthesize a complex beamformer.
- Sensing objective is **illumination / beampattern quality**, not detection / CRB.
- Low-level BD uses imperfect CSI generated by the simulator; realized rate uses the
  true channel; CSI error is simulated (no measured channel traces).
- Graph search evaluates a **fixed** trained policy (no online co-evolution).
- Recurrent actor only (feed-forward memory ablation not implemented).

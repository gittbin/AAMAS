"""Scientific-evaluation grid + statistics + Pareto reporting (Task 18).

Sweeps (regime x tau x actor-delay x comm-budget x comm-method x init) from an eval
config, evaluates a fixed policy on PAIRED evaluation seeds, and reports per-cell
mean +/- 95% CI plus paired comparisons (Holm-corrected) and a Pareto front over
(min-user rate up, sensing up, messages down).

Run:
  python scripts_marl/run_ablation.py --eval-config configs_marl/eval.yaml
  python scripts_marl/run_ablation.py --eval-config configs_marl/eval.yaml --checkpoint <ckpt.pt>

Outputs -> outputs_marl/ablation/<run_id>/ (never touches results/ or figures/).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import itertools
import json
import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import yaml

from momgrap.marl.algorithms.rmappo.networks import build_actors
from momgrap.marl.config import evocomm_config_to_dict, load_evocomm_config, save_resolved_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.evaluation.metrics import collect_metadata
from momgrap.marl.evaluation.rollout import run_episode
from momgrap.marl.evaluation.statistics import (
    cohens_d_paired,
    holm_correction,
    mean_ci,
    paired_t_test,
    pareto_front,
)
from momgrap.marl.normalization import RunningMeanStd
from momgrap.marl.observations.builder import observation_sizes
from momgrap.marl.policies.baselines import TrainedActorPolicy, make_policy

_REGIME = {"LoS": "fixed_los", "Rayleigh": "fixed_rayleigh", "Rician": "fixed_rician"}
_METRICS = ("episode_return", "min_rate_mean", "min_rate_p10", "sum_rate_mean",
            "sensing_mean_db", "outage_rate", "messages_per_step",
            "reconfiguration_per_step", "violation_rate")


def _budget_value(b):
    return 3 if b == "full" else int(b)


def _make_policy_factory(args, base_evo, device):
    if args.checkpoint:
        obs_sizes = observation_sizes(base_evo)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        actors = build_actors(base_evo, obs_sizes)
        for a in actors:
            actors[a].load_state_dict(ckpt["actors"][a])
            actors[a].to(device)
        obs_rms = {a: RunningMeanStd(obs_sizes[a], device=device) for a in actors}
        for a in obs_rms:
            obs_rms[a].load_state_dict(ckpt["obs_rms"][a])
        return lambda: TrainedActorPolicy(actors, obs_rms, device=device, deterministic=True)
    return lambda: make_policy(args.policy, seed=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-config", default=str(_REPO_ROOT / "configs_marl" / "eval.yaml"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--policy", default=None, help="override eval-config policy")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--outdir", default=str(_REPO_ROOT / "outputs_marl" / "ablation"))
    args = ap.parse_args()

    spec = yaml.safe_load(pathlib.Path(args.eval_config).read_text(encoding="utf-8"))
    base_config = spec.get("base_config", str(_REPO_ROOT / "configs_marl" / "smoke.yaml"))
    seeds = spec.get("seeds", [0, 1, 2])
    args.policy = args.policy or spec.get("policy", "heuristic_full_comm")
    grid = spec["grid"]
    base_evo = load_evocomm_config(base_config)
    make_pol = _make_policy_factory(args, base_evo, args.device)

    axes = [grid["regimes"], grid["taus"], grid["delays"], grid["budgets"],
            grid["comm_methods"], grid["inits"]]

    cells, per_episode = [], []
    for regime, tau, delay, budget, method, init in itertools.product(*axes):
        evo = load_evocomm_config(base_config, overrides={
            "dynamics": {"regime_sampling": _REGIME[regime]},
            "csi": {"actor_delay_steps": int(delay)},
            "communication": {"max_messages_per_step": _budget_value(budget)},
        })
        comm_mode = method if method in ("none", "full", "random", "learned") else "learned"
        env = MOMGRAPParallelEnv(evo, seed=0, eval_mode=True, comm_mode=comm_mode, init_mode=init)
        env.set_tau(float(tau))

        t0 = time.perf_counter()
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        per_seed = {m: [] for m in _METRICS}
        for seed in seeds:
            result, _steps = run_episode(env, make_pol(), seed)
            for m in _METRICS:
                per_seed[m].append(getattr(result, m))
            per_episode.append({"regime": regime, "tau": tau, "delay": delay, "budget": budget,
                                "method": method, "init": init, "seed": seed,
                                **{m: getattr(result, m) for m in _METRICS}})
        runtime = time.perf_counter() - t0
        gpu_mem = (torch.cuda.max_memory_allocated() / 1e6
                   if args.device == "cuda" and torch.cuda.is_available() else 0.0)

        cell = {"regime": regime, "tau": tau, "delay": delay, "budget": budget,
                "method": method, "init": init, "runtime_s": runtime, "gpu_mem_mb": gpu_mem}
        for m in _METRICS:
            ci = mean_ci(per_seed[m])
            cell[f"{m}_mean"] = ci["mean"]
            cell[f"{m}_ci"] = ci["ci"]
        cell["_per_seed"] = per_seed
        cells.append(cell)

    # ---- paired comparisons (methods vs 'none') + Holm, per (regime,tau,delay,budget,init) ----
    comparisons = []
    key_axes = ("regime", "tau", "delay", "budget", "init")
    groups = {}
    for c in cells:
        groups.setdefault(tuple(c[k] for k in key_axes), []).append(c)
    for key, group in groups.items():
        by_method = {c["method"]: c for c in group}
        if "none" not in by_method:
            continue
        base_ret = by_method["none"]["_per_seed"]["episode_return"]
        tests = []
        for method, c in by_method.items():
            if method == "none":
                continue
            ret = c["_per_seed"]["episode_return"]
            tt = paired_t_test(ret, base_ret)
            tests.append({"method": method, "p": tt["p"], "t": tt["t"],
                          "cohens_d": cohens_d_paired(ret, base_ret),
                          "mean_return": c["episode_return_mean"]})
        holm = holm_correction([t["p"] for t in tests]) if tests else {"adjusted": [], "reject": []}
        for t, adj, rej in zip(tests, holm["adjusted"], holm["reject"]):
            t["p_holm"] = adj
            t["significant"] = bool(rej)
        comparisons.append({"key": dict(zip(key_axes, key)), "vs": "none", "tests": tests})

    # ---- Pareto over comm methods: (min_rate up, sensing up, messages down) ----
    pareto = []
    for key, group in groups.items():
        pts = [[c["min_rate_mean_mean"], c["sensing_mean_db_mean"], c["messages_per_step_mean"]]
               for c in group]
        nd = pareto_front(pts, maximize=[True, True, False])
        pareto.append({"key": dict(zip(key_axes, key)),
                       "objective_names": ["min_rate", "sensing_db", "messages_per_step"],
                       "non_dominated": [{"method": group[i]["method"], "point": pts[i]} for i in nd]})

    # ---- write outputs ----
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"{pathlib.Path(args.eval_config).stem}_{stamp}"
    out = pathlib.Path(args.outdir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    cell_fields = [k for k in cells[0] if k != "_per_seed"]
    with open(out / "cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cell_fields)
        w.writeheader()
        for c in cells:
            w.writerow({k: c[k] for k in cell_fields})
    with open(out / "per_episode.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_episode[0].keys()))
        w.writeheader()
        w.writerows(per_episode)
    (out / "comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    (out / "pareto.json").write_text(json.dumps(pareto, indent=2), encoding="utf-8")
    save_resolved_config(base_evo, out / "config_resolved.yaml")
    meta = collect_metadata(seeds)
    meta.update({"eval_config": args.eval_config, "base_config": base_config,
                 "policy": args.policy if not args.checkpoint else f"ckpt:{args.checkpoint}",
                 "n_cells": len(cells), "grid": grid})
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"ablation: {len(cells)} cells, {len(seeds)} seeds -> {out}")
    for comp in comparisons[:3]:
        print(f"  {comp['key']}")
        for t in comp["tests"]:
            sig = "*" if t["significant"] else " "
            print(f"    {t['method']:8s} vs none  d={t['cohens_d']:+.2f}  p_holm={t['p_holm']:.3f} {sig}")


if __name__ == "__main__":
    main()

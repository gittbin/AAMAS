"""Extended paper results (all evaluation, no training) for the 8-page version.

E1  actor CSI-delay robustness (delta in {0,1,3,5}): learned vs noop
E2  per-regime paired significance (one-sample t-test, 5 training seeds)
E3  evolutionary communication-graph search on a fixed trained policy (real Pareto)

Writes outputs_marl/paper/ext_results.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
from scipy import stats

from momgrap.marl.algorithms.rmappo.networks import build_actors
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.evaluation.rollout import run_evaluation
from momgrap.marl.evaluation.statistics import mean_ci
from momgrap.marl.evolution.graph_encoding import GraphConstraints, encode_graph
from momgrap.marl.evolution.graph_fitness import make_cached_evaluator
from momgrap.marl.evolution.graph_search import evaluate_preset_graphs, run_graph_search
from momgrap.marl.normalization import RunningMeanStd
from momgrap.marl.observations.builder import observation_sizes
from momgrap.marl.policies.baselines import TrainedActorPolicy, make_policy

_REGIME = {"LoS": "fixed_los", "Rayleigh": "fixed_rayleigh", "Rician": "fixed_rician"}


def _load(evo, ckpt_dir, seeds, device):
    os_ = observation_sizes(evo)
    out = []
    for s in seeds:
        ck = torch.load(pathlib.Path(ckpt_dir) / f"seed{s}" / "checkpoint.pt",
                        map_location=device, weights_only=False)
        actors = build_actors(evo, os_)
        rms = {a: RunningMeanStd(os_[a]) for a in actors}
        for a in actors:
            actors[a].load_state_dict(ck["actors"][a])
            rms[a].load_state_dict(ck["obs_rms"][a])
        out.append((actors, rms))
    return out


def _learned_means(evo, learned, eval_seeds, device, metric):
    vals = []
    for actors, rms in learned:
        env = MOMGRAPParallelEnv(evo, seed=0, eval_mode=True, comm_mode="learned")
        res, _ = run_evaluation(env, TrainedActorPolicy(actors, rms, device=device), eval_seeds)
        vals.append(float(np.mean([getattr(r, metric) for r in res])))
    return np.array(vals)


def _noop_mean(evo, eval_seeds, metric):
    env = MOMGRAPParallelEnv(evo, seed=0, eval_mode=True, comm_mode="learned")
    res, _ = run_evaluation(env, make_policy("noop"), eval_seeds)
    return float(np.mean([getattr(r, metric) for r in res]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_REPO_ROOT / "configs_marl" / "mvp.yaml"))
    ap.add_argument("--ckpt-dir", default=str(_REPO_ROOT / "outputs_marl" / "multiseed"))
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default=str(_REPO_ROOT / "outputs_marl" / "paper"))
    args = ap.parse_args()
    dev = args.device
    md, J = [], {}

    # ---------- E1: actor CSI delay robustness ----------
    md.append("## E1 — Actor CSI-delay robustness (learned vs noop)\n")
    md.append("| delay | learned return | noop return | learned min_rate | noop min_rate |")
    md.append("|--:|--:|--:|--:|--:|")
    E1 = {}
    for d in (0, 1, 3, 5):
        evo = load_evocomm_config(args.config, overrides={
            "physical": {"device_str": dev}, "csi": {"actor_delay_steps": d}})
        learned = _load(evo, args.ckpt_dir, args.train_seeds, dev)
        lr = _learned_means(evo, learned, args.eval_seeds, dev, "episode_return")
        lm = _learned_means(evo, learned, args.eval_seeds, dev, "min_rate_mean")
        nr = _noop_mean(evo, args.eval_seeds, "episode_return")
        nm = _noop_mean(evo, args.eval_seeds, "min_rate_mean")
        E1[d] = {"learned_ret": float(lr.mean()), "noop_ret": nr,
                 "learned_min": float(lm.mean()), "noop_min": nm}
        md.append(f"| {d} | {lr.mean():.2f} | {nr:.2f} | {lm.mean():.3f} | {nm:.3f} |")
    J["E1_delay"] = E1

    # ---------- E2: per-regime paired significance ----------
    md.append("\n## E2 — Per-regime significance (learned vs noop, one-sample t-test, n=5 seeds)\n")
    md.append("| regime | learned return | noop return | t(ret) | p(ret) | learned min | noop min | t(min) | p(min) |")
    md.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    E2 = {}
    for reg, samp in _REGIME.items():
        evo = load_evocomm_config(args.config, overrides={
            "physical": {"device_str": dev}, "dynamics": {"regime_sampling": samp}})
        learned = _load(evo, args.ckpt_dir, args.train_seeds, dev)
        lr = _learned_means(evo, learned, args.eval_seeds, dev, "episode_return")
        lm = _learned_means(evo, learned, args.eval_seeds, dev, "min_rate_mean")
        nr = _noop_mean(evo, args.eval_seeds, "episode_return")
        nm = _noop_mean(evo, args.eval_seeds, "min_rate_mean")
        tr, pr = stats.ttest_1samp(lr, nr)
        tm, pm = stats.ttest_1samp(lm, nm)
        E2[reg] = {"learned_ret": float(lr.mean()), "noop_ret": nr, "t_ret": float(tr), "p_ret": float(pr),
                   "learned_min": float(lm.mean()), "noop_min": nm, "t_min": float(tm), "p_min": float(pm)}
        md.append(f"| {reg} | {lr.mean():.2f} | {nr:.2f} | {tr:+.2f} | {pr:.3f} | "
                  f"{lm.mean():.3f} | {nm:.3f} | {tm:+.2f} | {pm:.3f} |")
    J["E2_regime_sig"] = E2

    # ---------- E3: EA graph search on the trained policy (seed 0) ----------
    md.append("\n## E3 — Evolutionary comm-graph search on the trained policy (Pareto)\n")
    evo = load_evocomm_config(args.config, overrides={"physical": {"device_str": dev}})
    learned0 = _load(evo, args.ckpt_dir, [args.train_seeds[0]], dev)[0]
    actors0, rms0 = learned0
    env = MOMGRAPParallelEnv(evo, seed=0, eval_mode=True, comm_mode="learned")
    mk = lambda: TrainedActorPolicy(actors0, rms0, device=dev)
    evaluate_fn, cache = make_cached_evaluator(env, mk, seeds=[0, 1], taus=[0.1])
    result = run_graph_search(evaluate_fn, GraphConstraints(), pop_size=12, max_gen=6, seed=0)
    presets = evaluate_preset_graphs(evaluate_fn, extra_random=2)
    md.append("Objectives (all maximized): F1 min-rate, F2 sensing(dB), F3 -messages, F4 -outage.\n")
    md.append("| graph | bits | F1 min-rate | F2 sensing | F3 -msgs | F4 -outage |")
    md.append("|---|---|--:|--:|--:|--:|")
    for name, d in presets.items():
        o = d["objectives"]
        md.append(f"| {name} | {''.join(map(str,d['bits']))} | {o[0]:.3f} | {o[1]:.2f} | {o[2]:.3f} | {o[3]:.3f} |")
    md.append(f"\nPareto-optimal graphs found: {len(result.pareto_bits)} "
              f"(unique bitstrings evaluated: {len(cache.store)}).")
    J["E3_graph"] = {"pareto": [{"bits": b, "obj": o} for b, o in
                                zip(result.pareto_bits, result.pareto_objectives)],
                     "presets": presets, "n_evals": len(cache.store)}

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ext_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "ext_results.json").write_text(json.dumps(J, indent=2), encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> {out}/ext_results.md + ext_results.json")


if __name__ == "__main__":
    main()

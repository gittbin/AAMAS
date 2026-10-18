"""Compile all paper-ready results from the trained multi-seed checkpoints.

Evaluation only (no training). Produces, for the learned RMAPPO ensemble (independent
training seeds) evaluated on paired eval seeds:
  T1  main comparison vs baselines (mean +/- 95% CI, one-sample t-test + Cohen's d + Holm)
  T2  communication ablation (none/full/random/learned) + message utility
  T3  CSI robustness (tau sweep) learned vs noop
  T4  per-regime breakdown (LoS/Rayleigh/Rician)
  T5  learning curve (mean +/- std across seeds from eval.csv)

Writes outputs_marl/paper/{paper_results.md, paper_results.json}.
"""

from __future__ import annotations

import argparse
import csv
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
from momgrap.marl.evaluation.statistics import holm_correction, mean_ci
from momgrap.marl.normalization import RunningMeanStd
from momgrap.marl.observations.builder import observation_sizes
from momgrap.marl.policies.baselines import TrainedActorPolicy, make_policy

_METRICS = ("episode_return", "min_rate_mean", "sum_rate_mean", "sensing_mean_db",
            "messages_per_step", "outage_rate")
_REGIME = {"LoS": "fixed_los", "Rayleigh": "fixed_rayleigh", "Rician": "fixed_rician"}


def _load_learned(evo, ckpt_dir, train_seeds, device):
    obs_sizes = observation_sizes(evo)
    policies = []
    for s in train_seeds:
        ck = torch.load(pathlib.Path(ckpt_dir) / f"seed{s}" / "checkpoint.pt",
                        map_location=device, weights_only=False)
        actors = build_actors(evo, obs_sizes)
        rms = {a: RunningMeanStd(obs_sizes[a]) for a in actors}
        for a in actors:
            actors[a].load_state_dict(ck["actors"][a])
            rms[a].load_state_dict(ck["obs_rms"][a])
        policies.append((actors, rms))
    return policies


def _eval(evo, policy_factory, eval_seeds, comm_mode="learned", tau=None):
    env = MOMGRAPParallelEnv(evo, seed=0, eval_mode=True, comm_mode=comm_mode)
    if tau is not None:
        env.set_tau(tau)
    results, _ = run_evaluation(env, policy_factory(), eval_seeds)
    return {m: np.array([getattr(r, m) for r in results]) for m in _METRICS}


def _learned_per_seed(evo, learned, eval_seeds, device, comm_mode="learned", tau=None):
    """Return {metric: array over train seeds of the per-seed mean}."""
    out = {m: [] for m in _METRICS}
    for actors, rms in learned:
        res = _eval(evo, lambda: TrainedActorPolicy(actors, rms, device=device, deterministic=True),
                    eval_seeds, comm_mode=comm_mode, tau=tau)
        for m in _METRICS:
            out[m].append(float(res[m].mean()))
    return {m: np.array(v) for m, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_REPO_ROOT / "configs_marl" / "mvp.yaml"))
    ap.add_argument("--ckpt-dir", default=str(_REPO_ROOT / "outputs_marl" / "multiseed"))
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--sweep-seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default=str(_REPO_ROOT / "outputs_marl" / "paper"))
    args = ap.parse_args()

    evo = load_evocomm_config(args.config, overrides={"physical": {"device_str": args.device}})
    learned = _load_learned(evo, args.ckpt_dir, args.train_seeds, args.device)
    md, J = [], {}

    # -------- T1: main comparison --------
    baselines = ["noop", "equal_power_no_comm", "static_ramp_init", "heuristic_no_comm",
                 "heuristic_full_comm", "random_comm_matched_budget", "random"]
    learned_ps = _learned_per_seed(evo, learned, args.eval_seeds, args.device)
    base_eval = {b: _eval(evo, lambda b=b: make_policy(b, seed=0), args.eval_seeds) for b in baselines}

    def sig(metric):
        lm = learned_ps[metric]
        rows, pvals = [], []
        for b in baselines:
            bm = base_eval[b][metric]
            t, p = stats.ttest_1samp(lm, float(bm.mean()))
            d = (lm.mean() - bm.mean()) / (lm.std(ddof=1) + 1e-12)
            rows.append({"baseline": b, "base_mean": float(bm.mean()), "base_ci": mean_ci(bm)["ci"],
                         "diff": float(lm.mean() - bm.mean()), "t": float(t), "p": float(p), "d": float(d)})
            pvals.append(float(p))
        holm = holm_correction(pvals)
        for r, ph, rej in zip(rows, holm["adjusted"], holm["reject"]):
            r["p_holm"] = ph
            r["sig"] = bool(rej)
        return rows

    J["T1_main"] = {
        "learned": {m: {"mean": float(learned_ps[m].mean()),
                        "ci95_over_train_seeds": mean_ci(learned_ps[m].tolist())["ci"]} for m in _METRICS},
        "significance_return": sig("episode_return"),
        "significance_min_rate": sig("min_rate_mean"),
    }
    md.append("## T1 — Main comparison (learned RMAPPO, 5 train seeds vs baselines)\n")
    md.append(f"Eval: {len(args.eval_seeds)} paired seeds, deterministic, tau_eval={evo.csi.tau_eval}. "
              "Learned mean +/- 95% CI over 5 training seeds; one-sample t-test of the 5 "
              "per-seed means vs each baseline; Holm-corrected.\n")
    md.append("| policy | return | min_rate | sum_rate | sens_dB | msgs/step | outage |")
    md.append("|---|--:|--:|--:|--:|--:|--:|")
    L = learned_ps
    md.append(f"| **LEARNED** | **{L['episode_return'].mean():.2f}** | **{L['min_rate_mean'].mean():.3f}** | "
              f"{L['sum_rate_mean'].mean():.2f} | {L['sensing_mean_db'].mean():.2f} | "
              f"{L['messages_per_step'].mean():.2f} | {L['outage_rate'].mean():.3f} |")
    for b in baselines:
        e = base_eval[b]
        md.append(f"| {b} | {e['episode_return'].mean():.2f} | {e['min_rate_mean'].mean():.3f} | "
                  f"{e['sum_rate_mean'].mean():.2f} | {e['sensing_mean_db'].mean():.2f} | "
                  f"{e['messages_per_step'].mean():.2f} | {e['outage_rate'].mean():.3f} |")
    md.append("\n**Significance (learned vs baseline), Holm-corrected:**\n")
    md.append("| baseline | Δreturn | d | p_holm(ret) | Δmin_rate | d | p_holm(min) |")
    md.append("|---|--:|--:|--:|--:|--:|--:|")
    sr = {r["baseline"]: r for r in J["T1_main"]["significance_return"]}
    sm = {r["baseline"]: r for r in J["T1_main"]["significance_min_rate"]}
    for b in baselines:
        r, m = sr[b], sm[b]
        md.append(f"| {b} | {r['diff']:+.2f} | {r['d']:+.2f} | {r['p_holm']:.3f}{'*' if r['sig'] else ''} | "
                  f"{m['diff']:+.3f} | {m['d']:+.2f} | {m['p_holm']:.3f}{'*' if m['sig'] else ''} |")

    # -------- T2: communication ablation --------
    md.append("\n## T2 — Communication ablation (learned policy, forced modes)\n")
    md.append("| comm mode | return | min_rate | msgs/step | message_utility (vs none) |")
    md.append("|---|--:|--:|--:|--:|")
    comm = {}
    for mode in ("none", "full", "random", "learned"):
        ps = _learned_per_seed(evo, learned, args.sweep_seeds, args.device, comm_mode=mode)
        comm[mode] = {m: float(ps[m].mean()) for m in _METRICS}
    base_none = comm["none"]["episode_return"]
    for mode in ("none", "full", "random", "learned"):
        c = comm[mode]
        md.append(f"| {mode} | {c['episode_return']:.2f} | {c['min_rate_mean']:.3f} | "
                  f"{c['messages_per_step']:.2f} | {c['episode_return']-base_none:+.2f} |")
    J["T2_comm_ablation"] = comm

    # -------- T3: CSI robustness (tau) --------
    md.append("\n## T3 — CSI robustness (tau sweep): learned vs noop\n")
    md.append("| tau | learned min_rate | noop min_rate | learned return | noop return |")
    md.append("|--:|--:|--:|--:|--:|")
    T3 = {}
    for tau in (0.0, 0.05, 0.10, 0.20, 0.30):
        lp = _learned_per_seed(evo, learned, args.sweep_seeds, args.device, tau=tau)
        nb = _eval(evo, lambda: make_policy("noop"), args.sweep_seeds, tau=tau)
        T3[tau] = {"learned_min": float(lp["min_rate_mean"].mean()), "noop_min": float(nb["min_rate_mean"].mean()),
                   "learned_ret": float(lp["episode_return"].mean()), "noop_ret": float(nb["episode_return"].mean())}
        r = T3[tau]
        md.append(f"| {tau:.2f} | {r['learned_min']:.3f} | {r['noop_min']:.3f} | "
                  f"{r['learned_ret']:.2f} | {r['noop_ret']:.2f} |")
    J["T3_tau"] = T3

    # -------- T4: per-regime --------
    md.append("\n## T4 — Per-regime (learned vs noop)\n")
    md.append("| regime | learned return | noop return | learned min_rate | noop min_rate |")
    md.append("|---|--:|--:|--:|--:|")
    T4 = {}
    for reg, samp in _REGIME.items():
        evo_r = load_evocomm_config(args.config, overrides={
            "physical": {"device_str": args.device}, "dynamics": {"regime_sampling": samp}})
        lp = _learned_per_seed(evo_r, learned, args.sweep_seeds, args.device)
        nb = _eval(evo_r, lambda: make_policy("noop"), args.sweep_seeds)
        T4[reg] = {"learned_ret": float(lp["episode_return"].mean()), "noop_ret": float(nb["episode_return"].mean()),
                   "learned_min": float(lp["min_rate_mean"].mean()), "noop_min": float(nb["min_rate_mean"].mean())}
        r = T4[reg]
        md.append(f"| {reg} | {r['learned_ret']:.2f} | {r['noop_ret']:.2f} | "
                  f"{r['learned_min']:.3f} | {r['noop_min']:.3f} |")
    J["T4_regime"] = T4

    # -------- T5: learning curve --------
    md.append("\n## T5 — Learning curve (eval return, mean +/- std over 5 seeds)\n")
    curves = {}
    for s in args.train_seeds:
        p = pathlib.Path(args.ckpt_dir) / f"seed{s}" / "eval.csv"
        if p.exists():
            for row in csv.DictReader(open(p, encoding="utf-8")):
                curves.setdefault(int(row["global_step"]), []).append(float(row["eval_return_mean"]))
    T5 = []
    md.append("| step | eval_return mean | std |")
    md.append("|--:|--:|--:|")
    for step in sorted(curves):
        vals = curves[step]
        if len(vals) == len(args.train_seeds):
            T5.append({"step": step, "mean": float(np.mean(vals)), "std": float(np.std(vals))})
            md.append(f"| {step} | {np.mean(vals):.2f} | {np.std(vals):.2f} |")
    J["T5_learning_curve"] = T5

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "paper_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "paper_results.json").write_text(json.dumps(J, indent=2), encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> {out}/paper_results.md  +  paper_results.json")


if __name__ == "__main__":
    main()

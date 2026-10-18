"""Train N independent RMAPPO seeds and test significance of learned vs a baseline.

Protocol (CLAUDE_CODE_TASKS Sec 23): independent training seeds, paired evaluation
seeds. Each trained policy is evaluated on the SAME eval seeds; the baseline (default
noop) is evaluated once on those seeds. The primary test is a one-sample t-test over
the per-training-seed mean differences (learned_seed_mean - baseline_mean), which
accounts for training-seed variance (the real source of uncertainty for "did the
algorithm learn to beat the baseline").

Run:
  python scripts_marl/train_multiseed.py --config configs_marl/mvp.yaml \
      --train-seeds 0 1 2 3 4 --timesteps 400000 --eval-seeds 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import gc
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

import numpy as np
import torch
from scipy import stats

from momgrap.marl.algorithms.rmappo.checkpoint import load_checkpoint, save_checkpoint
from momgrap.marl.algorithms.rmappo.networks import build_actors
from momgrap.marl.algorithms.rmappo.trainer import RMAPPOTrainer
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.evaluation.rollout import run_evaluation
from momgrap.marl.evaluation.statistics import mean_ci
from momgrap.marl.normalization import RunningMeanStd
from momgrap.marl.observations.builder import observation_sizes
from momgrap.marl.policies.baselines import TrainedActorPolicy, make_policy


def _eval_metric(evo, policy_factory, eval_seeds):
    """Return per-eval-seed (episode_return, min_rate_mean) arrays for a policy."""
    env = MOMGRAPParallelEnv(evo, seed=0, eval_mode=True, comm_mode="learned")
    results, _ = run_evaluation(env, policy_factory(), eval_seeds)
    return (np.array([r.episode_return for r in results]),
            np.array([r.min_rate_mean for r in results]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_REPO_ROOT / "configs_marl" / "mvp.yaml"))
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--timesteps", type=int, default=400000)
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--baseline", default="noop")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default=str(_REPO_ROOT / "outputs_marl" / "multiseed"))
    args = ap.parse_args()

    evo = load_evocomm_config(args.config, overrides={"physical": {"device_str": args.device}})
    obs_sizes = observation_sizes(evo)
    outroot = pathlib.Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)

    # ---- train each independent seed ----
    ckpts = {}
    for s in args.train_seeds:
        t0 = time.perf_counter()
        log_dir = outroot / f"seed{s}"
        trainer = RMAPPOTrainer(
            evo, lambda i, s=s: MOMGRAPParallelEnv(evo, seed=10_000 * s + i),
            device=args.device, log_dir=str(log_dir), seed=s)
        trainer.train(args.timesteps)
        ckpt = log_dir / "checkpoint.pt"
        save_checkpoint(ckpt, trainer)
        ckpts[s] = ckpt
        print(f"[train] seed={s} done: {trainer.global_step} steps in "
              f"{time.perf_counter()-t0:.0f}s -> {ckpt}", flush=True)
        del trainer
        gc.collect()

    # ---- baseline eval (once, on the eval seeds) ----
    base_ret, base_min = _eval_metric(evo, lambda: make_policy(args.baseline), args.eval_seeds)
    base_ret_mean, base_min_mean = float(base_ret.mean()), float(base_min.mean())

    # ---- evaluate each trained policy ----
    learned_ret_means, learned_min_means, per_seed = [], [], []
    for s in args.train_seeds:
        actors = build_actors(evo, obs_sizes)
        rms = {a: RunningMeanStd(obs_sizes[a]) for a in actors}
        # minimal manual load (only what the policy needs: actor weights + obs_rms)
        ck = torch.load(ckpts[s], map_location=args.device, weights_only=False)
        for a in actors:
            actors[a].load_state_dict(ck["actors"][a])
        for a in rms:
            rms[a].load_state_dict(ck["obs_rms"][a])
        ret, mn = _eval_metric(
            evo, lambda: TrainedActorPolicy(actors, rms, device=args.device, deterministic=True),
            args.eval_seeds)
        learned_ret_means.append(float(ret.mean()))
        learned_min_means.append(float(mn.mean()))
        per_seed.append({"train_seed": s, "return_mean": float(ret.mean()),
                         "min_rate_mean": float(mn.mean())})
        print(f"[eval ] seed={s}: return={ret.mean():.2f}  min_rate={mn.mean():.3f}", flush=True)

    # ---- statistics: one-sample t-test of per-seed means vs baseline mean ----
    def _report(name, learned_means, base_mean):
        arr = np.array(learned_means)
        diffs = arr - base_mean
        t, p = stats.ttest_1samp(arr, base_mean)
        ci = mean_ci(arr.tolist())
        return {"metric": name, "baseline_mean": base_mean,
                "learned_mean": ci["mean"], "learned_ci95": ci["ci"],
                "mean_diff": float(diffs.mean()), "t": float(t), "p": float(p),
                "n_train_seeds": len(arr)}

    stats_return = _report("episode_return", learned_ret_means, base_ret_mean)
    stats_min = _report("min_rate_mean", learned_min_means, base_min_mean)

    summary = {
        "config": args.config, "train_seeds": args.train_seeds, "timesteps": args.timesteps,
        "eval_seeds": args.eval_seeds, "baseline": args.baseline,
        "baseline_return_mean": base_ret_mean, "baseline_min_rate_mean": base_min_mean,
        "per_seed": per_seed, "significance": {"return": stats_return, "min_rate": stats_min},
    }
    (outroot / "multiseed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n================= MULTI-SEED SIGNIFICANCE vs", args.baseline, "=================")
    print(f"baseline({args.baseline})  return={base_ret_mean:.2f}  min_rate={base_min_mean:.3f}")
    for st in (stats_return, stats_min):
        print(f"{st['metric']:16s}: learned={st['learned_mean']:.3f}+/-{st['learned_ci95']:.3f}  "
              f"diff={st['mean_diff']:+.3f}  t={st['t']:+.2f}  p={st['p']:.4f}  "
              f"(n={st['n_train_seeds']} train seeds)  {'SIGNIFICANT' if st['p'] < 0.05 else 'n.s.'}")
    print("per-seed:", [f"s{d['train_seed']}={d['return_mean']:.1f}" for d in per_seed])


if __name__ == "__main__":
    main()

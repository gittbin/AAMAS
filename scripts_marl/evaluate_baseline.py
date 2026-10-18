"""Evaluate a baseline policy over several seeds and write a full result bundle.

Run:
  python scripts_marl/evaluate_baseline.py --policy heuristic_full_comm \
      --config configs_marl/smoke.yaml --seeds 0 1 2

Outputs (never touches results/ or figures/):
  outputs_marl/eval/<run_id>/{config_resolved.yaml, metadata.json, episodes.csv,
                              steps.csv, summary.json}
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
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

from momgrap.marl.config import load_evocomm_config, save_resolved_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.evaluation.metrics import collect_metadata, summarize_episodes
from momgrap.marl.evaluation.rollout import run_evaluation
from momgrap.marl.policies.baselines import BASELINE_REGISTRY, make_policy


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_REPO_ROOT / "configs_marl" / "smoke.yaml"))
    ap.add_argument("--policy", default="heuristic_full_comm", choices=sorted(BASELINE_REGISTRY))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tau", type=float, default=None, help="override per-step tau")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--outdir", default=str(_REPO_ROOT / "outputs_marl" / "eval"))
    args = ap.parse_args()

    evo = load_evocomm_config(args.config)
    env = MOMGRAPParallelEnv(evo, seed=args.seeds[0], eval_mode=True)
    if args.tau is not None:
        env.set_tau(args.tau)
    policy = make_policy(args.policy, seed=args.seeds[0])

    results, steps = run_evaluation(env, policy, args.seeds)
    summary = summarize_episodes(results)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"{args.policy}_{pathlib.Path(args.config).stem}_{stamp}"
    out = pathlib.Path(args.outdir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    save_resolved_config(evo, out / "config_resolved.yaml")
    meta = collect_metadata(args.seeds)
    meta.update({"policy": args.policy, "config": str(args.config), "tau_override": args.tau})
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_csv(out / "episodes.csv", [r.as_dict() for r in results])
    _write_csv(out / "steps.csv", steps)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"policy={args.policy}  episodes={len(results)}  -> {out}")
    er = summary.get("episode_return", {})
    mr = summary.get("min_rate_mean", {})
    mpp = summary.get("messages_per_step", {})
    print(f"  episode_return  = {er.get('mean', float('nan')):.3f} +/- {er.get('ci95', 0):.3f}")
    print(f"  min_rate_mean   = {mr.get('mean', float('nan')):.3f}")
    print(f"  messages/step   = {mpp.get('mean', float('nan')):.3f}")


if __name__ == "__main__":
    main()

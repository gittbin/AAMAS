"""Train the review-requested heavy experiments (centralized PPO, agent ablations,
message-cost sweep). Reduced-but-converged budget (200k steps, seeds as noted).
Each run saves its own checkpoint immediately so partial progress is preserved.

Writes outputs_marl/review/<run_name>/checkpoint.pt and a running summary.
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

from momgrap.marl.algorithms.rmappo.centralized import CentralizedPPOTrainer
from momgrap.marl.algorithms.rmappo.checkpoint import save_checkpoint
from momgrap.marl.algorithms.rmappo.trainer import RMAPPOTrainer
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv


def _factory(evo, base):
    return lambda i: MOMGRAPParallelEnv(evo, seed=base * 1000 + i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--outdir", default=str(_REPO_ROOT / "outputs_marl" / "review"))
    args = ap.parse_args()
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    def log(name, ev, t0):
        summary[name] = {"eval_return_mean": ev["eval_return_mean"], "eval_return_std": ev["eval_return_std"]}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[done] {name}: return={ev['eval_return_mean']:.2f} ({time.perf_counter()-t0:.0f}s)", flush=True)

    ev_cfg = load_evocomm_config(str(_REPO_ROOT / "configs_marl" / "mvp.yaml"),
                                 overrides={"physical": {"device_str": args.device}})

    # ---- 1. Centralized single-agent PPO (3 seeds) ----
    for s in (0, 1, 2):
        t0 = time.perf_counter()
        name = f"centralized_seed{s}"
        tr = CentralizedPPOTrainer(ev_cfg, _factory(ev_cfg, 20 + s), device=args.device,
                                   log_dir=str(out / name), seed=s)
        tr.train(args.timesteps)
        save_checkpoint_c(out / name / "checkpoint.pt", tr)
        log(name, tr.evaluate(args.eval_seeds), t0)
        del tr; gc.collect()

    # ---- 2. Agent ablations (power/ris/ma only, 2 seeds each) ----
    for active, tag in (["bs_agent"], "power_only"), (["ris_agent"], "ris_only"), (["ma_agent"], "ma_only"):
        for s in (0, 1):
            t0 = time.perf_counter()
            name = f"ablation_{tag}_seed{s}"
            tr = RMAPPOTrainer(ev_cfg, _factory(ev_cfg, 30 + s), device=args.device,
                               log_dir=str(out / name), seed=s, active_agents=active)
            tr.train(args.timesteps)
            save_checkpoint(out / name / "checkpoint.pt", tr)
            log(name, tr.evaluate(args.eval_seeds), t0)
            del tr; gc.collect()

    # ---- 3. Message-cost sweep (w_msg = 0 and 10x, 2 seeds each) ----
    for wmsg, tag in (0.0, "w0"), (0.1, "w10x"):
        cfg = load_evocomm_config(str(_REPO_ROOT / "configs_marl" / "mvp.yaml"),
                                  overrides={"physical": {"device_str": args.device},
                                             "reward": {"w_message": wmsg}})
        for s in (0, 1):
            t0 = time.perf_counter()
            name = f"msgcost_{tag}_seed{s}"
            tr = RMAPPOTrainer(cfg, _factory(cfg, 40 + s), device=args.device,
                               log_dir=str(out / name), seed=s, comm_mode="learned")
            tr.train(args.timesteps)
            save_checkpoint(out / name / "checkpoint.pt", tr)
            log(name, tr.evaluate(args.eval_seeds), t0)
            del tr; gc.collect()

    print("ALL REVIEW TRAINING DONE", flush=True)


def save_checkpoint_c(path, tr):
    import torch
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": tr.actor.state_dict(), "critic": tr.critic.state_dict(),
                "state_rms": tr.state_rms.state_dict(), "global_step": tr.global_step}, path)


if __name__ == "__main__":
    main()

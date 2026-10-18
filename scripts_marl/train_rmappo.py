"""Train recurrent MAPPO on the MARL environment, checkpoint, reload and evaluate.

Run (smoke):
  python scripts_marl/train_rmappo.py --config configs_marl/smoke.yaml

Outputs go under outputs_marl/train/<run_id>/ (never touches results/ or figures/).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from momgrap.marl.algorithms.rmappo.checkpoint import load_checkpoint, save_checkpoint
from momgrap.marl.algorithms.rmappo.trainer import RMAPPOTrainer
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_REPO_ROOT / "configs_marl" / "smoke.yaml"))
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    # keep env physics on the same device as the networks (CPU is faster here: small
    # N=1 tensors make GPU kernel-launch overhead dominate).
    evo = load_evocomm_config(args.config, overrides={"physical": {"device_str": args.device}})

    def factory(i):
        return MOMGRAPParallelEnv(evo, seed=1000 + i)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = pathlib.Path(args.log_dir or (_REPO_ROOT / "outputs_marl" / "train"
                                            / f"{pathlib.Path(args.config).stem}_{stamp}"))
    trainer = RMAPPOTrainer(evo, factory, device=args.device, log_dir=str(log_dir), seed=args.seed)

    before = trainer.evaluate(args.eval_seeds)
    print(f"[before] eval_return = {before['eval_return_mean']:.3f} +/- {before['eval_return_std']:.3f}")

    trainer.train(args.total_timesteps)
    ckpt = log_dir / "checkpoint.pt"
    save_checkpoint(ckpt, trainer)
    print(f"trained {trainer.update_index} updates ({trainer.global_step} steps); checkpoint -> {ckpt}")

    # reload into a fresh trainer and evaluate (verifies checkpoint round-trips)
    reloaded = RMAPPOTrainer(evo, factory, device=args.device, seed=args.seed)
    load_checkpoint(ckpt, reloaded)
    after = reloaded.evaluate(args.eval_seeds)
    print(f"[after ] eval_return = {after['eval_return_mean']:.3f} +/- {after['eval_return_std']:.3f}")


if __name__ == "__main__":
    main()

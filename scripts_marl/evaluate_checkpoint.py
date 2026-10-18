"""Evaluate a trained RMAPPO checkpoint over several seeds.

Run:
  python scripts_marl/evaluate_checkpoint.py --checkpoint outputs_marl/train/<run>/checkpoint.pt \
      --config configs_marl/smoke.yaml --seeds 0 1 2
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

from momgrap.marl.algorithms.rmappo.checkpoint import load_checkpoint
from momgrap.marl.algorithms.rmappo.trainer import RMAPPOTrainer
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=str(_REPO_ROOT / "configs_marl" / "smoke.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    evo = load_evocomm_config(args.config)

    def factory(i):
        return MOMGRAPParallelEnv(evo, seed=1000 + i)

    trainer = RMAPPOTrainer(evo, factory, device=args.device, seed=0)
    load_checkpoint(args.checkpoint, trainer)
    res = trainer.evaluate(args.seeds)
    res["checkpoint"] = args.checkpoint
    res["global_step"] = trainer.global_step
    res["update_index"] = trainer.update_index
    print(json.dumps(res, indent=2))
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

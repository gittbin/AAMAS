"""Train and measure known-value communication controls before wireless training."""
import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from momgrap.marl.algorithms.rmappo.trainer import RMAPPOTrainer
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.communication_diagnostic import CommunicationDiagnosticEnv
from momgrap.marl.policies.baselines import TrainedActorPolicy
from momgrap.marl.protocol import EVALUATOR_VERSION


def measure(trainer, factory, seeds):
    env = factory(0)
    policy = TrainedActorPolicy(trainer.actors, trainer.obs_rms, deterministic=False)
    requests, deliveries, accuracy, returns = [], [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        policy.reset()
        total = 0
        while env.agents:
            obs, rewards, _, _, infos = env.step(policy.act(obs, env))
            info = infos["bs_agent"]
            requests.append(info["message_requested"])
            deliveries.append(info["message_delivered"])
            if info["accuracy"] is not None:
                accuracy.append(info["accuracy"])
            total += rewards["bs_agent"]
        returns.append(total)
    return {"requests_per_step": float(np.mean(requests)), "delivered_per_step": float(np.mean(deliveries)),
            "accuracy": float(np.mean(accuracy)) if accuracy else None, "return": float(np.mean(returns)),
            "evaluation_episodes": len(returns)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16384)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", default="outputs_marl/paper_v2/diagnostics")
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"evaluator_version": EVALUATOR_VERSION, "steps": args.steps, "runs": [], "complete": False}
    for task, price in [("cost_only", 0.0), ("cost_only", 0.1), ("cost_only", 1.0), ("positive", 0.0)]:
        for seed in args.seeds:
            evo = load_evocomm_config(ROOT / "configs_marl/smoke.yaml", overrides={
                "communication": {"max_messages_per_step": 3 if task == "cost_only" else 1},
                "rmappo": {"total_timesteps": args.steps, "num_envs": 8, "rollout_length": 64,
                           "sequence_length": 16, "hidden_size": 32, "learning_rate": 0.003,
                           "entropy_coef": 0.001, "comm_entropy_coef": 0.001,
                           "eval_interval": 0, "save_interval": 0}})
            factory = lambda _, e=evo, t=task, p=price: CommunicationDiagnosticEnv(e, t, p)
            trainer = RMAPPOTrainer(evo, factory, seed=seed)
            before = measure(trainer, factory, range(10000, 10128))
            start = time.perf_counter()
            trainer.train()
            after = measure(trainer, factory, range(20000, 20256))
            run = {"task": task, "price": price, "training_seed": seed, "before": before,
                   "after": after, "training_seconds": time.perf_counter() - start,
                   "actual_transitions": trainer.global_step}
            result["runs"].append(run)
            (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(run), flush=True)
    result["complete"] = True
    positive = [r["after"]["accuracy"] for r in result["runs"] if r["task"] == "positive"]
    paid = [r for r in result["runs"] if r["task"] == "cost_only" and r["price"] > 0]
    result["passed"] = (all(a >= 0.8 for a in positive) and
                        all(r["after"]["requests_per_step"] < 0.2 * r["before"]["requests_per_step"] for r in paid))
    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("DIAGNOSTICS PASSED" if result["passed"] else "DIAGNOSTICS FAILED", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Train one immutable paper-v2 condition with explicit budget and provenance."""
import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from momgrap.marl.algorithms.rmappo.centralized import CentralizedPPOTrainer
from momgrap.marl.algorithms.rmappo.checkpoint import save_checkpoint
from momgrap.marl.algorithms.rmappo.trainer import RMAPPOTrainer
from momgrap.marl.config import load_evocomm_config, evocomm_config_to_dict, save_resolved_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.protocol import EVALUATOR_VERSION, OBSERVATION_SCHEMA_VERSION

SUBSETS = {"power": ["bs_agent"], "ris": ["ris_agent"], "ma": ["ma_agent"],
           "power_ris": ["bs_agent", "ris_agent"], "power_ma": ["bs_agent", "ma_agent"],
           "ris_ma": ["ris_agent", "ma_agent"]}
CONDITIONS = ["full_nocomm", "full_comm", "centralized", "feedforward", "masked_csi",
              "price_free", "price_tenfold", "budget3_default", "budget3_free", "budget3_tenfold",
              "request_price"] + list(SUBSETS)


def source_digest():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "momgrap").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_config(condition, config_path, steps, seed):
    comm = condition in ("full_comm", "price_free", "price_tenfold", "request_price") or condition.startswith("budget3")
    price = 0.0 if condition.endswith("free") else 0.1 if condition.endswith("tenfold") else 0.01
    overrides = {"physical": {"device_str": "cpu"},
                 "communication": {"enabled": comm, "graph_type": "full" if comm else "none",
                                   "max_messages_per_step": 3 if condition.startswith("budget3") else 1,
                                   "message_cost": price},
                 "reward": {"w_message": price},
                 "csi": {"feature_mode": "masked" if condition == "masked_csi" else "normal"},
                 "rmappo": {"seed": seed, "total_timesteps": steps,
                            "recurrent_actor": condition != "feedforward",
                            "comm_entropy_coef": 0.01,
                            "centralized_actor_hidden_size": 224}}
    return load_evocomm_config(config_path, overrides=overrides)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400384)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default="configs_marl/paper_v2.yaml")
    parser.add_argument("--out", default="outputs_marl/paper_v2/train")
    args = parser.parse_args()
    torch.set_num_threads(1)
    evo = make_config(args.condition, args.config, args.steps, args.seed)
    folder = pathlib.Path(args.out) / args.condition / f"seed_{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    status_file = folder / "status.json"
    digest = source_digest()
    if status_file.exists():
        previous = json.loads(status_file.read_text(encoding="utf-8"))
        if previous.get("complete") and previous["source_sha256"] == digest and previous["config"] == evocomm_config_to_dict(evo):
            print(f"Already complete: {folder}", flush=True)
            return
        raise RuntimeError(f"Existing incompatible or incomplete run at {folder}; use another --out to preserve it")
    status = {"condition": args.condition, "training_seed": args.seed, "complete": False,
              "evaluator_version": EVALUATOR_VERSION, "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
              "source_sha256": digest, "config": evocomm_config_to_dict(evo), "network_device": args.device,
              "physics_device": "cpu", "torch_version": torch.__version__}
    status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
    save_resolved_config(evo, folder / "config_resolved.yaml")
    def factory(index):
        env = MOMGRAPParallelEnv(evo, seed=1000 + index, eval_mode=index == 10000)
        if args.condition == "request_price":
            env.set_message_pricing("requested")
        return env
    central = args.condition == "centralized"
    if central:
        trainer = CentralizedPPOTrainer(evo, factory, device=args.device, log_dir=folder, seed=args.seed)
        policy_parameters = sum(p.numel() for p in trainer.actor.parameters())
    else:
        trainer = RMAPPOTrainer(evo, factory, device=args.device, log_dir=folder, seed=args.seed,
                               active_agents=SUBSETS.get(args.condition))
        policy_parameters = sum(p.numel() for a in trainer.actors.values() for p in a.parameters())
    status["policy_parameters"] = policy_parameters
    status["critic_parameters"] = sum(p.numel() for p in trainer.critic.parameters())
    start = time.perf_counter()
    def progress(steps, updates):
        status.update(actual_transitions=steps, updates=updates, training_seconds=time.perf_counter() - start)
        temporary = status_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(status_file)
        if updates % 25 == 0:
            print(f"{args.condition}/seed_{args.seed}: {steps}/{args.steps} transitions", flush=True)
    # Development seeds are never final held-out scenario IDs.
    trainer.train(eval_seeds=[10000, 10001, 10002], progress_callback=progress)
    if central:
        torch.save({"actor": trainer.actor.state_dict(), "critic": trainer.critic.state_dict(),
                    "actor_obs_rms": trainer.actor_obs_rms.state_dict(), "state_rms": trainer.state_rms.state_dict(),
                    "config": evocomm_config_to_dict(evo), "global_step": trainer.global_step,
                    "evaluator_version": EVALUATOR_VERSION, "observation_schema_version": OBSERVATION_SCHEMA_VERSION},
                   folder / "checkpoint.pt")
    else:
        save_checkpoint(folder / "checkpoint.pt", trainer)
        if trainer.writer is not None:
            trainer.writer.close()
    status.update(complete=True, actual_transitions=trainer.global_step, training_seconds=time.perf_counter() - start)
    status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in status.items() if k != "config"}, indent=2), flush=True)


if __name__ == "__main__":
    main()

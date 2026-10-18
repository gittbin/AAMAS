"""Fresh seeds and selected checkpoints for all paper conditions; unchanged PPO/physics."""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path
import types
import numpy as np
import torch

from experiments_marl.paper_v3 import ImprovementTrainer, training_seed
from experiments_marl_v4.messages import MessageEnv, source_digest as v4_digest
from scripts_marl.train_paper_condition import make_config as condition_config, SUBSETS
from scripts_marl.evaluate_paper_v2 import CentralPolicy
from momgrap.marl.algorithms.rmappo.centralized import CentralizedPPOTrainer, CentralizedActor
from momgrap.marl.algorithms.rmappo.checkpoint import save_checkpoint
from momgrap.marl.algorithms.rmappo.networks import build_actors
from momgrap.marl.policies.baselines import TrainedActorPolicy
from momgrap.marl.normalization import RunningMeanStd
from momgrap.marl.observations.builder import observation_sizes
from momgrap.marl.config import load_evocomm_config, evocomm_config_to_dict
from momgrap.config import RegimeSpec
from momgrap.marl.protocol import EVALUATOR_VERSION

ROOT = Path(__file__).resolve().parents[1]
VERSION = "paper-completion-1"


def source_digest():
    digest = hashlib.sha256(v4_digest().encode())
    paths = list((ROOT / "experiments_paper_completion").glob("*.py"))
    paths += [ROOT / "configs_marl/paper_completion.json", ROOT / "scripts_marl/run_paper_completion.py",
              ROOT / "scripts_marl/report_paper_completion.py", ROOT / "scripts_marl/train_paper_condition.py",
              ROOT / "scripts_marl/evaluate_paper_baselines.py"]
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def config_for(condition, seed, steps):
    base = "full_comm" if condition.startswith("candidate_") or condition == "selected_graph" else condition
    cfg = condition_config(base, ROOT / "configs_marl/paper_v2.yaml", steps, seed)
    from dataclasses import replace
    return replace(cfg, rmappo=replace(cfg.rmappo, eval_interval=50176, save_interval=50176))


def env_for(cfg, condition, graph=None, **kwargs):
    variant = "comm_payload" if condition == "candidate_payload" else "comm_rotating_router" if condition == "candidate_router" else "comm_reference"
    if graph is not None:
        adjacency = torch.zeros(3, 3, dtype=torch.bool)
        for bit, (source, target) in enumerate(((0,1),(0,2),(1,0),(1,2),(2,0),(2,1))):
            adjacency[source, target] = bool(graph & (1 << bit))
        kwargs["custom_adjacency"] = adjacency
        from dataclasses import replace
        cfg = replace(cfg, communication=replace(cfg.communication, graph_type="custom"))
    env = MessageEnv(cfg, variant, **kwargs)
    if condition == "request_price":
        env.set_message_pricing("requested")
    return env


class CompletionTrainer(ImprovementTrainer):
    def __init__(self, cfg, factory, condition, **kwargs):
        base = "comm_reference" if cfg.communication.enabled else "no_comm"
        super().__init__(cfg, factory, base, active_agents=SUBSETS.get(condition), **kwargs)
        self.condition = condition
        self.experiment_settings = {"actor_projection":"raw", "clip_mode":"joint", "comm_entropy_coef":0.01,
                                   "payload":"local_semantic_12_v1" if condition == "candidate_payload" else "original",
                                   "router":"rotating_ties" if condition == "candidate_router" else "sender_index",
                                   "observation_schema_version":4 if condition == "candidate_payload" else 2}


class CompletionCentralTrainer(CentralizedPPOTrainer):
    def __init__(self, *args, **kwargs):
        self.episode_indices = {}
        super().__init__(*args, **kwargs)

    def _reset(self, env, seed):
        env_id = self.envs.index(env)
        index = self.episode_indices.get(env_id, -1) + 1
        self.episode_indices[env_id] = index
        actual = training_seed(self.seed, env_id, index, self.num_envs)
        if self.log_dir is not None:
            path = self.log_dir / "episode_seeds.csv"
            first = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if first:
                    writer.writerow(("training_seed","env_id","episode_index","scenario_seed"))
                writer.writerow((self.seed,env_id,index,actual))
        return super()._reset(env, actual)


def save_policy(path, trainer, condition):
    if condition == "centralized":
        ck = {"actor":trainer.actor.state_dict(), "critic":trainer.critic.state_dict(),
              "actor_obs_rms":trainer.actor_obs_rms.state_dict(), "state_rms":trainer.state_rms.state_dict(),
              "config":evocomm_config_to_dict(trainer.evo), "global_step":trainer.global_step,
              "optimizer":trainer.optimizer.state_dict(), "scheduler":trainer.scheduler.state_dict(),
              "update_index":trainer.update_index, "evaluator_version":EVALUATOR_VERSION, "observation_schema_version":2}
    else:
        save_checkpoint(path, trainer)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ck.update(experiment_settings=trainer.experiment_settings,
                  observation_schema_version=trainer.experiment_settings["observation_schema_version"])
    ck.update(experiment_version=VERSION, experiment_condition=condition, source_sha256=source_digest(),
              episode_indices=trainer.episode_indices, runtime_restorable=False)
    torch.save(ck, path)


def load_policy(path, condition=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    name = ck.get("experiment_condition")
    if ck.get("experiment_version") != VERSION or (condition is not None and name != condition) or ck.get("evaluator_version") != EVALUATOR_VERSION:
        raise ValueError("Requires a compatible completion checkpoint")
    expected_schema = 4 if name == "candidate_payload" else 2
    if ck.get("observation_schema_version") != expected_schema:
        raise ValueError("Completion observation contract mismatch")
    if name != "centralized":
        expected = {"actor_projection":"raw", "clip_mode":"joint", "comm_entropy_coef":0.01,
                    "payload":"local_semantic_12_v1" if name == "candidate_payload" else "original",
                    "router":"rotating_ties" if name == "candidate_router" else "sender_index",
                    "observation_schema_version":expected_schema}
        if ck.get("experiment_settings") != expected:
            raise ValueError("Completion payload/router contract mismatch")
    raw = dict(ck["config"])
    raw["physical"] = dict(raw["physical"])
    raw["physical"]["regimes"] = [RegimeSpec(**r) for r in raw["physical"]["regimes"]]
    cfg = load_evocomm_config(ROOT / "configs_marl/paper_v2.yaml", overrides=raw)
    sizes = observation_sizes(cfg)
    if name == "centralized":
        actor = CentralizedActor(sum(sizes.values()),9,19,9,cfg.rmappo.centralized_actor_hidden_size or cfg.rmappo.hidden_size)
        actor.load_state_dict(ck["actor"])
        actor.eval()
        rms = RunningMeanStd(sum(sizes.values()))
        rms.load_state_dict(ck["actor_obs_rms"])
        policy = CentralPolicy(actor, rms)
    else:
        actors = build_actors(cfg,sizes)
        rms = {}
        for agent, actor in actors.items():
            actor.load_state_dict(ck["actors"][agent])
            actor.eval()
            rms[agent] = RunningMeanStd(sizes[agent])
            rms[agent].load_state_dict(ck["obs_rms"][agent])
        policy = TrainedActorPolicy(actors,rms,active_agents=ck["active_agents"])
    return cfg,policy,ck

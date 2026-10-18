"""Isolated paper-v3 screening: seed namespaces, actor CSI scaling, gradient ablations."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from momgrap.config import RegimeSpec
from momgrap.marl.config import load_evocomm_config, evocomm_config_to_dict
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.algorithms.rmappo.trainer import RMAPPOTrainer
from momgrap.marl.algorithms.rmappo.checkpoint import save_checkpoint
from momgrap.marl.algorithms.rmappo.networks import build_actors
from momgrap.marl.normalization import RunningMeanStd
from momgrap.marl.observations.builder import observation_sizes
from momgrap.marl.policies.baselines import TrainedActorPolicy
from momgrap.marl.protocol import EVALUATOR_VERSION

ROOT = Path(__file__).resolve().parents[1]
VERSION = "paper-v3-screening-1"
CONDITIONS = ("no_comm", "comm_reference", "comm_rms_csi", "comm_low_entropy", "comm_separate_clip")
TRAIN_BASE, TRAIN_STRIDE = 100000000, 1000000


def source_digest():
    digest = hashlib.sha256()
    paths = list((ROOT / "momgrap").rglob("*.py")) + list((ROOT / "experiments_marl").rglob("*.py"))
    paths += [ROOT / "scripts_marl/run_paper_improvements.py", ROOT / "scripts_marl/paper_seed_audit.py",
              ROOT / "configs_marl/paper_v3_screening.json", ROOT / "configs_marl/paper_v2.yaml"]
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def training_seed(training_replicate, env_id, episode_index, num_envs):
    if training_replicate < 0 or not 0 <= env_id < num_envs or episode_index < 0:
        raise ValueError("Invalid training seed coordinates")
    offset = episode_index * num_envs + env_id
    if offset >= TRAIN_STRIDE:
        raise ValueError("Training seed namespace exhausted")
    return TRAIN_BASE + training_replicate * TRAIN_STRIDE + offset


def settings(condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    return {"actor_projection": "component_rms" if condition == "comm_rms_csi" else "raw",
            "clip_mode": "actor_critic_separate" if condition == "comm_separate_clip" else "joint",
            "comm_entropy_coef": 0.001 if condition == "comm_low_entropy" else 0.01,
            "observation_schema_version": 3 if condition == "comm_rms_csi" else 2}


def make_config(condition, seed, steps):
    spec = settings(condition)
    enabled = condition != "no_comm"
    return load_evocomm_config(ROOT / "configs_marl/paper_v2.yaml", overrides={
        "communication": {"enabled": enabled, "graph_type": "full" if enabled else "none"},
        "rmappo": {"seed": seed, "total_timesteps": steps, "comm_entropy_coef": spec["comm_entropy_coef"],
                   "eval_interval": 50176, "save_interval": 50176}})


def rms_projection(projection, channel):
    # Preserve physical gain in the unchanged log-magnitude/log-norm features.
    # The clamp handles an exactly zero component without amplifying roundoff.
    rms = channel.abs().square().mean().sqrt().clamp_min(1e-30)
    return projection.project(channel / rms)


class ImprovementEnv(MOMGRAPParallelEnv):
    def __init__(self, cfg, actor_projection="raw", **kwargs):
        if actor_projection not in ("raw", "component_rms"):
            raise ValueError(actor_projection)
        self.actor_projection = actor_projection
        super().__init__(cfg, **kwargs)

    def _apply_csi_features(self, observations):
        if self.actor_projection == "component_rms":
            # Only delayed estimates: physical evaluation, critic inputs, payloads,
            # channel acquisition and all RNG streams stay identical to v2.
            est = self.current_estimate()
            start = 4 + self.cfg.K
            for agent, component in (("bs_agent", est.aggregate), ("ris_agent", est.reflected),
                                     ("ma_agent", est.direct)):
                observations[agent][start:start + self.projection.out_dim] = rms_projection(self.projection, component[0, 0])
        return super()._apply_csi_features(observations)


def clip_gradients(actor_params, critic_params, maximum, mode):
    actor_params, critic_params = list(actor_params), list(critic_params)
    def norm(params):
        return torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in params if p.grad is not None]))
    actor_norm, critic_norm = norm(actor_params), norm(critic_params)
    if mode == "joint":
        norm_all = nn.utils.clip_grad_norm_(actor_params + critic_params, maximum, error_if_nonfinite=True)
        factor = min(1.0, maximum / (float(norm_all) + 1e-6))
        actor_factor = critic_factor = factor
    elif mode == "actor_critic_separate":
        nn.utils.clip_grad_norm_(actor_params, maximum, error_if_nonfinite=True)
        nn.utils.clip_grad_norm_(critic_params, maximum, error_if_nonfinite=True)
        actor_factor = min(1.0, maximum / (float(actor_norm) + 1e-6))
        critic_factor = min(1.0, maximum / (float(critic_norm) + 1e-6))
    else:
        raise ValueError(mode)
    return {"actor_grad_norm": float(actor_norm), "critic_grad_norm": float(critic_norm),
            "actor_clip_factor": actor_factor, "critic_clip_factor": critic_factor}


class ImprovementTrainer(RMAPPOTrainer):
    def __init__(self, cfg, env_factory, condition, **kwargs):
        self.condition = condition
        self.experiment_settings = settings(condition)
        self.episode_indices = {}
        super().__init__(cfg, env_factory, **kwargs)

    def _reset_env(self, env, seed):
        env_id = self.envs.index(env)
        episode = self.episode_indices.get(env_id, -1) + 1
        actual_seed = training_seed(self.seed, env_id, episode, self.num_envs)
        self.episode_indices[env_id] = episode
        if self.log_dir is not None:
            path = self.log_dir / "episode_seeds.csv"
            first = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if first:
                    writer.writerow(["training_seed", "env_id", "episode_index", "scenario_seed"])
                writer.writerow([self.seed, env_id, episode, actual_seed])
        return super()._reset_env(env, actual_seed)

    def update(self):
        # Same PPO objective/minibatches/Adam/LR as the frozen v2 trainer. The
        # clipping ablation changes only grouping, with common diagnostics.
        rc, clip = self.rc, self.rc.clip_coef
        names = ("policy", "value", "entropy", "physical_entropy", "communication_entropy", "approx_kl", "clipfrac",
                 "actor_grad_norm", "critic_grad_norm", "actor_clip_factor", "critic_clip_factor", "actor_parameter_update_norm")
        stats = {name: [] for name in names}
        gen = torch.Generator(device="cpu").manual_seed(self.seed + self.update_index)
        actor_params = [p for actor in self.actors.values() for p in actor.parameters()]
        critic_params = list(self.critic.parameters())
        for _ in range(rc.update_epochs):
            for batch in self.buffer.recurrent_minibatches(rc.num_minibatches, rc.sequence_length, gen):
                valid, adv = batch["valid"], batch["advantages"]
                n_valid = valid.sum().clamp_min(1.0)
                policy_loss, physical_entropy, communication_entropy, approx_kl, clipfrac = [torch.zeros((), device=self.device) for _ in range(5)]
                for a in self.agents:
                    lp, pe, ce, _ = self.actors[a].evaluate_actions_with_entropies(
                        batch["obs"][a], batch["actor_hidden"][a], batch["masks"],
                        batch["phys_actions"][a], batch["comm_actions"][a],
                        action_masks=(batch["phys_masks"][a], batch["comm_masks"][a]))
                    logratio = lp - batch["log_probs_old"][a]
                    ratio = logratio.exp()
                    policy_loss += (-torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv) * valid).sum() / n_valid
                    physical_entropy += (pe * valid).sum() / n_valid
                    communication_entropy += (ce * valid).sum() / n_valid
                    approx_kl += (((ratio - 1) - logratio) * valid).sum() / n_valid
                    clipfrac += (((ratio - 1).abs() > clip).float() * valid).sum() / n_valid
                entropy = physical_entropy + communication_entropy
                values, _ = self.critic.forward(batch["state"], batch["critic_hidden"], batch["masks"])
                returns, old = batch["returns"], batch["values_old"]
                v_clipped = old + (values - old).clamp(-clip, clip)
                value_loss = 0.5 * (torch.maximum((values - returns) ** 2, (v_clipped - returns) ** 2) * valid).sum() / n_valid
                comm_coef = rc.entropy_coef if rc.comm_entropy_coef is None else rc.comm_entropy_coef
                loss = policy_loss - rc.entropy_coef * physical_entropy - comm_coef * communication_entropy + rc.value_coef * value_loss
                if not torch.isfinite(loss):
                    self._on_nan(loss, policy_loss, value_loss, entropy)
                self.optimizer.zero_grad()
                loss.backward()
                diagnostics = clip_gradients(actor_params, critic_params, rc.max_grad_norm, self.experiment_settings["clip_mode"])
                previous = [p.detach().clone() for p in actor_params]
                self.optimizer.step()
                diagnostics["actor_parameter_update_norm"] = math.sqrt(sum(float((p.detach() - oldp).square().sum()) for p, oldp in zip(actor_params, previous)))
                for name, value in {"policy": policy_loss, "value": value_loss, "entropy": entropy,
                                    "physical_entropy": physical_entropy, "communication_entropy": communication_entropy,
                                    "approx_kl": approx_kl, "clipfrac": clipfrac}.items():
                    stats[name].append(float(value.detach()))
                for name, value in diagnostics.items():
                    stats[name].append(value)
        self.scheduler.step()
        self.update_index += 1
        return {**{key: float(np.mean(values)) for key, values in stats.items()},
                "explained_variance": self._explained_variance(), "learning_rate": self.optimizer.param_groups[0]["lr"]}


def save_experiment_checkpoint(path, trainer):
    save_checkpoint(path, trainer)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck.update(experiment_version=VERSION, experiment_condition=trainer.condition,
              experiment_settings=trainer.experiment_settings,
              observation_schema_version=trainer.experiment_settings["observation_schema_version"],
              seed_namespace={"base": TRAIN_BASE, "stride": TRAIN_STRIDE}, episode_indices=trainer.episode_indices,
              runtime_restorable=False, source_sha256=source_digest())
    torch.save(ck, path)


def load_experiment_policy(path, expected_condition=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    condition = ck.get("experiment_condition")
    if ck.get("experiment_version") != VERSION or condition not in CONDITIONS:
        raise ValueError("Requires a paper-v3 checkpoint; v2 and unversioned policies are rejected")
    spec = settings(condition)
    if (ck.get("experiment_settings") != spec or ck.get("observation_schema_version") != spec["observation_schema_version"]
            or ck.get("evaluator_version") != EVALUATOR_VERSION
            or expected_condition is not None and condition != expected_condition):
        raise ValueError("Checkpoint experiment/observation contract mismatch")
    raw = dict(ck["config"])
    raw["physical"] = dict(raw["physical"])
    raw["physical"]["regimes"] = [RegimeSpec(**r) for r in raw["physical"]["regimes"]]
    cfg = load_evocomm_config(ROOT / "configs_marl/paper_v2.yaml", overrides=raw)
    actors = build_actors(cfg, observation_sizes(cfg))
    rms = {}
    for a, actor in actors.items():
        actor.load_state_dict(ck["actors"][a])
        actor.eval()
        rms[a] = RunningMeanStd(observation_sizes(cfg)[a])
        rms[a].load_state_dict(ck["obs_rms"][a])
    policy = TrainedActorPolicy(actors, rms, active_agents=ck["active_agents"])
    return cfg, policy, ck


def restore_experiment_checkpoint(path, trainer):
    _, _, ck = load_experiment_policy(path, expected_condition=trainer.condition)
    if ck["config"] != evocomm_config_to_dict(trainer.evo):
        raise ValueError("Trainer configuration does not match checkpoint")
    for a in trainer.actors:
        trainer.actors[a].load_state_dict(ck["actors"][a])
        trainer.obs_rms[a].load_state_dict(ck["obs_rms"][a])
    trainer.critic.load_state_dict(ck["critic"])
    trainer.optimizer.load_state_dict(ck["optimizer"])
    trainer.scheduler.load_state_dict(ck["scheduler"])
    trainer.state_rms.load_state_dict(ck["state_rms"])
    trainer.global_step, trainer.update_index = ck["global_step"], ck["update_index"]
    trainer.active_agents = set(ck["active_agents"])
    # Explicitly do not claim simulator/runtime resumption. Restored weights and
    # statistics are for evaluation/probes; training processes never auto-resume.
    if ck["projection"] is not None:
        for env in trainer.envs:
            env.projection.load_state_dict(ck["projection"])
        trainer.projection.load_state_dict(ck["projection"])
    random.setstate(ck["py_rng"])
    np.random.set_state(ck["np_rng"])
    torch.set_rng_state(ck["torch_rng"].cpu())
    return ck

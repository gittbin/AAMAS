"""Versioned semantic payload and rotating-tie router ablations of frozen physics."""
from __future__ import annotations
import hashlib
from pathlib import Path
import types
import numpy as np
import torch

from experiments_marl.paper_v3 import (ImprovementTrainer, make_config as v3_config,
    source_digest as v3_digest)
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.communication.messages import build_agent_messages
from momgrap.marl.communication.budget import MessageRouter, ReceivedMessage, DeliveryLog
from momgrap.marl.algorithms.rmappo.checkpoint import save_checkpoint
from momgrap.marl.algorithms.rmappo.networks import build_actors
from momgrap.marl.normalization import RunningMeanStd
from momgrap.marl.observations.builder import observation_sizes
from momgrap.marl.policies.baselines import TrainedActorPolicy
from momgrap.marl.config import load_evocomm_config
from momgrap.config import RegimeSpec
from momgrap.marl.protocol import EVALUATOR_VERSION
from momgrap.marl.actions.decoder import decode_power_action
from momgrap.comm import bd_precoding

ROOT = Path(__file__).resolve().parents[1]
VERSION = "paper-v4-messages-1"
CONDITIONS = ("no_comm", "comm_reference", "comm_payload", "comm_rotating_router")


def source_digest():
    digest = hashlib.sha256(v3_digest().encode())
    paths = list((ROOT / "experiments_marl_v4").glob("*.py"))
    paths += [ROOT / "configs_marl/paper_v4_messages.json", ROOT / "scripts_marl/run_paper_messages.py"]
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def settings(condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    return {"actor_projection": "raw", "clip_mode": "joint", "comm_entropy_coef": 0.01,
            "payload": "local_semantic_12_v1" if condition == "comm_payload" else "original",
            "router": "rotating_ties" if condition == "comm_rotating_router" else "sender_index",
            "observation_schema_version": 4 if condition == "comm_payload" else 2}


def make_config(condition, seed, steps):
    settings(condition)
    return v3_config("no_comm" if condition == "no_comm" else "comm_reference", seed, steps)


def channel_summary(channel):
    """Only a sender's delayed local component; scale-free gains and coherence."""
    power = channel.abs().square().sum(-1).clamp_min(1e-30)
    log_norm = 0.5 * torch.log10(power)
    relative = log_norm - log_norm.mean()
    unit = channel / power.sqrt().unsqueeze(-1)
    gram = unit @ unit.conj().transpose(-1, -2)
    k = len(channel)
    coherence = gram[torch.triu(torch.ones(k, k, dtype=torch.bool, device=channel.device), diagonal=1)]
    pair_mean = coherence.mean() if coherence.numel() else channel.new_zeros(())
    return relative, log_norm, torch.stack((pair_mean.real, pair_mean.imag))


def semantic_messages(ctx, dimension):
    if dimension != 12 or ctx.K != 4:
        raise ValueError("local_semantic_12_v1 contract requires K=4, message_dim=12")
    old = build_agent_messages(ctx, dimension)
    prev_ris = int(getattr(ctx, "previous_ris_action", 0))
    changed = 0 if prev_ris == 0 else (1 if prev_ris <= 2 * ctx.ris_group_count else ctx.ris_group_count)
    old["ris_agent"][4] = ctx.ris_last_sign * changed / ctx.ris_group_count
    bs_relative, bs_norms, bs_coherence = channel_summary(ctx.estimated.aggregate[0, 0])
    ris_relative, _, _ = channel_summary(ctx.estimated.reflected[0, 0])
    ma_relative, _, _ = channel_summary(ctx.estimated.direct[0, 0])
    # The payload branch exposes private actuator summaries, rather than only
    # duplicating channel norms. BS fractions are a belief under the previous
    # profile and delayed local estimate, never realized post-action powers.
    offsets = (ctx.positions[0, :, 1:] - ctx.centers[:, 1:]) / max(ctx.half_extent, 1e-30)
    geometry = torch.cat((offsets.mean(0), offsets.std(0, unbiased=False)))
    half = ctx.theta.shape[-1] // 2
    phase_contrast = torch.stack((torch.cos(ctx.theta[0, :half]).mean() - torch.cos(ctx.theta[0, half:]).mean(),
                                  torch.sin(ctx.theta[0, :half]).mean() - torch.sin(ctx.theta[0, half:]).mean()))
    vectors = {"bs_agent": torch.cat((old["bs_agent"][:5], ctx.power_fractions, bs_coherence, bs_norms.max().reshape(1))),
               "ris_agent": torch.cat((old["ris_agent"][:6], ris_relative, phase_contrast)),
               "ma_agent": torch.cat((old["ma_agent"][:4], ma_relative, geometry))}
    output = {}
    for agent, vector in vectors.items():
        if vector.shape != (12,) or not torch.isfinite(vector).all():
            raise ValueError("Invalid semantic payload")
        output[agent] = vector
    return output


class RotatingRouter(MessageRouter):
    def submit(self, step, requests):
        requested = [r for r in requests if r.dest is not None]
        valid = [r for r in requested if bool(self.adj[r.sender, r.dest])]
        # Rotate only equal-priority ties; metadata, graph, budget and delay unchanged.
        ranked = sorted(valid, key=lambda r: (-float(r.priority), (int(r.sender) - step) % len(self.adj)))
        selected = ranked[:self.budget]
        bucket = self._pending.setdefault(step + self.delay, [])
        for request in selected:
            bucket.append((int(request.dest), ReceivedMessage(int(request.sender), request.message.clone())))
        return DeliveryLog(len(requested), len(requested) - len(valid), len(valid) - len(selected), len(selected))


class MessageEnv(MOMGRAPParallelEnv):
    def __init__(self, cfg, condition="comm_reference", **kwargs):
        self.condition = condition
        self.spec = settings(condition)
        super().__init__(cfg, **kwargs)
        if self.spec["router"] == "rotating_ties":
            self.router = RotatingRouter(self.adjacency, cfg.communication.max_messages_per_step, cfg.communication.delivery_delay_steps)
        # Bind an unchanged step bytecode to per-instance globals. No module-global
        # monkeypatch, source edit, altered physical calculation or cross-env coupling.
        namespace = dict(MOMGRAPParallelEnv.step.__globals__)
        namespace["build_agent_messages"] = self._messages
        method = types.FunctionType(MOMGRAPParallelEnv.step.__code__, namespace)
        self._base_step = types.MethodType(method, self)

    def _messages(self, ctx, dimension):
        if self.spec["payload"] == "original":
            return build_agent_messages(ctx, dimension)
        ctx.previous_ris_action = self._prev_actions.get("ris_agent", 0)
        estimate = ctx.estimated.aggregate
        ctx.power_fractions = (decode_power_action(ctx.power_action_index, estimate, bd_precoding(estimate, self.cfg),
                                self.cfg, self.evo.actions)[0, 0] / self.cfg.p_max_watt)
        return semantic_messages(ctx, dimension)

    def step(self, actions):
        return self._base_step(actions)


class MessageTrainer(ImprovementTrainer):
    def __init__(self, cfg, factory, condition, **kwargs):
        super().__init__(cfg, factory, "no_comm" if condition == "no_comm" else "comm_reference", **kwargs)
        self.condition = condition
        self.experiment_settings = settings(condition)


def save_policy(path, trainer):
    save_checkpoint(path, trainer)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck.update(experiment_version=VERSION, experiment_condition=trainer.condition,
              experiment_settings=trainer.experiment_settings,
              observation_schema_version=trainer.experiment_settings["observation_schema_version"],
              source_sha256=source_digest(), runtime_restorable=False,
              episode_indices=trainer.episode_indices, seed_namespace={"base": 100000000, "stride": 1000000})
    torch.save(ck, path)


def load_policy(path, condition=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    name = ck.get("experiment_condition")
    if ck.get("experiment_version") != VERSION or name not in CONDITIONS or (condition is not None and condition != name):
        raise ValueError("Requires matching paper-v4 condition/checkpoint")
    spec = settings(name)
    if ck.get("experiment_settings") != spec or ck.get("observation_schema_version") != spec["observation_schema_version"] or ck.get("evaluator_version") != EVALUATOR_VERSION:
        raise ValueError("Payload/router/schema checkpoint contract mismatch")
    raw = dict(ck["config"])
    raw["physical"] = dict(raw["physical"])
    raw["physical"]["regimes"] = [RegimeSpec(**r) for r in raw["physical"]["regimes"]]
    cfg = load_evocomm_config(ROOT / "configs_marl/paper_v2.yaml", overrides=raw)
    actors = build_actors(cfg, observation_sizes(cfg))
    rms = {}
    for agent, actor in actors.items():
        actor.load_state_dict(ck["actors"][agent])
        actor.eval()
        rms[agent] = RunningMeanStd(observation_sizes(cfg)[agent])
        rms[agent].load_state_dict(ck["obs_rms"][agent])
    return cfg, TrainedActorPolicy(actors, rms, active_agents=ck["active_agents"]), ck

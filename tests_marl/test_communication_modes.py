"""Tests for learned/forced communication modes + curriculum + ablation (Task 16)."""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pytest

from momgrap.marl.algorithms.rmappo.networks import build_actors
from momgrap.marl.algorithms.rmappo.trainer import RMAPPOTrainer
from momgrap.marl.communication.graph import build_adjacency
from momgrap.marl.communication.modes import (
    COMM_MODES,
    CommunicationCurriculum,
    comm_mask_for_mode,
    forced_comm_action,
)
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.evaluation.comm_metrics import evaluate_comm_modes
from momgrap.marl.normalization import RunningMeanStd
from momgrap.marl.observations.builder import observation_sizes
from momgrap.marl.policies.baselines import TrainedActorPolicy, make_policy

SMOKE = _REPO_ROOT / "configs_marl" / "smoke.yaml"


def _env(comm_mode="learned", **overrides):
    evo = load_evocomm_config(SMOKE, overrides=overrides or None)
    return MOMGRAPParallelEnv(evo, seed=0, comm_mode=comm_mode)


def _fixed(comm):
    return {"bs_agent": np.array([0, comm]), "ris_agent": np.array([0, comm]),
            "ma_agent": np.array([0, comm])}


# ---------------------------------------------------------------- unit: modes
def test_forced_comm_action():
    assert forced_comm_action(0, 2, "none") == 0
    assert forced_comm_action(0, 0, "full") == 1
    assert forced_comm_action(0, 2, "learned") == 2
    assert forced_comm_action(0, 0, "random", rand_int=5) == 5 % 3
    with pytest.raises(ValueError):
        forced_comm_action(0, 0, "bogus")


def test_comm_mask_for_mode():
    adj = build_adjacency("full")
    m_none = comm_mask_for_mode(0, adj, "none")
    assert m_none.tolist() == [True, False, False]
    m_full = comm_mask_for_mode(0, adj, "full")
    assert m_full[0].item() is False and (m_full[1] or m_full[2])
    m_learned = comm_mask_for_mode(0, adj, "learned")
    assert m_learned[0].item() is True   # no-send valid under learned


def test_curriculum():
    c = CommunicationCurriculum(enabled=True, warmup_updates=10, start_cost=0.0, final_cost=1.0)
    assert c.cost_at(0) == 0.0
    assert abs(c.cost_at(5) - 0.5) < 1e-9
    assert c.cost_at(10) == 1.0 and c.cost_at(20) == 1.0
    assert CommunicationCurriculum(enabled=False).cost_at(3) is None


# ---------------------------------------------------------------- env forcing
def test_env_full_mode_forces_send():
    env = _env(comm_mode="full", communication={"graph_type": "full"})
    env.reset(seed=0)
    _o, _r, _t, _tr, info = env.step(_fixed(comm=0))    # agents chose no-send
    assert info["bs_agent"]["message_delivered"] == 1   # forced -> delivered (budget 1)


def test_env_none_mode_forces_no_send():
    env = _env(comm_mode="none", communication={"graph_type": "full"})
    env.reset(seed=0)
    _o, _r, _t, _tr, info = env.step(_fixed(comm=1))     # agents chose send
    assert info["bs_agent"]["message_delivered"] == 0    # forced no-send
    assert info["bs_agent"]["message_requested"] == 0


def test_env_message_cost_override_reduces_reward():
    env1 = _env(comm_mode="full", communication={"graph_type": "full"})
    env1.reset(seed=0)
    _o, r1, _t, _tr, _i = env1.step(_fixed(comm=0))

    env2 = _env(comm_mode="full", communication={"graph_type": "full"})
    env2.set_message_cost(10.0)
    env2.reset(seed=0)
    _o, r2, _t, _tr, _i = env2.step(_fixed(comm=0))
    assert r2["bs_agent"] < r1["bs_agent"]               # higher message cost -> lower reward


# ---------------------------------------------------------------- ablation
def test_evaluate_comm_modes_ablation():
    def make_env(mode):
        evo = load_evocomm_config(SMOKE, overrides={"communication": {"graph_type": "full"}})
        return MOMGRAPParallelEnv(evo, seed=0, comm_mode=mode)

    out = evaluate_comm_modes(make_env, lambda: make_policy("noop"), seeds=[0, 1], modes=COMM_MODES)
    assert set(out.keys()) == set(COMM_MODES)
    assert out["none"]["messages_per_step"] == 0.0
    assert out["full"]["messages_per_step"] > 0.0
    assert out["none"]["message_utility"] == 0.0        # baseline
    for mode in COMM_MODES:
        assert np.isfinite(out[mode]["episode_return"])


# ---------------------------------------------------------------- trainer modes
def _tiny_evo():
    return load_evocomm_config(SMOKE, overrides={
        "rmappo": {"total_timesteps": 64},
        "communication": {"graph_type": "full"},
    })


def test_trainer_comm_mode_none_sends_nothing():
    evo = _tiny_evo()
    factory = lambda i: MOMGRAPParallelEnv(evo, seed=1000 + i)
    tr = RMAPPOTrainer(evo, factory, device="cpu", seed=0, comm_mode="none")
    metrics = tr.collect_rollout()
    assert metrics["messages"] == 0.0


def test_trainer_comm_mode_full_sends_messages():
    evo = _tiny_evo()
    factory = lambda i: MOMGRAPParallelEnv(evo, seed=1000 + i)
    tr = RMAPPOTrainer(evo, factory, device="cpu", seed=0, comm_mode="full")
    metrics = tr.collect_rollout()
    assert metrics["messages"] > 0.0


# ---------------------------------------------------------------- trained policy
def test_trained_actor_policy_runs():
    evo = load_evocomm_config(SMOKE)
    obs_sizes = observation_sizes(evo)
    actors = build_actors(evo, obs_sizes, hidden_size=16)
    obs_rms = {a: RunningMeanStd(obs_sizes[a]) for a in actors}
    policy = TrainedActorPolicy(actors, obs_rms, device="cpu", deterministic=True)

    env = MOMGRAPParallelEnv(evo, seed=0)
    from momgrap.marl.evaluation.rollout import run_episode
    result, steps = run_episode(env, policy, seed=0)
    assert result.length == evo.dynamics.episode_length
    assert np.isfinite(result.episode_return)

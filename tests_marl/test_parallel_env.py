"""Tests for the PettingZoo parallel environment (Task 10)."""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pytest
import torch
from pettingzoo.test import parallel_api_test

from momgrap.marl.communication.graph import AGENT_NAMES
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.observations.builder import build_actor_observations
from momgrap.marl.physics.csi import EstimatedChannels

SMOKE = _REPO_ROOT / "configs_marl" / "smoke.yaml"


def _env(**overrides) -> MOMGRAPParallelEnv:
    evo = load_evocomm_config(SMOKE, overrides=overrides or None)
    return MOMGRAPParallelEnv(evo, seed=0)


def _fixed_actions(env):
    return {a: np.array([0, 0], dtype=np.int64) for a in env.agents}


# ---------------------------------------------------------------- API
def test_pettingzoo_parallel_api():
    parallel_api_test(_env(), num_cycles=30)


def test_observations_within_space():
    env = _env()
    obs, _ = env.reset(seed=0)
    for a in AGENT_NAMES:
        assert env.observation_space(a).contains(obs[a])


# ---------------------------------------------------------------- determinism
def test_deterministic_reset():
    e1, e2 = _env(), _env()
    o1, _ = e1.reset(seed=3)
    o2, _ = e2.reset(seed=3)
    for a in AGENT_NAMES:
        assert np.array_equal(o1[a], o2[a])
    o3, _ = e1.reset(seed=4)
    assert not np.array_equal(o1["bs_agent"], o3["bs_agent"])


def test_random_rollout_runs():
    env = _env()
    env.reset(seed=1)
    steps = 0
    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, r, term, trunc, infos = env.step(actions)
        steps += 1
    assert steps == env.evo.dynamics.episode_length


# ---------------------------------------------------------------- finiteness
def test_thousand_steps_finite():
    env = _env()
    env.reset(seed=0)
    count = 0
    while count < 1000:
        if not env.agents:
            env.reset(seed=count)
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, term, trunc, infos = env.step(actions)
        for a in obs:
            assert np.isfinite(obs[a]).all()
        assert all(np.isfinite(v) for v in rewards.values())
        count += 1


# ---------------------------------------------------------------- reward
def test_reward_shared_across_agents():
    env = _env()
    env.reset(seed=0)
    _obs, rewards, _t, _tr, _i = env.step(_fixed_actions(env))
    vals = list(rewards.values())
    assert len(vals) == 3
    assert all(v == vals[0] for v in vals)


# ---------------------------------------------------------------- MA timescale
def test_ma_update_period():
    env = _env(actions={"ma_update_period": 4})
    env.reset(seed=0)
    positions = [env._pstate.positions.clone()]
    for _ in range(9):
        actions = {
            "bs_agent": np.array([0, 0]),
            "ris_agent": np.array([0, 0]),
            "ma_agent": np.array([1, 0]),   # expand_y every step
        }
        env.step(actions)
        positions.append(env._pstate.positions.clone())
    # positions change only on calls where step_count%4==0 (i.e. calls 1 and 5)
    changed = [not torch.equal(positions[i], positions[i - 1]) for i in range(1, len(positions))]
    assert changed[0] is True            # call 1 (step_count 0)
    assert changed[1] is False and changed[2] is False and changed[3] is False
    assert changed[4] is True            # call 5 (step_count 4)


# ---------------------------------------------------------------- messages
def test_message_delivery_and_recipient():
    send = {
        "bs_agent": np.array([0, 1]),    # bs -> ris (its first other)
        "ris_agent": np.array([0, 0]),
        "ma_agent": np.array([0, 0]),
    }
    nosend = _fixed_actions_dict()
    e_send, e_none = _env(), _env()
    e_send.reset(seed=0)
    e_none.reset(seed=0)
    o_s, _r, _t, _tr, info_s = e_send.step(send)
    o_n, _r, _t, _tr, info_n = e_none.step(nosend)

    assert info_s["bs_agent"]["message_delivered"] == 1
    assert info_n["bs_agent"]["message_delivered"] == 0
    # the delivered message shows up (only) in the recipient's next observation
    assert not np.array_equal(o_s["ris_agent"], o_n["ris_agent"])
    assert np.array_equal(o_s["bs_agent"], o_n["bs_agent"])
    assert np.array_equal(o_s["ma_agent"], o_n["ma_agent"])


def _fixed_actions_dict():
    return {
        "bs_agent": np.array([0, 0]),
        "ris_agent": np.array([0, 0]),
        "ma_agent": np.array([0, 0]),
    }


def test_full_vs_no_graph():
    send = {
        "bs_agent": np.array([0, 1]),
        "ris_agent": np.array([0, 0]),
        "ma_agent": np.array([0, 0]),
    }
    e_full = _env(communication={"graph_type": "full"})
    e_none = _env(communication={"graph_type": "none"})
    e_full.reset(seed=0)
    e_none.reset(seed=0)
    _o, _r, _t, _tr, info_f = e_full.step(send)
    _o, _r, _t, _tr, info_n = e_none.step(send)
    assert info_f["bs_agent"]["message_delivered"] == 1
    assert info_f["bs_agent"]["graph_rejected"] == 0
    assert info_n["bs_agent"]["message_delivered"] == 0
    assert info_n["bs_agent"]["graph_rejected"] == 1


# ---------------------------------------------------------------- tau
def test_tau_zero_and_high():
    for tau in (0.0, 0.4):
        env = _env()
        env.set_tau(tau)
        env.reset(seed=0)
        while env.agents:
            _o, r, _t, _tr, infos = env.step({a: env.action_space(a).sample() for a in env.agents})
        assert abs(infos["bs_agent"]["tau"] - tau) < 1e-9
        assert np.isfinite(r["bs_agent"])


# ---------------------------------------------------------------- truncation
def test_correct_truncation():
    env = _env()
    env.reset(seed=0)
    ep_len = env.evo.dynamics.episode_length
    for t in range(ep_len):
        _o, _r, term, trunc, _i = env.step(_fixed_actions(env))
        expected = (t == ep_len - 1)
        assert all(v == expected for v in trunc.values())
        assert all(v is False for v in term.values())
    assert env.agents == []


# ---------------------------------------------------------------- no leak
def test_actor_uses_estimate_not_truth():
    env = _env()
    env.set_tau(0.3)      # est != true, so this is a meaningful check
    env.reset(seed=2)
    env.step({a: env.action_space(a).sample() for a in env.agents})

    i = env._last_obs_inputs
    assert isinstance(i["delayed"], EstimatedChannels)     # no true-channel object
    rebuilt = build_actor_observations(
        i["delayed"], i["theta"], i["positions"], env.centers, env.group_map,
        i["prev_actions"], i["delivered"], i["tau"], i["step"], env.evo,
        env.projection, i["countdown"],
    )
    for a in AGENT_NAMES:
        assert torch.equal(env._current_obs[a], rebuilt[a])


# ---------------------------------------------------------------- torch wrapper
def test_torch_wrapper():
    from momgrap.marl.envs.wrappers import TorchParallelWrapper
    env = TorchParallelWrapper(_env())
    obs, _ = env.reset(seed=0)
    for a in AGENT_NAMES:
        assert isinstance(obs[a], torch.Tensor)
    assert isinstance(env.state(), torch.Tensor)
    obs, r, term, trunc, infos = env.step({a: env.action_space(a).sample() for a in env.agents})
    assert isinstance(obs["bs_agent"], torch.Tensor)

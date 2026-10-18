"""Tests for baseline policies + evaluation rollout (Task 11)."""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import math

import numpy as np
import pytest

from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.evaluation.metrics import EpisodeResult, summarize_episodes
from momgrap.marl.evaluation.rollout import run_episode, run_evaluation
from momgrap.marl.policies.baselines import BASELINE_REGISTRY, make_policy

SMOKE = _REPO_ROOT / "configs_marl" / "smoke.yaml"


def _env(**overrides) -> MOMGRAPParallelEnv:
    evo = load_evocomm_config(SMOKE, overrides=overrides or None)
    return MOMGRAPParallelEnv(evo, seed=0, eval_mode=True)


def _finite_result(r: EpisodeResult):
    for v in (r.episode_return, r.min_rate_mean, r.min_rate_p10, r.sum_rate_mean,
              r.sensing_mean_db, r.outage_rate, r.messages_per_step,
              r.reconfiguration_per_step, r.violation_rate, r.tau):
        assert math.isfinite(v)


# ---------------------------------------------------------------- all baselines
@pytest.mark.parametrize("name", sorted(BASELINE_REGISTRY))
def test_each_baseline_runs(name):
    env = _env()
    policy = make_policy(name, seed=0)
    result, steps = run_episode(env, policy, seed=0)
    assert result.length == env.evo.dynamics.episode_length
    assert len(steps) == result.length
    _finite_result(result)


@pytest.mark.parametrize("name", sorted(BASELINE_REGISTRY))
def test_baseline_actions_valid(name):
    env = _env()
    policy = make_policy(name, seed=0)
    env.reset(seed=0)
    obs = env._current_obs and env._to_numpy(env._current_obs)
    actions = policy.act(obs, env)
    for a in env.agents:
        assert env.action_space(a).contains(actions[a])


# ---------------------------------------------------------------- determinism
def test_deterministic_policies_reproducible():
    for name in ("noop", "heuristic_full_comm", "heuristic_no_comm", "random"):
        e1, e2 = _env(), _env()
        r1, _ = run_episode(e1, make_policy(name, seed=0), seed=0)
        r2, _ = run_episode(e2, make_policy(name, seed=0), seed=0)
        assert r1.as_dict() == r2.as_dict()


# ---------------------------------------------------------------- comm behaviour
def test_no_comm_policy_sends_nothing():
    env = _env(communication={"graph_type": "full"})
    r, _ = run_episode(env, make_policy("heuristic_no_comm"), seed=0)
    assert r.messages_per_step == 0.0


def test_full_comm_policy_sends_messages():
    env = _env(communication={"graph_type": "full"})
    r, _ = run_episode(env, make_policy("heuristic_full_comm"), seed=0)
    assert r.messages_per_step > 0.0


def test_equal_power_no_comm_sends_nothing():
    env = _env(communication={"graph_type": "full"})
    r, _ = run_episode(env, make_policy("equal_power_no_comm"), seed=0)
    assert r.messages_per_step == 0.0


# ---------------------------------------------------------------- static/no-op
def test_noop_no_reconfiguration():
    env = _env()
    r, steps = run_episode(env, make_policy("noop"), seed=0)
    assert r.reconfiguration_per_step == 0.0
    assert all(s["reconfiguration_cost"] == 0.0 for s in steps)


def test_static_ramp_is_static():
    env = _env()
    r, _ = run_episode(env, make_policy("static_ramp_init"), seed=0)
    assert r.reconfiguration_per_step == 0.0


# ---------------------------------------------------------------- aggregation
def test_run_evaluation_and_summary():
    env = _env()
    seeds = [0, 1, 2]
    results, all_steps = run_evaluation(env, make_policy("heuristic_full_comm"), seeds)
    assert len(results) == 3
    assert len(all_steps) == 3 * env.evo.dynamics.episode_length
    summary = summarize_episodes(results)
    assert summary["n_episodes"] == 3
    assert "mean" in summary["episode_return"] and "ci95" in summary["episode_return"]
    assert math.isfinite(summary["min_rate_mean"]["mean"])


def test_heuristic_uses_estimate_only_smoke():
    # sanity: heuristic runs at high tau (estimate != truth) and stays finite/valid
    env = _env()
    env.set_tau(0.3)
    r, _ = run_episode(env, make_policy("heuristic_full_comm"), seed=1)
    _finite_result(r)

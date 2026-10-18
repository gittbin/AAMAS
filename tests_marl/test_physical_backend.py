"""Tests for the one-interval physical backend (Task 07)."""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from momgrap.channels import sample_environment
from momgrap.comm import evm_sinr_rate
from momgrap.genotype import decode
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.physics.backend import evaluate_control_interval
from momgrap.objectives import evaluate_single_regime

SMOKE = _REPO_ROOT / "configs_marl" / "smoke.yaml"


def _setup(regime_index: int = 0, seed: int = 0):
    evo = load_evocomm_config(SMOKE)
    cfg = evo.physical                       # M=4, L=16, K=2, Q=1, cpu
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    geno = torch.rand(1, cfg.D, generator=g, device="cpu", dtype=cfg.dtype_real)
    P, theta = decode(geno, cfg)
    env = sample_environment(cfg.regimes[regime_index], 1, g, cfg)  # S=1
    return evo, cfg, geno, P, theta, env


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------- shapes
def test_output_shapes_and_finite():
    evo, cfg, _geno, P, theta, env = _setup()
    res = evaluate_control_interval(env, P, theta, 0, 0.1, cfg, evo, _gen(1))
    assert res.true_components.aggregate.shape == (1, 1, cfg.K, cfg.M)
    assert res.precoder.shape == (1, 1, cfg.M, cfg.K)
    assert res.p_user.shape == (1, 1, cfg.K)
    assert res.user_rates.shape == (cfg.K,)
    assert res.sensing_per_target.shape == (cfg.Q,)
    assert torch.isfinite(res.user_rates).all()
    assert torch.isfinite(res.sensing_per_target).all()
    assert torch.isfinite(res.precoder).all()
    for v in (res.min_user_rate, res.sum_rate, res.min_sensing_illumination_db):
        assert v == v and abs(v) < float("inf")
    assert isinstance(res.outage_count, int)


# ---------------------------------------------------------------- tau behaviour
def test_tau_zero_estimate_equals_truth():
    evo, cfg, _geno, P, theta, env = _setup()
    res = evaluate_control_interval(env, P, theta, 0, 0.0, cfg, evo, _gen(1))
    assert torch.equal(res.estimated_components.aggregate, res.true_components.aggregate)


def test_precoder_changes_with_csi_noise():
    evo, cfg, _geno, P, theta, env = _setup()
    res0 = evaluate_control_interval(env, P, theta, 0, 0.0, cfg, evo, _gen(1))
    res_hi = evaluate_control_interval(env, P, theta, 0, 0.3, cfg, evo, _gen(1))
    assert not torch.equal(res_hi.precoder, res0.precoder)
    assert torch.isfinite(res_hi.precoder).all()


# ---------------------------------------------------------------- truth vs est
def test_realized_rate_uses_true_channel():
    evo, cfg, _geno, P, theta, env = _setup()
    res = evaluate_control_interval(env, P, theta, 0, 0.25, cfg, evo, _gen(2))
    # rate recomputed from the TRUE aggregate matches the reported rate
    expected = evm_sinr_rate(res.true_components.aggregate, res.precoder, cfg, res.p_user)[0, 0]
    assert torch.allclose(res.user_rates, expected)
    # using the ESTIMATE instead would give a different rate (tau>0)
    est_based = evm_sinr_rate(res.estimated_components.aggregate, res.precoder, cfg, res.p_user)[0, 0]
    assert not torch.allclose(res.user_rates, est_based)


# ---------------------------------------------------------------- power action
def test_power_action_changes_results():
    evo, cfg, _geno, P, theta, env = _setup()
    # identical generator seed -> identical estimate -> identical W; only power differs
    res_a = evaluate_control_interval(env, P, theta, 0, 0.1, cfg, evo, _gen(5))
    res_b = evaluate_control_interval(env, P, theta, 2, 0.1, cfg, evo, _gen(5))
    assert torch.equal(res_a.precoder, res_b.precoder)        # same W
    assert not torch.equal(res_a.p_user, res_b.p_user)         # different power
    assert not torch.allclose(res_a.user_rates, res_b.user_rates)


# ---------------------------------------------------------------- vs evaluator
def test_equal_action_tau0_matches_evaluator():
    evo, cfg, geno, P, theta, env = _setup(regime_index=0, seed=3)
    F = evaluate_single_regime(geno, env, cfg)                 # (1,2) -- power_policy="equal"
    res = evaluate_control_interval(env, P, theta, 0, 0.0, cfg, evo, _gen(9))  # action 0 = equal
    assert abs(res.min_user_rate - float(F[0, 0])) <= 1e-3 * abs(float(F[0, 0])) + 1e-4
    assert abs(res.min_sensing_illumination_db - float(F[0, 1])) <= 1e-2


def test_matches_evaluator_across_regimes():
    for regime_index in range(3):
        evo, cfg, geno, P, theta, env = _setup(regime_index=regime_index, seed=11)
        F = evaluate_single_regime(geno, env, cfg)
        res = evaluate_control_interval(env, P, theta, 0, 0.0, cfg, evo, _gen(0))
        assert abs(res.min_user_rate - float(F[0, 0])) <= 1e-3 * abs(float(F[0, 0])) + 1e-4
        assert abs(res.min_sensing_illumination_db - float(F[0, 1])) <= 1e-2


# ---------------------------------------------------------------- no policy leak
def test_does_not_use_config_power_policy():
    # Even if the physical Config asks for inv_gain, the backend uses the decoded
    # power action, not Config.power_policy.
    evo, cfg, _geno, P, theta, env = _setup()
    import dataclasses
    cfg_inv = dataclasses.replace(cfg, power_policy="inv_gain")
    res_equal_cfg = evaluate_control_interval(env, P, theta, 0, 0.0, cfg, evo, _gen(7))
    res_inv_cfg = evaluate_control_interval(env, P, theta, 0, 0.0, cfg_inv, evo, _gen(7))
    # action index 0 = equal power in both -> identical p_user regardless of power_policy
    assert torch.equal(res_equal_cfg.p_user, res_inv_cfg.p_user)

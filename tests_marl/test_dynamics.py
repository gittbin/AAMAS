"""Tests for temporal channel dynamics (mobility + Gauss-Markov fading) — Task 04."""

from __future__ import annotations

import math
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from momgrap import smoke_config
from momgrap.genotype import decode
from momgrap.marl.config import DynamicsConfig
from momgrap.marl.physics.dynamics import DynamicChannelProcess
from momgrap.marl.types import DynamicPhysicalState


def _dyn(**over) -> DynamicsConfig:
    base = dict(
        episode_length=16,
        dt_seconds=0.02,
        regime_sampling="fixed_los",
        gauss_markov_rho=0.9,
        mobility_enabled=True,
        user_speed_min_mps=0.5,
        user_speed_max_mps=2.0,
        resample_scatter_directions_each_episode=True,
    )
    base.update(over)
    return DynamicsConfig(**base)


def _init_pt(cfg, seed: int = 0):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    geno = torch.rand(1, cfg.D, generator=g, device="cpu", dtype=cfg.dtype_real)
    return decode(geno, cfg)


def _process(dyn=None):
    cfg = smoke_config(device_str="cpu")
    return cfg, DynamicChannelProcess(cfg, dyn or _dyn())


def _rollout(process, cfg, seed, steps):
    P, theta = _init_pt(cfg)
    state = process.reset(seed, P, theta)
    states = [state]
    for _ in range(steps):
        state = process.advance(state)
        states.append(state)
    return states


# ---------------------------------------------------------------- determinism
def test_same_seed_same_trajectory():
    cfg, p1 = _process()
    _cfg2, p2 = _process()
    s1 = _rollout(p1, cfg, seed=5, steps=6)
    s2 = _rollout(p2, cfg, seed=5, steps=6)
    for a, b in zip(s1, s2):
        assert torch.equal(a.environment.user_pos, b.environment.user_pos)
        assert torch.equal(a.environment.g_nlos, b.environment.g_nlos)
        assert torch.equal(a.user_velocity, b.user_velocity)


def test_different_seed_different_trajectory():
    cfg, p1 = _process()
    _cfg2, p2 = _process()
    s1 = _rollout(p1, cfg, seed=0, steps=6)
    s2 = _rollout(p2, cfg, seed=1, steps=6)
    assert not torch.equal(s1[-1].environment.g_nlos, s2[-1].environment.g_nlos)


# ---------------------------------------------------------------- fading corr
def _lag1_corr(rho: float, steps: int = 60) -> float:
    cfg, process = _process(_dyn(gauss_markov_rho=rho))
    states = _rollout(process, cfg, seed=0, steps=steps)
    seq = [s.environment.g_nlos for s in states]
    sims = []
    for a, b in zip(seq[:-1], seq[1:]):
        num = (a.conj() * b).sum().real
        den = a.abs().pow(2).sum().sqrt() * b.abs().pow(2).sum().sqrt()
        sims.append(float(num / den.clamp_min(1e-12)))
    return sum(sims) / len(sims)


def test_high_rho_more_correlated_than_zero():
    corr_hi = _lag1_corr(0.99)
    corr_lo = _lag1_corr(0.0)
    assert corr_hi > 0.9
    assert abs(corr_lo) < 0.3
    assert corr_hi > corr_lo


# ---------------------------------------------------------------- region
def test_users_stay_in_region():
    cfg, process = _process(_dyn(user_speed_min_mps=1.5, user_speed_max_mps=2.0))
    states = _rollout(process, cfg, seed=3, steps=1000)
    hw = math.radians(cfg.user_sector_width_deg) / 2.0
    eps = 1e-3
    for st in states:
        up = st.environment.user_pos[0]           # (K,3)
        r = torch.sqrt(up[:, 0] ** 2 + up[:, 1] ** 2)
        az = torch.atan2(up[:, 1], up[:, 0])
        az_c = math.radians(st.environment.sector_center_deg)
        assert torch.all(r >= cfg.user_radius_min - eps)
        assert torch.all(r <= cfg.user_radius_max + eps)
        assert torch.all(az >= az_c - hw - eps)
        assert torch.all(az <= az_c + hw + eps)
        assert torch.all(up[:, 2].abs() < eps)     # stays in z=0 plane


# ---------------------------------------------------------------- geometry
def test_geometry_fields_consistent():
    cfg, process = _process()
    states = _rollout(process, cfg, seed=7, steps=5)
    for st in states[1:]:                          # check advanced states
        env = st.environment
        up = env.user_pos
        r = torch.linalg.norm(up, dim=-1)
        exp_dir = up / r.clamp_min(1e-6).unsqueeze(-1)
        assert torch.allclose(env.user_dir, exp_dir, atol=1e-5)
        assert torch.allclose(torch.linalg.norm(env.user_dir, dim=-1), torch.ones_like(r), atol=1e-4)

        pl_db = cfg.pl0_db + 10.0 * cfg.pl_exponent * torch.log10(r.clamp_min(1.0))
        exp_amp = torch.sqrt(10.0 ** (-pl_db / 10.0))
        assert torch.allclose(env.user_amp, exp_amp, rtol=1e-4)

        p_ris = torch.tensor(cfg.ris_position, dtype=torch.float64)
        rtu = up.to(torch.float64) - p_ris
        d = torch.linalg.norm(rtu, dim=-1)
        exp_gld = (rtu / d.clamp_min(1e-6).unsqueeze(-1)).to(up.dtype)
        assert torch.allclose(env.g_los_dir, exp_gld, atol=1e-5)
        pl2 = cfg.pl0_db + 10.0 * cfg.pl_exponent * torch.log10(d.clamp_min(1.0))
        exp_gamp = torch.sqrt(10.0 ** (-pl2 / 10.0)).to(up.dtype)
        assert torch.allclose(env.g_amp, exp_gamp, rtol=1e-4)


# ---------------------------------------------------------------- finiteness
def test_thousand_transitions_finite():
    cfg, process = _process()
    states = _rollout(process, cfg, seed=1, steps=1000)
    last = states[-1]
    assert last.step_index == 1000
    env = last.environment
    for t in (env.user_pos, env.user_dir, env.user_amp, env.g_los_dir, env.g_amp,
              env.scat_gain_bu, env.G_nlos, env.g_nlos, last.user_velocity):
        assert torch.isfinite(t).all()


# ---------------------------------------------------------------- immutability
def test_advance_does_not_mutate_previous_state():
    cfg, process = _process()
    P, theta = _init_pt(cfg)
    state = process.reset(0, P, theta)

    snap_pos = state.environment.user_pos.clone()
    snap_gn = state.environment.g_nlos.clone()
    snap_vel = state.user_velocity.clone()
    snap_scatdir = state.environment.scat_dir_bu.clone()
    snap_step = state.step_index
    state.previous_physical_actions["bs_agent"] = 3   # populate to test dict copy

    new = process.advance(state)

    assert new is not state
    assert isinstance(new, DynamicPhysicalState)
    assert new.step_index == snap_step + 1
    # previous state untouched
    assert torch.equal(state.environment.user_pos, snap_pos)
    assert torch.equal(state.environment.g_nlos, snap_gn)
    assert torch.equal(state.user_velocity, snap_vel)
    assert state.step_index == snap_step
    # scatter directions fixed within an episode
    assert torch.equal(new.environment.scat_dir_bu, snap_scatdir)
    # the new state's action dict is a copy, not an alias
    new.previous_physical_actions["bs_agent"] = 9
    assert state.previous_physical_actions["bs_agent"] == 3
    # fading actually evolved
    assert not torch.equal(new.environment.g_nlos, state.environment.g_nlos)


# ---------------------------------------------------------------- mobility off
def test_mobility_disabled_keeps_users_fixed():
    cfg, process = _process(_dyn(mobility_enabled=False))
    states = _rollout(process, cfg, seed=2, steps=10)
    first = states[0].environment.user_pos
    for st in states[1:]:
        assert torch.equal(st.environment.user_pos, first)
        assert torch.all(st.user_velocity == 0)


# ---------------------------------------------------------------- regime select
def test_fixed_regime_selection():
    cfg = smoke_config(device_str="cpu")
    P, theta = _init_pt(cfg)
    for sampling, expected_name in (
        ("fixed_los", "LoS"), ("fixed_rayleigh", "Rayleigh"), ("fixed_rician", "Rician")
    ):
        proc = DynamicChannelProcess(cfg, _dyn(regime_sampling=sampling))
        st = proc.reset(0, P, theta)
        assert cfg.regimes[st.regime_index].name == expected_name

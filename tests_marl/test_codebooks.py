"""Tests for the discrete action codebooks / decoders (Task 05).

Runs on the smoke dimensions M=4, L=16, K=2 (from configs_marl/smoke.yaml).
"""

from __future__ import annotations

import math
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch

from momgrap.genotype import decode
from momgrap.marl.actions.codebooks import (
    build_ris_group_map,
    ma_action_count,
    ris_action_count,
)
from momgrap.marl.actions.decoder import (
    decode_ma_action,
    decode_power_action,
    decode_ris_action,
)
from momgrap.marl.config import load_evocomm_config

SMOKE = _REPO_ROOT / "configs_marl" / "smoke.yaml"


def _cfgs():
    evo = load_evocomm_config(SMOKE)
    return evo.physical, evo.actions   # M=4, L=16, K=2 ; power=7, groups=4, step=30deg


def _synth_hw(cfg, seed: int = 0):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    M, K = cfg.M, cfg.K
    H = torch.complex(
        torch.randn(1, 1, K, M, generator=g), torch.randn(1, 1, K, M, generator=g)
    )
    W = torch.complex(
        torch.randn(1, 1, M, K, generator=g), torch.randn(1, 1, M, K, generator=g)
    )
    return H, W


def _theta(cfg, seed: int = 1):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.rand(1, cfg.L, generator=g) * (2.0 * math.pi)


def _positions(cfg, seed: int = 2):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    geno = torch.rand(1, cfg.D, generator=g, dtype=cfg.dtype_real)
    P, _theta_ = decode(geno, cfg)
    return P


# ---------------------------------------------------------------- power
def test_power_sums_floor_shape():
    cfg, acfg = _cfgs()
    H, W = _synth_hw(cfg)
    p_max = float(cfg.p_max_watt)
    floor = 0.05 * p_max / cfg.K
    for idx in range(acfg.power_codebook_size):
        p = decode_power_action(idx, H, W, cfg, acfg)
        assert p.shape == (1, 1, cfg.K)
        assert torch.isfinite(p).all()
        assert torch.all(p >= 0)
        assert torch.all(p >= floor - 1e-9)
        assert abs(float(p.sum()) - p_max) <= 1e-6 * p_max


def test_power_deterministic():
    cfg, acfg = _cfgs()
    H, W = _synth_hw(cfg)
    assert torch.equal(
        decode_power_action(2, H, W, cfg, acfg), decode_power_action(2, H, W, cfg, acfg)
    )


def test_power_equal_profile_is_uniform():
    cfg, acfg = _cfgs()
    H, W = _synth_hw(cfg)
    p = decode_power_action(0, H, W, cfg, acfg)   # index 0 == equal
    assert torch.allclose(p, torch.full_like(p, float(cfg.p_max_watt) / cfg.K))


def test_power_focus_profile_peaks_on_user():
    cfg, acfg = _cfgs()
    H, W = _synth_hw(cfg)
    # specs: 0 equal,1 inv0.5,2 inv1,3 inv2,4 blend,5 focus0,6 focus1 (K=2)
    p_focus0 = decode_power_action(5, H, W, cfg, acfg)
    p_focus1 = decode_power_action(6, H, W, cfg, acfg)
    assert float(p_focus0[0, 0, 0]) == float(p_focus0.max())
    assert float(p_focus1[0, 0, 1]) == float(p_focus1.max())


def test_power_invalid_index():
    cfg, acfg = _cfgs()
    H, W = _synth_hw(cfg)
    with pytest.raises(ValueError):
        decode_power_action(-1, H, W, cfg, acfg)
    with pytest.raises(ValueError):
        decode_power_action(acfg.power_codebook_size, H, W, cfg, acfg)


def test_power_no_input_mutation():
    cfg, acfg = _cfgs()
    H, W = _synth_hw(cfg)
    H0, W0 = H.clone(), W.clone()
    decode_power_action(3, H, W, cfg, acfg)
    assert torch.equal(H, H0) and torch.equal(W, W0)


# ---------------------------------------------------------------- RIS
def test_ris_noop_exact():
    cfg, acfg = _cfgs()
    gmap = build_ris_group_map(cfg, acfg)
    theta = _theta(cfg)
    tn, changed = decode_ris_action(theta, 0, gmap, math.radians(acfg.ris_phase_step_deg))
    assert changed == 0
    assert torch.equal(tn, theta)


def test_ris_group_action_changes_only_that_group_and_wraps():
    cfg, acfg = _cfgs()
    gmap = build_ris_group_map(cfg, acfg)
    theta = _theta(cfg)
    step = math.radians(acfg.ris_phase_step_deg)
    tn, changed = decode_ris_action(theta, 1, gmap, step)   # +Delta group 0
    assert changed == 1
    mask0 = gmap == 0
    assert not torch.allclose(tn[..., mask0], theta[..., mask0])
    assert torch.equal(tn[..., ~mask0], theta[..., ~mask0])
    assert torch.all((tn >= 0) & (tn < 2.0 * math.pi))


def test_ris_global_action_changes_all():
    cfg, acfg = _cfgs()
    gmap = build_ris_group_map(cfg, acfg)
    G = acfg.ris_group_count
    theta = _theta(cfg)
    tn, changed = decode_ris_action(theta, 2 * G + 1, gmap, math.radians(acfg.ris_phase_step_deg))
    assert changed == G
    assert torch.all((tn >= 0) & (tn < 2.0 * math.pi))


def test_ris_all_valid_indices_and_invalid():
    cfg, acfg = _cfgs()
    gmap = build_ris_group_map(cfg, acfg)
    theta = _theta(cfg)
    step = math.radians(acfg.ris_phase_step_deg)
    n = ris_action_count(acfg)
    assert n == 2 * acfg.ris_group_count + 3
    for idx in range(n):
        tn, _ = decode_ris_action(theta, idx, gmap, step)
        assert tn.shape == theta.shape
        assert torch.isfinite(tn).all()
    with pytest.raises(ValueError):
        decode_ris_action(theta, -1, gmap, step)
    with pytest.raises(ValueError):
        decode_ris_action(theta, n, gmap, step)


def test_ris_deterministic_and_no_mutation():
    cfg, acfg = _cfgs()
    gmap = build_ris_group_map(cfg, acfg)
    theta = _theta(cfg)
    snap = theta.clone()
    step = math.radians(acfg.ris_phase_step_deg)
    a, _ = decode_ris_action(theta, 3, gmap, step)
    b, _ = decode_ris_action(theta, 3, gmap, step)
    assert torch.equal(a, b)
    assert torch.equal(theta, snap)   # input unchanged


# ---------------------------------------------------------------- MA
def test_ma_shape_and_x_plane_preserved():
    cfg, acfg = _cfgs()
    P = _positions(cfg)
    for idx in range(ma_action_count(acfg)):
        Pc = decode_ma_action(P, idx, cfg, acfg)
        assert Pc.shape == (1, cfg.M, 3)
        assert torch.isfinite(Pc).all()
        assert torch.equal(Pc[..., 0], P[..., 0])   # x never moved


def test_ma_noop_exact_and_patterns_differ():
    cfg, acfg = _cfgs()
    P = _positions(cfg)
    assert torch.equal(decode_ma_action(P, 0, cfg, acfg), P)
    for idx in range(1, ma_action_count(acfg)):
        assert not torch.equal(decode_ma_action(P, idx, cfg, acfg), P)


def test_ma_deterministic_no_mutation_and_invalid():
    cfg, acfg = _cfgs()
    P = _positions(cfg)
    snap = P.clone()
    a = decode_ma_action(P, 5, cfg, acfg)
    b = decode_ma_action(P, 5, cfg, acfg)
    assert torch.equal(a, b)
    assert torch.equal(P, snap)
    with pytest.raises(ValueError):
        decode_ma_action(P, -1, cfg, acfg)
    with pytest.raises(ValueError):
        decode_ma_action(P, ma_action_count(acfg), cfg, acfg)

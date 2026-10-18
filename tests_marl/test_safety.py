"""Tests for the safety shield (Task 06)."""

from __future__ import annotations

import math
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch

from momgrap import smoke_config
from momgrap.channels import bs_cell_centers
from momgrap.marl.actions.safety import (
    _min_pairwise_distance,
    _positions_feasible,
    project_to_safety,
)


def _cfg(**over):
    base = dict(
        M=4, L=16, K=2, Q=1,
        bs_grid_rows=2, bs_grid_cols=2,
        ris_grid_rows=4, ris_grid_cols=4,
        device_str="cpu",
    )
    base.update(over)
    return smoke_config(**base)


def _feasible(cfg, P):
    centers = bs_cell_centers(cfg).to(dtype=P.dtype)
    return _positions_feasible(P, centers, cfg, 1e-6 * cfg.d_min, 1e-6)


def _equal_power(cfg):
    return torch.full((1, 1, cfg.K), cfg.p_max_watt / cfg.K)


# ---------------------------------------------------------------- no-op
def test_feasible_input_passes_through_unchanged():
    cfg = _cfg()
    centers = bs_cell_centers(cfg).to(dtype=cfg.dtype_real).unsqueeze(0)  # (1,M,3) USPA
    theta = torch.zeros(1, cfg.L)
    power = _equal_power(cfg)
    out = project_to_safety(centers, theta, power, cfg)

    assert not out.repaired
    assert not out.fallback_used
    assert torch.equal(out.positions, centers)
    assert torch.equal(out.theta, theta)
    assert torch.equal(out.power_alloc, power)
    assert out.boundary_violation_after == 0.0
    assert out.power_violation_after < 1e-9


# ---------------------------------------------------------------- repair works
def test_out_of_bounds_positions_repaired_feasible():
    cfg = _cfg()
    centers = bs_cell_centers(cfg).to(dtype=cfg.dtype_real).unsqueeze(0)
    bad = centers + torch.tensor([0.0, 1.0, 0.0])       # push way outside cells (+1 m in y)
    out = project_to_safety(bad, torch.zeros(1, cfg.L), _equal_power(cfg), cfg)

    assert out.repaired
    assert not out.fallback_used
    assert _feasible(cfg, out.positions)
    assert out.boundary_violation_before > 0.0
    assert out.boundary_violation_after <= 1e-6


# ---------------------------------------------------------------- fallback
def test_fallback_to_previous_when_repair_fails():
    # Overlapping cells (extent 1.5*lambda) let coincident antennas survive repair
    # (repulsion is degenerate at zero separation) -> shield must fall back.
    cfg = _cfg(cell_extent_factor=1.5)
    coincident = torch.zeros(1, cfg.M, 3)               # all antennas at the centroid
    prev = bs_cell_centers(cfg).to(dtype=cfg.dtype_real).unsqueeze(0)  # feasible layout

    out = project_to_safety(
        coincident, torch.zeros(1, cfg.L), _equal_power(cfg), cfg, previous_positions=prev
    )
    assert out.fallback_used
    assert out.repaired
    assert torch.equal(out.positions, prev)
    assert _feasible(cfg, out.positions)
    assert out.min_spacing_before == 0.0                # coincident before


def test_fallback_to_centers_when_no_previous():
    cfg = _cfg(cell_extent_factor=1.5)
    coincident = torch.zeros(1, cfg.M, 3)
    out = project_to_safety(coincident, torch.zeros(1, cfg.L), _equal_power(cfg), cfg)
    assert out.fallback_used
    centers = bs_cell_centers(cfg).to(dtype=cfg.dtype_real).unsqueeze(0)
    assert torch.equal(out.positions, centers)
    assert _feasible(cfg, out.positions)


# ---------------------------------------------------------------- RIS wrap
def test_ris_phases_wrapped_into_range():
    cfg = _cfg()
    centers = bs_cell_centers(cfg).to(dtype=cfg.dtype_real).unsqueeze(0)
    theta = torch.tensor([[10.0, -3.0, 0.0, 100.0] + [1.0] * (cfg.L - 4)])
    out = project_to_safety(centers, theta, _equal_power(cfg), cfg)
    assert torch.all((out.theta >= 0) & (out.theta < 2.0 * math.pi))
    assert torch.isfinite(out.theta).all()


def test_ris_quantization_optional():
    cfg = _cfg()
    centers = bs_cell_centers(cfg).to(dtype=cfg.dtype_real).unsqueeze(0)
    g = torch.Generator().manual_seed(0)
    theta = torch.rand(1, cfg.L, generator=g) * (2.0 * math.pi)
    out = project_to_safety(centers, theta, _equal_power(cfg), cfg, ris_phase_bits=2)
    step_q = 2.0 * math.pi / 4
    ratio = out.theta / step_q
    assert torch.allclose(ratio, ratio.round(), atol=1e-5)
    assert torch.all((out.theta >= 0) & (out.theta < 2.0 * math.pi))


# ---------------------------------------------------------------- power
def test_power_clamped_and_renormalised():
    cfg = _cfg()
    centers = bs_cell_centers(cfg).to(dtype=cfg.dtype_real).unsqueeze(0)
    power = torch.tensor([[[-1.0, 5.0]]])               # negative + wrong sum
    out = project_to_safety(centers, torch.zeros(1, cfg.L), power, cfg)
    assert out.power_alloc.shape == (1, 1, cfg.K)
    assert torch.all(out.power_alloc >= 0)
    assert abs(float(out.power_alloc.sum()) - cfg.p_max_watt) <= 1e-6 * cfg.p_max_watt
    assert out.power_violation_before > 0.0
    assert out.power_violation_after < 1e-9


def test_power_all_zero_becomes_equal():
    cfg = _cfg()
    centers = bs_cell_centers(cfg).to(dtype=cfg.dtype_real).unsqueeze(0)
    power = torch.zeros(1, 1, cfg.K)
    out = project_to_safety(centers, torch.zeros(1, cfg.L), power, cfg)
    assert torch.allclose(out.power_alloc, torch.full_like(power, cfg.p_max_watt / cfg.K))


# ---------------------------------------------------------------- generic
def test_output_finite_and_no_input_mutation():
    cfg = _cfg()
    centers = bs_cell_centers(cfg).to(dtype=cfg.dtype_real).unsqueeze(0)
    pos = centers + torch.tensor([0.0, 0.5, -0.3])
    theta = torch.rand(1, cfg.L) * 20.0 - 10.0
    power = torch.randn(1, 1, cfg.K)

    pos0, theta0, power0 = pos.clone(), theta.clone(), power.clone()
    out = project_to_safety(pos, theta, power, cfg, previous_positions=centers)

    for t in (out.positions, out.theta, out.power_alloc):
        assert torch.isfinite(t).all()
    assert _feasible(cfg, out.positions)
    # inputs untouched
    assert torch.equal(pos, pos0)
    assert torch.equal(theta, theta0)
    assert torch.equal(power, power0)


def test_before_metrics_capture_violation():
    cfg = _cfg()
    centers = bs_cell_centers(cfg).to(dtype=cfg.dtype_real).unsqueeze(0)
    bad = centers + torch.tensor([0.0, 2.0, 0.0])
    out = project_to_safety(bad, torch.zeros(1, cfg.L), _equal_power(cfg), cfg)
    # reward should see a positive pre-repair boundary violation
    assert out.boundary_violation_before > out.boundary_violation_after
    assert out.min_spacing_after >= cfg.d_min - 1e-6 * cfg.d_min

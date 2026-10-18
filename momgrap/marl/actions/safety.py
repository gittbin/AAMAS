"""Safety shield: project raw agent proposals onto feasible physical configs.

Reuses the existing :func:`momgrap.genotype.repair_positions` for MA layout
repair, but **validates the result** — if the fixed number of repair iterations
did not resolve a ``d_min`` / cell-bound violation, it falls back to the last
known-feasible layout (or the USPA cell centres) rather than emitting an
infeasible action. The pre-repair violation magnitudes are recorded so the reward
can penalise bad actions (the penalty must use the *before* values so agents learn
to avoid them). ``cfg.repair_iters`` is never mutated.

Also wraps RIS phases (optional quantisation) and clamps / renormalises power to a
nonnegative allocation summing to ``p_max_watt`` while preserving shape ``(1,1,K)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ...channels import bs_cell_centers
from ...config import Config
from ...genotype import repair_positions

_TWO_PI = 2.0 * math.pi


@dataclass
class SafetyProjection:
    positions: torch.Tensor        # (1,M,3) feasible
    theta: torch.Tensor            # (1,L) wrapped [0,2pi)
    power_alloc: torch.Tensor      # (1,1,K) nonneg, sums to p_max
    min_spacing_before: float
    min_spacing_after: float
    boundary_violation_before: float
    boundary_violation_after: float
    power_violation_before: float
    power_violation_after: float
    repaired: bool
    fallback_used: bool


def _half_extent(cfg: Config) -> float:
    return cfg.cell_extent_factor * cfg.wavelength / 2.0


def _min_pairwise_distance(P: torch.Tensor) -> float:
    """Minimum pairwise antenna distance of ``P`` (1,M,3); inf if M<2."""
    M = P.shape[1]
    if M < 2:
        return float("inf")
    diff = P[:, :, None, :] - P[:, None, :, :]           # (1,M,M,3)
    dist = torch.linalg.norm(diff, dim=-1)               # (1,M,M)
    eye = torch.eye(M, dtype=torch.bool, device=P.device).unsqueeze(0)
    dist = dist.masked_fill(eye, float("inf"))
    return float(dist.min())


def _boundary_overshoot(P: torch.Tensor, centers: torch.Tensor, half: float) -> float:
    """Total metres by which antennas exceed their cell bounds."""
    over_y = (P[..., 1] - centers[:, 1]).abs() - half
    over_z = (P[..., 2] - centers[:, 2]).abs() - half
    return float(over_y.clamp_min(0.0).sum() + over_z.clamp_min(0.0).sum())


def _power_violation(p: torch.Tensor, p_max: float) -> float:
    off_budget = (p.sum() - p_max).abs()
    neg_mass = (-p).clamp_min(0.0).sum()
    return float(off_budget + neg_mass)


def _positions_feasible(P, centers, cfg, spacing_tol, boundary_tol) -> bool:
    return (
        _min_pairwise_distance(P) >= cfg.d_min - spacing_tol
        and _boundary_overshoot(P, centers, _half_extent(cfg)) <= boundary_tol
    )


def project_to_safety(
    positions: torch.Tensor,
    theta: torch.Tensor,
    power_alloc: torch.Tensor,
    cfg: Config,
    previous_positions: torch.Tensor | None = None,
    ris_phase_bits: int | None = None,
) -> SafetyProjection:
    """Return a feasible :class:`SafetyProjection` for a raw (candidate) action.

    ``previous_positions`` (last feasible layout) is used as the MA fallback when
    repair fails; if it is missing or itself infeasible, the USPA cell centres are
    used (always feasible). RIS phases are wrapped (optionally quantised to
    ``ris_phase_bits`` bits) and power is clamped nonnegative and renormalised to
    ``p_max_watt`` (equal power if the total is ~0). Inputs are never mutated.
    """
    device = positions.device
    dtype = positions.dtype
    centers = bs_cell_centers(cfg).to(device=device, dtype=dtype)   # (M,3)
    half = _half_extent(cfg)
    spacing_tol = 1e-6 * cfg.d_min
    boundary_tol = 1e-6

    # ----- before-repair violation magnitudes (reward uses these) -----
    min_spacing_before = _min_pairwise_distance(positions)
    boundary_before = _boundary_overshoot(positions, centers, half)
    power_before = _power_violation(power_alloc, cfg.p_max_watt)

    # ----- MA repair + validate + fallback -----
    P_rep = repair_positions(positions, centers, cfg)
    fallback_used = False
    if _positions_feasible(P_rep, centers, cfg, spacing_tol, boundary_tol):
        positions_out = P_rep
    else:
        cand = None
        if previous_positions is not None:
            prev = previous_positions.to(device=device, dtype=dtype)
            if _positions_feasible(prev, centers, cfg, spacing_tol, boundary_tol):
                cand = prev
        if cand is None:
            cand = centers.unsqueeze(0).clone()        # USPA, always feasible
        positions_out = cand
        fallback_used = True

    # ----- RIS wrap (+ optional quantisation) -----
    theta_out = torch.remainder(theta, _TWO_PI)
    if ris_phase_bits is not None:
        levels = 2 ** int(ris_phase_bits)
        step_q = _TWO_PI / levels
        theta_out = torch.remainder(torch.round(theta_out / step_q) * step_q, _TWO_PI)

    # ----- power clamp + renormalise -----
    p = power_alloc.clamp_min(0.0)
    total = p.sum()
    if float(total) <= 1e-30:
        power_out = torch.full_like(power_alloc, cfg.p_max_watt / cfg.K)
    else:
        power_out = p * (cfg.p_max_watt / total)

    # ----- after metrics -----
    min_spacing_after = _min_pairwise_distance(positions_out)
    boundary_after = _boundary_overshoot(positions_out, centers, half)
    power_after = _power_violation(power_out, cfg.p_max_watt)

    for name, t in (("positions", positions_out), ("theta", theta_out), ("power", power_out)):
        if not torch.isfinite(t).all():
            raise ValueError(f"safety projection produced non-finite {name}")

    repaired = not (
        torch.equal(positions_out, positions)
        and torch.equal(theta_out, theta)
        and torch.equal(power_out, power_alloc)
    )

    return SafetyProjection(
        positions=positions_out,
        theta=theta_out,
        power_alloc=power_out,
        min_spacing_before=min_spacing_before,
        min_spacing_after=min_spacing_after,
        boundary_violation_before=boundary_before,
        boundary_violation_after=boundary_after,
        power_violation_before=power_before,
        power_violation_after=power_after,
        repaired=repaired,
        fallback_used=fallback_used,
    )

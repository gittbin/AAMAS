"""Per-step decoding of discrete agent actions into physical proposals.

* :func:`decode_power_action` — power codebook index -> per-user power ``(1,1,K)``
  summing to ``p_max_watt`` with a maintained minimum floor.
* :func:`decode_ris_action` — grouped incremental phase index -> wrapped ``theta``.
* :func:`decode_ma_action` — layout-pattern index -> candidate MA positions
  ``(1,M,3)`` (an offset proposal; feasibility is enforced later by the safety shield).

None of these mutate their inputs or touch ``Config.power_policy``.
"""

from __future__ import annotations

import math

import torch

from ...channels import bs_cell_centers
from ...config import Config
from ..config import ActionConfig
from .codebooks import (
    MA_PATTERNS,
    ma_action_count,
    power_action_count,
    power_profile_specs,
    ris_action_count,
)

_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------- power
def _enforce_floor(p_raw: torch.Tensor, p_max: float, floor: float) -> torch.Tensor:
    """Water-fill ``p_raw`` so every user gets >= ``floor`` and the sum is p_max."""
    K = p_raw.shape[0]
    budget = p_max - K * floor
    if budget <= 0.0:                                   # floor too high -> equal
        return torch.full_like(p_raw, p_max / K)
    p = p_raw.clamp_min(floor)
    excess = (p - floor).clamp_min(0.0)
    total = excess.sum()
    if float(total) <= 1e-30:
        return torch.full_like(p_raw, p_max / K)
    return floor + excess * (budget / total)


def decode_power_action(
    action_index: int,
    H_est: torch.Tensor,
    W_est: torch.Tensor,
    cfg: Config,
    action_cfg: ActionConfig,
    min_power_fraction: float = 0.05,
) -> torch.Tensor:
    """Return per-user power ``p_user`` ``(1,1,K)`` for a codebook index.

    Profiles are derived from the *estimated* effective channel gains
    ``|h_k^H w_k|^2`` (from ``H_est``/``W_est``). Output is nonnegative, finite,
    sums to ``cfg.p_max_watt`` and respects a minimum floor (``min_power_fraction``
    of the equal share). Never reads/sets ``Config.power_policy``.
    """
    size = power_action_count(action_cfg)
    if not (0 <= action_index < size):
        raise ValueError(f"power action_index={action_index} out of range [0,{size})")

    K = int(cfg.K)
    device = H_est.device
    real_dtype = cfg.dtype_real

    hkwk = torch.einsum("nskm,nsmk->nsk", H_est, W_est)         # (N,S,K)
    gain = (hkwk.abs() ** 2).mean(dim=(0, 1)).clamp_min(1e-30).to(real_dtype)  # (K,)

    spec = power_profile_specs(K, size)[action_index]
    kind = spec[0]
    if kind == "equal":
        w = torch.full((K,), 1.0 / K, dtype=real_dtype, device=device)
    elif kind == "invgain":
        inv = gain.pow(-float(spec[1]))
        w = inv / inv.sum()
    elif kind == "blend":
        eq = torch.full((K,), 1.0 / K, dtype=real_dtype, device=device)
        inv = gain.pow(-1.0)
        w = 0.5 * eq + 0.5 * (inv / inv.sum())
    elif kind == "focus":
        w = torch.zeros(K, dtype=real_dtype, device=device)
        w[int(spec[1])] = 1.0
    else:  # pragma: no cover - specs are closed set
        raise ValueError(f"unknown power profile spec {spec!r}")

    p_max = float(cfg.p_max_watt)
    floor = float(min_power_fraction) * p_max / K
    p = _enforce_floor(w * p_max, p_max, floor)                 # (K,)
    return p.reshape(1, 1, K)


# ---------------------------------------------------------------- RIS
def decode_ris_action(
    current_theta: torch.Tensor,
    action_index: int,
    group_map: torch.Tensor,
    phase_step: float,
) -> tuple[torch.Tensor, int]:
    """Apply a grouped incremental phase action.

    Returns ``(theta_next, n_changed_groups)``. Action 0 is an exact no-op; actions
    ``1..2G`` are ``+/-phase_step`` on a single group; the last two are global
    ``+/-phase_step``. Modified phases are wrapped into ``[0, 2*pi)``. ``phase_step``
    is in radians. The input tensor is never mutated.
    """
    G = int(group_map.max().item()) + 1
    n_actions = 2 * G + 3
    if not (0 <= action_index < n_actions):
        raise ValueError(f"ris action_index={action_index} out of range [0,{n_actions})")

    theta = current_theta.clone()
    if action_index == 0:
        return theta, 0                                        # exact no-op

    if action_index <= 2 * G:
        g = (action_index - 1) // 2
        sign = 1.0 if (action_index - 1) % 2 == 0 else -1.0
        mask = group_map == g                                  # (L,)
        theta[..., mask] = torch.remainder(theta[..., mask] + sign * phase_step, _TWO_PI)
        return theta, 1

    sign = 1.0 if action_index == 2 * G + 1 else -1.0          # global +/-
    theta = torch.remainder(theta + sign * phase_step, _TWO_PI)
    return theta, G


# ---------------------------------------------------------------- MA
def decode_ma_action(
    current_positions: torch.Tensor,
    action_index: int,
    cfg: Config,
    action_cfg: ActionConfig,
) -> torch.Tensor:
    """Return a candidate MA layout ``(1,M,3)`` = current positions + a pattern offset.

    The offset is a small coherent layout deformation (see
    :data:`~momgrap.marl.actions.codebooks.MA_PATTERNS`); the x-component is never
    moved (antennas stay in the y-z plane). Feasibility (spacing, cell bounds) is
    enforced afterwards by the safety shield, not here. The input is not mutated.
    """
    n_actions = ma_action_count(action_cfg)
    if not (0 <= action_index < n_actions):
        raise ValueError(f"ma action_index={action_index} out of range [0,{n_actions})")

    device = current_positions.device
    dtype = current_positions.dtype
    M = int(cfg.M)
    cols = int(cfg.bs_grid_cols)

    centers = bs_cell_centers(cfg).to(device=device, dtype=dtype)   # (M,3)
    uy = torch.sign(centers[:, 1])                                  # side in y
    uz = torch.sign(centers[:, 2])                                  # side in z
    idx = torch.arange(M, device=device)
    parity = (((idx // cols + idx % cols) % 2).to(dtype) * 2.0 - 1.0)  # +/-1 checker

    cell_extent = cfg.cell_extent_factor * cfg.wavelength
    step = float(action_cfg.ma_step_fraction_of_cell) * cell_extent

    d = torch.zeros(M, 3, device=device, dtype=dtype)
    name = MA_PATTERNS[action_index]
    if name == "no_op":
        pass
    elif name == "expand_y":
        d[:, 1] = step * uy
    elif name == "contract_y":
        d[:, 1] = -step * uy
    elif name == "expand_z":
        d[:, 2] = step * uz
    elif name == "contract_z":
        d[:, 2] = -step * uz
    elif name == "positive_shear":
        d[:, 1] = step * uz
    elif name == "negative_shear":
        d[:, 1] = -step * uz
    elif name == "checkerboard_y":
        d[:, 1] = step * parity
    elif name == "checkerboard_z":
        d[:, 2] = step * parity

    return current_positions + d.unsqueeze(0)                      # (1,M,3)

"""Discrete action codebooks for the three heterogeneous agents.

Static definitions only (agent names, RIS grouping, MA layout patterns, power
profile ordering, action-count helpers). The per-step decoding lives in
:mod:`momgrap.marl.actions.decoder`.

The codebooks are deliberately coarse (codebook / grouped / layout-level) so the
MVP action spaces stay small and discrete — no per-RIS-element control, no direct
complex-beamformer synthesis (CLAUDE_CODE_TASKS Sec 3).
"""

from __future__ import annotations

import torch

from ...config import Config
from ..config import ActionConfig

# Canonical agent order — every adjacency / action tensor uses this order.
AGENTS = ("bs_agent", "ris_agent", "ma_agent")

# MA layout-level movement patterns (index -> name). Action 0 is always no-op.
MA_PATTERNS = (
    "no_op",
    "expand_y",
    "contract_y",
    "expand_z",
    "contract_z",
    "positive_shear",
    "negative_shear",
    "checkerboard_y",
    "checkerboard_z",
)


# ---------------------------------------------------------------- action counts
def power_action_count(action_cfg: ActionConfig) -> int:
    return int(action_cfg.power_codebook_size)


def ris_action_count(action_cfg: ActionConfig) -> int:
    """no-op + (+/-Delta per group) + (global +Delta / global -Delta)."""
    return 2 * int(action_cfg.ris_group_count) + 3


def ma_action_count(action_cfg: ActionConfig | None = None) -> int:
    return len(MA_PATTERNS)


# ---------------------------------------------------------------- power profiles
def power_profile_specs(K: int, size: int) -> list[tuple]:
    """Deterministic ordered list of ``size`` power-profile specs.

    Order: equal, inverse-gain (alpha=0.5, 1.0, 2.0), equal/fairness blend, then
    one focus-user profile per user; if ``size`` exceeds that, extra higher-alpha
    inverse-gain profiles are appended so any codebook size is well defined.
    """
    specs: list[tuple] = [
        ("equal",),
        ("invgain", 0.5),
        ("invgain", 1.0),
        ("invgain", 2.0),
        ("blend",),
    ]
    for j in range(K):
        specs.append(("focus", j))
    alpha = 3.0
    while len(specs) < size:
        specs.append(("invgain", alpha))
        alpha += 1.0
    return specs[:size]


# ---------------------------------------------------------------- RIS grouping
def build_ris_group_map(
    cfg: Config, action_cfg: ActionConfig, device: torch.device | None = None
) -> torch.Tensor:
    """Contiguous element->group id map ``(L,)`` long, values in ``[0, G)``."""
    L = int(cfg.L)
    G = int(action_cfg.ris_group_count)
    if G <= 0 or L % G != 0:
        raise ValueError(f"L={L} must be divisible by ris_group_count={G}")
    per = L // G
    gmap = torch.arange(L, dtype=torch.long) // per
    if device is not None:
        gmap = gmap.to(device)
    return gmap

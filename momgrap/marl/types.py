"""Shared MARL data types.

Kept dependency-light (only torch + the momgrap Environment) so any submodule can
import these without pulling in gymnasium / pettingzoo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..channels import Environment


@dataclass
class DynamicPhysicalState:
    """Full physical state of one MARL episode at a single control step (S=1).

    Fields::

        step_index                 int, 0 at reset, +1 each advance
        regime_index               index into cfg.regimes (fixed within an episode)
        environment                momgrap Environment snapshot (S=1), evolves in time
        user_velocity              (K,3) per-user Cartesian velocity (m/s), z=0 plane
        positions                  (1,M,3) movable-antenna layout being controlled
        theta                      (1,L) RIS phases being controlled
        power_alloc                (1,1,K) current per-user power allocation (W)
        previous_physical_actions  {agent_name: last physical action index}
    """

    step_index: int
    regime_index: int
    environment: Environment
    user_velocity: torch.Tensor
    positions: torch.Tensor
    theta: torch.Tensor
    power_alloc: torch.Tensor
    previous_physical_actions: dict = field(default_factory=dict)

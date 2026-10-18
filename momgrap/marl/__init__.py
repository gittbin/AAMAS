"""EvoComm-MOMGRAP: Multi-Agent Learning + Evolutionary Communication extension.

Additive subpackage layered on top of ``momgrap`` — it reuses the existing
physics, GPU kernels, MFEA, NSGA-II and robust archive without modifying them.
Only the lightweight configuration layer is exported at package level; heavier
submodules (physics, envs, algorithms, evolution) import their own optional
dependencies (gymnasium, pettingzoo, tensorboard) lazily and are imported directly.
"""

from __future__ import annotations

from .config import (  # noqa: F401
    ActionConfig,
    CommunicationConfig,
    CSIConfig,
    DynamicsConfig,
    EvoCommConfig,
    RewardConfig,
    RMAPPOConfig,
    evocomm_config_to_dict,
    load_evocomm_config,
    save_resolved_config,
)

__all__ = [
    "DynamicsConfig",
    "CSIConfig",
    "ActionConfig",
    "CommunicationConfig",
    "RewardConfig",
    "RMAPPOConfig",
    "EvoCommConfig",
    "load_evocomm_config",
    "evocomm_config_to_dict",
    "save_resolved_config",
]

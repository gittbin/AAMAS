"""Physics adapters for the MARL layer — thin reuse of the momgrap core.

Nothing here re-implements the physics; the modules expose the existing momgrap
channel / comm / sensing building blocks under the ``momgrap.marl.physics``
namespace so the environment and observation code has a single import surface.
"""

from __future__ import annotations

from ..types import DynamicPhysicalState  # noqa: F401
from .components import CommChannelComponents, build_comm_channel_components  # noqa: F401
from .csi import (  # noqa: F401
    CSIHistoryBuffer,
    EstimatedChannels,
    apply_complex_csi_error,
    estimate_channel_components,
)
from .backend import PhysicalStepResult, evaluate_control_interval  # noqa: F401
from .dynamics import DynamicChannelProcess  # noqa: F401
from .initialization import (  # noqa: F401
    RAMP_MODES,
    RampInitialization,
    build_ramp_initializations,
    check_compatibility,
    initial_config_from,
    load_initialization,
    save_initialization,
)

__all__ = [
    "CommChannelComponents",
    "build_comm_channel_components",
    "EstimatedChannels",
    "apply_complex_csi_error",
    "estimate_channel_components",
    "CSIHistoryBuffer",
    "DynamicPhysicalState",
    "DynamicChannelProcess",
    "PhysicalStepResult",
    "evaluate_control_interval",
    "RampInitialization",
    "RAMP_MODES",
    "build_ramp_initializations",
    "save_initialization",
    "load_initialization",
    "initial_config_from",
    "check_compatibility",
]

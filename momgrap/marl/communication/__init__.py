"""Inter-agent communication: graph presets, semantic messages, budgeted routing."""

from __future__ import annotations

from .budget import DeliveryLog, MessageRouter, SendRequest  # noqa: F401
from .graph import (  # noqa: F401
    AGENT_NAMES,
    GRAPH_PRESETS,
    N_AGENTS,
    action_is_valid,
    build_adjacency,
    destination_from_action,
    valid_action_mask,
    validate_adjacency,
)
from .messages import (  # noqa: F401
    MessageContext,
    build_agent_messages,
    bs_message,
    ma_message,
    message_priority,
    ris_message,
)
from .modes import (  # noqa: F401
    COMM_MODES,
    TRAIN_COMM_MODES,
    CommunicationCurriculum,
    comm_mask_for_mode,
    forced_comm_action,
)

__all__ = [
    "AGENT_NAMES",
    "N_AGENTS",
    "GRAPH_PRESETS",
    "build_adjacency",
    "validate_adjacency",
    "destination_from_action",
    "action_is_valid",
    "valid_action_mask",
    "MessageContext",
    "build_agent_messages",
    "bs_message",
    "ris_message",
    "ma_message",
    "message_priority",
    "SendRequest",
    "DeliveryLog",
    "MessageRouter",
    "COMM_MODES",
    "TRAIN_COMM_MODES",
    "CommunicationCurriculum",
    "comm_mask_for_mode",
    "forced_comm_action",
]

"""Inter-agent communication graph + destination action semantics.

Adjacency is a directed ``(3,3)`` boolean matrix over the canonical agent order
``AGENT_NAMES`` (``adj[i,j]`` = agent ``i`` may send to agent ``j``); the diagonal
is always zero (no self-messages). Each agent's discrete communication action is
``0=no-send``, ``1=first other agent``, ``2=second other agent`` (others taken in
ascending index order, excluding self). ``no-send`` is always valid; a destination
action is valid only if the graph permits that edge.
"""

from __future__ import annotations

import torch

from ..actions.codebooks import AGENTS as AGENT_NAMES  # single source of truth for order

N_AGENTS = len(AGENT_NAMES)
GRAPH_PRESETS = ("none", "full", "ring", "star_bs", "custom")


def _others(agent_index: int) -> list[int]:
    return [j for j in range(N_AGENTS) if j != agent_index]


def validate_adjacency(adj: torch.Tensor) -> None:
    if tuple(adj.shape) != (N_AGENTS, N_AGENTS):
        raise ValueError(f"adjacency must be {(N_AGENTS, N_AGENTS)}, got {tuple(adj.shape)}")
    if adj.dtype != torch.bool:
        raise ValueError(f"adjacency must be bool, got {adj.dtype}")
    if bool(adj.diagonal().any()):
        raise ValueError("adjacency diagonal must be zero (no self-loops)")


def build_adjacency(
    graph_type: str,
    custom: torch.Tensor | list | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a directed ``(3,3)`` bool adjacency for a preset (or a custom matrix)."""
    A = torch.zeros(N_AGENTS, N_AGENTS, dtype=torch.bool)
    if graph_type == "none":
        pass
    elif graph_type == "full":
        A = ~torch.eye(N_AGENTS, dtype=torch.bool)
    elif graph_type == "ring":
        A[0, 1] = A[1, 2] = A[2, 0] = True                # bs -> ris -> ma -> bs
    elif graph_type == "star_bs":
        A[0, 1] = A[0, 2] = A[1, 0] = A[2, 0] = True      # bs <-> ris, bs <-> ma
    elif graph_type == "custom":
        if custom is None:
            raise ValueError("graph_type='custom' requires a `custom` adjacency")
        A = torch.as_tensor(custom, dtype=torch.bool)
    else:
        raise ValueError(f"unknown graph_type {graph_type!r}; expected {GRAPH_PRESETS}")
    validate_adjacency(A)
    if device is not None:
        A = A.to(device)
    return A


def destination_from_action(agent_index: int, action: int) -> int | None:
    """Map a discrete comm action to a destination agent index (or None for no-send)."""
    if action == 0:
        return None
    if action in (1, 2):
        return _others(agent_index)[action - 1]
    raise ValueError(f"comm action {action} out of range [0,2]")


def action_is_valid(agent_index: int, action: int, adj: torch.Tensor) -> bool:
    dest = destination_from_action(agent_index, action)
    if dest is None:
        return True                                        # no-send always valid
    return bool(adj[agent_index, dest])


def valid_action_mask(agent_index: int, adj: torch.Tensor) -> torch.Tensor:
    """Bool mask over the 3 comm actions {no-send, other0, other1} for one agent."""
    return torch.tensor(
        [action_is_valid(agent_index, a, adj) for a in range(3)], dtype=torch.bool
    )

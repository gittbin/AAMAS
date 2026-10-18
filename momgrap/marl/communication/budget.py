"""Message routing under a graph mask + per-step budget, with delivery delay.

Timing (MVP): a message *selected* at step ``t`` becomes *available* in the
recipient's observation at step ``t + delivery_delay_steps`` (must be >= 1, i.e.
never same-step — that would leak the sender's current action). When more messages
are requested than the budget allows, requests are filtered by the graph, ranked by
priority (uncertainty + urgency) with a deterministic sender-index tie-break, and
the top ``max_messages_per_step`` are delivered.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .graph import validate_adjacency


@dataclass
class SendRequest:
    sender: int                      # agent index
    dest: int | None                 # destination index, or None for no-send
    message: torch.Tensor
    priority: float = 0.0


@dataclass
class DeliveryLog:
    requested: int                   # send requests (dest not None)
    graph_rejected: int              # rejected because the graph forbids the edge
    budget_dropped: int              # dropped because over the per-step budget
    delivered: int

    @property
    def eligible(self) -> int:
        return self.requested - self.graph_rejected


@dataclass(frozen=True)
class ReceivedMessage:
    sender: int
    payload: torch.Tensor


class MessageRouter:
    """Applies graph + budget filtering and delays delivery by ``delivery_delay_steps``."""

    def __init__(self, adjacency: torch.Tensor, max_messages_per_step: int, delivery_delay_steps: int):
        validate_adjacency(adjacency)
        if max_messages_per_step < 0:
            raise ValueError("max_messages_per_step must be >= 0")
        if delivery_delay_steps < 1:
            raise ValueError("delivery_delay_steps must be >= 1 (no same-step delivery)")
        self.adj = adjacency
        self.budget = int(max_messages_per_step)
        self.delay = int(delivery_delay_steps)
        self._pending: dict[int, list[tuple[int, ReceivedMessage]]] = {}

    def reset(self) -> None:
        self._pending = {}

    def submit(self, step: int, requests: list[SendRequest]) -> DeliveryLog:
        """Filter + rank + queue; return a per-step delivery log."""
        requested = [r for r in requests if r.dest is not None]
        valid = [r for r in requested if bool(self.adj[r.sender, r.dest])]
        graph_rejected = len(requested) - len(valid)

        # rank by priority (desc), deterministic tie-break by sender index (asc)
        ranked = sorted(valid, key=lambda r: (-float(r.priority), int(r.sender)))
        delivered = ranked[: self.budget]
        budget_dropped = len(valid) - len(delivered)

        bucket = self._pending.setdefault(step + self.delay, [])
        for r in delivered:
            bucket.append((int(r.dest), ReceivedMessage(int(r.sender), r.message.clone())))

        return DeliveryLog(
            requested=len(requested),
            graph_rejected=graph_rejected,
            budget_dropped=budget_dropped,
            delivered=len(delivered),
        )

    def messages_for_step(self, step: int) -> dict[int, list[torch.Tensor]]:
        """Legacy payload-only interface. New observations use deliveries_for_step."""
        return {dest: [m.payload for m in msgs] for dest, msgs in self.deliveries_for_step(step).items()}

    def deliveries_for_step(self, step: int) -> dict[int, list[ReceivedMessage]]:
        """Pop messages with sender metadata, preserving zero-payload availability."""
        items = self._pending.pop(step, [])
        out: dict[int, list[ReceivedMessage]] = {}
        for dest, msg in items:
            out.setdefault(dest, []).append(msg)
        return out

    def pending_count(self) -> int:
        return sum(len(v) for v in self._pending.values())

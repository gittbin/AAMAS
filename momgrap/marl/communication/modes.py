"""Forced communication modes + message-cost curriculum.

Four modes (CLAUDE_CODE_TASKS Sec 21):
  none    — never send;
  full    — always send (to the first permitted other);
  random  — send to a random destination (eval ablation only);
  learned — the actor's communication head decides.

Two application points:
* **Training** uses :func:`comm_mask_for_mode` to constrain the actor's comm head so
  the sampled action stays consistent with what the env executes (PPO-correct):
  ``none`` masks everything but no-send; ``full`` masks off no-send; ``learned`` uses
  the graph mask. (``random`` is not a training mode.)
* **Evaluation** uses :func:`forced_comm_action` at the env to override whatever the
  policy chose, so the same policy can be measured under every mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .graph import valid_action_mask

COMM_MODES = ("none", "full", "random", "learned")
TRAIN_COMM_MODES = ("none", "full", "learned")


def comm_mask_for_mode(agent_index: int, adjacency: torch.Tensor, mode: str) -> torch.Tensor:
    """Return a (3,) bool comm-action mask enforcing ``mode`` (training-consistent)."""
    vm = valid_action_mask(agent_index, adjacency)          # [no-send, other0, other1]
    if mode == "learned":
        return vm
    if mode == "none":
        return torch.tensor([True, False, False])
    if mode == "full":
        has_dest = bool(vm[1]) or bool(vm[2])
        return torch.tensor([not has_dest, bool(vm[1]), bool(vm[2])])
    raise ValueError(f"comm_mask_for_mode supports {TRAIN_COMM_MODES}, got {mode!r}")


def forced_comm_action(agent_index: int, actor_comm: int, mode: str, rand_int: int = 0) -> int:
    """Override an actor's chosen comm action according to ``mode`` (eval forcing)."""
    if mode == "learned":
        return int(actor_comm)
    if mode == "none":
        return 0
    if mode == "full":
        return 1                                            # first other agent
    if mode == "random":
        return int(rand_int) % 3
    raise ValueError(f"unknown comm mode {mode!r}; expected {COMM_MODES}")


@dataclass
class CommunicationCurriculum:
    """Anneal the delivered-message cost from ``start_cost`` to ``final_cost``.

    Avoids collapse without an arbitrary bonus: early training is cheap to communicate,
    then the cost ramps to its final value over ``warmup_updates`` updates.
    """

    enabled: bool = False
    warmup_updates: int = 0
    start_cost: float = 0.0
    final_cost: float = 0.0

    def cost_at(self, update_index: int) -> float | None:
        """Effective message cost at ``update_index`` (None = use the config default)."""
        if not self.enabled:
            return None
        if self.warmup_updates <= 0 or update_index >= self.warmup_updates:
            return self.final_cost
        frac = update_index / self.warmup_updates
        return self.start_cost + frac * (self.final_cost - self.start_cost)

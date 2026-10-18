"""Communication metrics + the none/full/random/learned ablation (Task 16)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..communication.modes import COMM_MODES
from .rollout import run_episode


@dataclass
class CommStats:
    send_request_rate: float       # requested messages per step
    delivery_rate: float           # delivered / requested
    drop_rate: float               # (budget-dropped + graph-rejected) / requested
    graph_reject_rate: float       # graph-rejected / requested
    messages_per_step: float       # delivered per step
    episode_return: float
    length: int

    def as_dict(self) -> dict:
        return asdict(self)


def comm_stats_from_steps(steps: list[dict], episode_return: float) -> CommStats:
    req = sum(s["message_requested"] for s in steps)
    deliv = sum(s["message_delivered"] for s in steps)
    dropped = sum(s["message_dropped"] for s in steps)
    grej = sum(s["graph_rejected"] for s in steps)
    n = max(1, len(steps))
    denom = max(1, req)
    return CommStats(
        send_request_rate=req / n,
        delivery_rate=deliv / denom,
        drop_rate=(dropped + grej) / denom,
        graph_reject_rate=grej / denom,
        messages_per_step=deliv / n,
        episode_return=episode_return,
        length=len(steps),
    )


def _mean_stats(stats: list[CommStats]) -> dict:
    keys = ("send_request_rate", "delivery_rate", "drop_rate", "graph_reject_rate",
            "messages_per_step", "episode_return")
    return {k: float(np.mean([getattr(s, k) for s in stats])) for k in keys}


def evaluate_comm_modes(make_env, make_policy, seeds, modes=COMM_MODES) -> dict:
    """Run ``make_policy`` under each forced comm mode and report comm metrics + utility.

    ``make_env(mode)`` must return an env with that communication mode forced.
    ``message_utility`` = return(mode) - return('none'); ``performance_per_delivered_message``
    normalises that by the delivered-message count.
    """
    out: dict = {}
    for mode in modes:
        env = make_env(mode)
        stats = []
        for seed in seeds:
            policy = make_policy()
            result, steps = run_episode(env, policy, seed)
            stats.append(comm_stats_from_steps(steps, result.episode_return))
        out[mode] = _mean_stats(stats)

    base = out.get("none", {}).get("episode_return", 0.0)
    for m in out.values():
        m["message_utility"] = m["episode_return"] - base
        delivered = m["messages_per_step"]
        m["performance_per_delivered_message"] = (
            m["message_utility"] / delivered if delivered > 1e-9 else 0.0
        )
    return out

"""Rollout + metrics for MARL policy evaluation."""

from __future__ import annotations

from .comm_metrics import CommStats, comm_stats_from_steps, evaluate_comm_modes  # noqa: F401
from .metrics import EpisodeResult, collect_metadata, summarize_episodes  # noqa: F401
from .rollout import run_episode, run_evaluation  # noqa: F401
from .statistics import (  # noqa: F401
    cohens_d_paired,
    epsilon_indicator,
    holm_correction,
    mean_ci,
    paired_t_test,
    pareto_coverage,
    pareto_front,
)

__all__ = [
    "EpisodeResult",
    "summarize_episodes",
    "collect_metadata",
    "run_episode",
    "run_evaluation",
    "CommStats",
    "comm_stats_from_steps",
    "evaluate_comm_modes",
    "mean_ci",
    "paired_t_test",
    "cohens_d_paired",
    "holm_correction",
    "pareto_front",
    "pareto_coverage",
    "epsilon_indicator",
]

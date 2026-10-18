"""Deterministic episode / multi-seed rollout of a policy in the MARL env."""

from __future__ import annotations

import numpy as np

from .metrics import EpisodeResult


def run_episode(env, policy, seed: int) -> tuple[EpisodeResult, list[dict]]:
    """Run one full episode; return (EpisodeResult, per-step records)."""
    policy.reset()
    obs, _infos = env.reset(seed=seed)
    K = env.cfg.K

    steps: list[dict] = []
    episode_return = 0.0
    while env.agents:
        actions = policy.act(obs, env)
        obs, rewards, _term, _trunc, infos = env.step(actions)
        info = infos["bs_agent"]
        reward = float(rewards["bs_agent"])
        episode_return += reward
        steps.append({
            "step": len(steps),
            "reward": reward,
            "regime": info["regime"],
            "tau": info["tau"],
            "min_user_rate": info["min_user_rate"],
            "sum_rate": info["sum_rate"],
            "min_sensing_illumination_db": info["min_sensing_illumination_db"],
            "outage_count": info["outage_count"],
            "message_requested": info["message_requested"],
            "message_delivered": info["message_delivered"],
            "message_dropped": info["message_dropped"],
            "graph_rejected": info["graph_rejected"],
            "reconfiguration_cost": info["reconfiguration_cost"],
            "violation_before": info["violation_before"],
            "violation_after": info["violation_after"],
        })

    min_rates = np.array([s["min_user_rate"] for s in steps], dtype=np.float64)
    result = EpisodeResult(
        episode_return=episode_return,
        min_rate_mean=float(min_rates.mean()),
        min_rate_p10=float(np.percentile(min_rates, 10)),
        sum_rate_mean=float(np.mean([s["sum_rate"] for s in steps])),
        sensing_mean_db=float(np.mean([s["min_sensing_illumination_db"] for s in steps])),
        outage_rate=float(np.mean([s["outage_count"] for s in steps]) / max(1, K)),
        messages_per_step=float(np.mean([s["message_delivered"] for s in steps])),
        reconfiguration_per_step=float(np.mean([s["reconfiguration_cost"] for s in steps])),
        violation_rate=float(np.mean([1.0 if s["violation_before"] > 1e-9 else 0.0 for s in steps])),
        regime=steps[-1]["regime"] if steps else "n/a",
        tau=float(np.mean([s["tau"] for s in steps])) if steps else 0.0,
        length=len(steps),
    )
    return result, steps


def run_evaluation(env, policy, seeds: list[int]) -> tuple[list[EpisodeResult], list[dict]]:
    """Run one episode per seed; return (results, all step records tagged by seed/episode)."""
    results: list[EpisodeResult] = []
    all_steps: list[dict] = []
    for ep, seed in enumerate(seeds):
        result, steps = run_episode(env, policy, seed)
        results.append(result)
        for s in steps:
            all_steps.append({"episode": ep, "seed": seed, **s})
    return results, all_steps

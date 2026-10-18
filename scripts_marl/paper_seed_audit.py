"""Audit the episode-seed schedule used by the frozen paper-v2 trainers."""
from pathlib import Path
import json


def training_scenario_ids(status):
    config = status["config"]
    n = config["rmappo"]["num_envs"]
    horizon = config["dynamics"]["episode_length"]
    steps = status["actual_transitions"] // n
    base = status["training_seed"] * 1000
    ids = set(range(base, base + n))
    # Both v2 trainers reset before incrementing global_step by n.
    ids.update(base + (e * horizon - 1) * n + i + 1
               for e in range(1, steps // horizon + 1) for i in range(n))
    return ids


def audit_seed_split(campaign, evaluation_ids, development_ids=(10000, 10001, 10002)):
    overlaps, maximum = {}, -1
    for path in sorted((Path(campaign) / "train").glob("*/seed_*/status.json")):
        status = json.loads(path.read_text(encoding="utf-8"))
        if not status.get("complete"):
            continue
        ids = training_scenario_ids(status)
        maximum = max(maximum, max(ids))
        final = sorted(ids & set(evaluation_ids))
        dev = sorted(ids & set(development_ids))
        if final or dev:
            overlaps[f"{status['condition']}/seed_{status['training_seed']}"] = {
                "evaluation_overlap": final, "development_overlap": dev}
    return {"schedule": "frozen-v2: initial s*1000+i; resets s*1000+i+global_step+1 before increment",
            "includes_reset_at_final_episode_boundary": True,
            "evaluation_disjoint_from_training": not any(x["evaluation_overlap"] for x in overlaps.values()),
            "development_disjoint_from_training": not any(x["development_overlap"] for x in overlaps.values()),
            "maximum_training_scenario_id": maximum, "overlaps": overlaps}


SEED_WARNING = (
    "Seed audit: evaluation scenario 40000 occurred during training for policies 0 and 4; "
    "development scenario 10000 occurred for policy 2. The original 50-scenario results "
    "are diagnostic comparisons, not a fully unseen final test. Excluding 40000 is a "
    "sensitivity check; select improvements on new development scenarios and use an untouched final set."
)

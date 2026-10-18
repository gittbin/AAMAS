"""Raw objective vector + explicit reward scalarisation.

The scalar cooperative reward is a fixed, explicit linear combination of
config-normalised terms (CLAUDE_CODE_TASKS Sec 15):

    r = w1 R~min + w2 R~sum + w3 B~min - w4 C_outage - w5 C_msg - w6 C_reconfig - w7 C_viol

All normalisation constants come from ``RewardConfig`` — never from statistics of
future evaluation episodes. Deterministic given the objective vector + config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ObjectiveVector:
    """Raw per-step objective quantities (before any reward shaping)."""

    min_user_rate: float
    sum_rate: float
    min_sensing_illumination_db: float
    outage_count: int
    delivered_messages: int
    reconfiguration_cost: float
    constraint_violation_before: float

    def as_dict(self) -> dict:
        return asdict(self)


def normalized_components(obj: ObjectiveVector, reward_cfg) -> dict:
    """Config-normalised reward terms (all finite)."""
    return {
        "r_min": obj.min_user_rate / reward_cfg.rate_scale,
        "r_sum": obj.sum_rate / reward_cfg.rate_scale,
        "b_min": (obj.min_sensing_illumination_db - reward_cfg.sensing_offset_db)
        / reward_cfg.sensing_scale_db,
        "c_outage": float(obj.outage_count),
        "c_msg": float(obj.delivered_messages),
        "c_reconfig": float(obj.reconfiguration_cost),
        "c_viol": float(obj.constraint_violation_before),
    }


def compute_reward(obj: ObjectiveVector, reward_cfg) -> tuple[float, dict]:
    """Return ``(scalar_reward, normalized_components)``."""
    c = normalized_components(obj, reward_cfg)
    r = (
        reward_cfg.w_min_rate * c["r_min"]
        + reward_cfg.w_sum_rate * c["r_sum"]
        + reward_cfg.w_sensing * c["b_min"]
        - reward_cfg.w_outage * c["c_outage"]
        - reward_cfg.w_message * c["c_msg"]
        - reward_cfg.w_reconfiguration * c["c_reconfig"]
        - reward_cfg.w_violation * c["c_viol"]
    )
    return float(r), c

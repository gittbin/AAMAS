"""Assemble per-agent actor observations and the centralized critic state.

Design contract (no leak): :func:`build_actor_observations` has **no** true-channel
parameter — actors only ever see the *delayed estimated* components, their own
previous action, delivered messages, tau (if exposed) and episode time. The
centralized critic state (:func:`build_centralized_state`) may use the true channel,
full geometry, all previous actions and the regime; it is never fed to an actor.

All observation sizes are fixed given the config (:func:`observation_sizes` /
:func:`centralized_state_size`), so the environment spaces and networks are static.
"""

from __future__ import annotations

import torch

from ..actions.codebooks import ma_action_count, power_action_count, ris_action_count
from ..normalization import FeatureProjection
from .features import (
    complex_to_features,
    grouped_phase_sincos,
    inbox_features,
    min_pairwise_distance,
    onehot,
)

COMPLEX_PROJ_DIM = 8
_LOG_MAG_STATS = 4
_MA_OFFSET_COORDS = 2   # (dy, dz) per antenna
_EPS = 1e-30


def _complex_feat_size(K: int, proj_dim: int) -> int:
    return _LOG_MAG_STATS + K + proj_dim


def _half_extent(cfg) -> float:
    return cfg.cell_extent_factor * cfg.wavelength / 2.0


def _step_frac(step_index: int, episode_length: int, device, dtype) -> torch.Tensor:
    return torch.tensor([min(1.0, step_index / max(1, episode_length))], device=device, dtype=dtype)


def observation_sizes(evo_cfg, proj_dim: int = COMPLEX_PROJ_DIM) -> dict[str, int]:
    """Fixed actor observation dimensionality per agent."""
    cfg = evo_cfg.physical
    K, M = cfg.K, cfg.M
    G = evo_cfg.actions.ris_group_count
    pcs = power_action_count(evo_cfg.actions)
    ris_ac = ris_action_count(evo_cfg.actions)
    ma_ac = ma_action_count(evo_cfg.actions)
    budget = evo_cfg.communication.max_messages_per_step
    mdim = evo_cfg.communication.message_dim
    tau_flag = 1 if evo_cfg.csi.expose_tau_to_actor else 0
    msg = budget * (mdim + 4)
    cf = _complex_feat_size(K, proj_dim)
    return {
        "bs_agent": cf + pcs + msg + tau_flag + 1,
        "ris_agent": cf + 1 + 2 * G + ris_ac + msg + tau_flag + 1,
        "ma_agent": cf + _MA_OFFSET_COORDS * M + 1 + 1 + ma_ac + msg + tau_flag + 1,
    }


def centralized_state_size(evo_cfg, proj_dim: int = COMPLEX_PROJ_DIM) -> int:
    """Fixed centralized critic state dimensionality."""
    cfg = evo_cfg.physical
    K, M, T = cfg.K, cfg.M, cfg.T
    G = evo_cfg.actions.ris_group_count
    pcs = power_action_count(evo_cfg.actions)
    ris_ac = ris_action_count(evo_cfg.actions)
    ma_ac = ma_action_count(evo_cfg.actions)
    cf = _complex_feat_size(K, proj_dim)
    return 3 * cf + 2 * G + _MA_OFFSET_COORDS * M + 1 + pcs + ris_ac + ma_ac + 1 + T + 1


def build_actor_observations(
    delayed_estimated,
    theta: torch.Tensor,
    positions: torch.Tensor,
    centers: torch.Tensor,
    group_map: torch.Tensor,
    prev_actions: dict,
    delivered_messages: dict,
    tau: float,
    step_index: int,
    evo_cfg,
    projection: FeatureProjection | None = None,
    ma_update_countdown: int = 0,
) -> dict[str, torch.Tensor]:
    """Per-agent actor observations from *delayed estimated* CSI + local state.

    ``delayed_estimated`` is an ``EstimatedChannels`` (never the truth).
    ``delivered_messages`` is ``{agent_index: [msg, ...]}`` (0=bs, 1=ris, 2=ma).
    """
    cfg = evo_cfg.physical
    device = positions.device
    dtype = cfg.dtype_real
    G = evo_cfg.actions.ris_group_count
    budget = evo_cfg.communication.max_messages_per_step
    mdim = evo_cfg.communication.message_dim
    expose_tau = evo_cfg.csi.expose_tau_to_actor
    ep_len = evo_cfg.dynamics.episode_length
    half = _half_extent(cfg)

    step_frac = _step_frac(step_index, ep_len, device, dtype)
    tau_feat = (
        torch.tensor([float(tau)], device=device, dtype=dtype)
        if expose_tau else torch.zeros(0, device=device, dtype=dtype)
    )
    est = delayed_estimated

    bs = torch.cat([
        complex_to_features(est.aggregate[0, 0], projection=projection),
        onehot(prev_actions.get("bs_agent"), power_action_count(evo_cfg.actions), device, dtype),
        inbox_features(delivered_messages.get(0, []), budget, mdim, device, dtype),
        tau_feat, step_frac,
    ])

    rp = est.reflected.abs().pow(2).mean()
    dp = est.direct.abs().pow(2).mean().clamp_min(_EPS)
    refl_ratio = torch.log10((rp / dp).clamp_min(_EPS)).reshape(1).to(dtype)
    ris = torch.cat([
        complex_to_features(est.reflected[0, 0], projection=projection),
        refl_ratio,
        grouped_phase_sincos(theta, group_map, G).to(dtype),
        onehot(prev_actions.get("ris_agent"), ris_action_count(evo_cfg.actions), device, dtype),
        inbox_features(delivered_messages.get(1, []), budget, mdim, device, dtype),
        tau_feat, step_frac,
    ])

    offsets = ((positions[0] - centers)[:, 1:] / max(half, _EPS)).reshape(-1).to(dtype)  # (2M,)
    margin = torch.tensor(
        [(min_pairwise_distance(positions) - cfg.d_min) / max(cfg.d_min, _EPS)],
        device=device, dtype=dtype,
    )
    countdown = torch.tensor(
        [ma_update_countdown / max(1, evo_cfg.actions.ma_update_period)], device=device, dtype=dtype
    )
    ma = torch.cat([
        complex_to_features(est.direct[0, 0], projection=projection),
        offsets, margin, countdown,
        onehot(prev_actions.get("ma_agent"), ma_action_count(evo_cfg.actions), device, dtype),
        inbox_features(delivered_messages.get(2, []), budget, mdim, device, dtype),
        tau_feat, step_frac,
    ])

    return {"bs_agent": bs, "ris_agent": ris, "ma_agent": ma}


def build_centralized_state(
    true_components,
    positions: torch.Tensor,
    theta: torch.Tensor,
    centers: torch.Tensor,
    group_map: torch.Tensor,
    prev_actions: dict,
    tau: float,
    step_index: int,
    regime_index: int,
    evo_cfg,
    projection: FeatureProjection | None = None,
) -> torch.Tensor:
    """Centralized critic state (may use the TRUE channel; never given to actors)."""
    cfg = evo_cfg.physical
    device = positions.device
    dtype = cfg.dtype_real
    G = evo_cfg.actions.ris_group_count
    T = cfg.T
    half = _half_extent(cfg)

    offsets = ((positions[0] - centers)[:, 1:] / max(half, _EPS)).reshape(-1).to(dtype)
    margin = torch.tensor(
        [(min_pairwise_distance(positions) - cfg.d_min) / max(cfg.d_min, _EPS)],
        device=device, dtype=dtype,
    )
    return torch.cat([
        complex_to_features(true_components.aggregate[0, 0], projection=projection),
        complex_to_features(true_components.direct[0, 0], projection=projection),
        complex_to_features(true_components.reflected[0, 0], projection=projection),
        grouped_phase_sincos(theta, group_map, G).to(dtype),
        offsets, margin,
        onehot(prev_actions.get("bs_agent"), power_action_count(evo_cfg.actions), device, dtype),
        onehot(prev_actions.get("ris_agent"), ris_action_count(evo_cfg.actions), device, dtype),
        onehot(prev_actions.get("ma_agent"), ma_action_count(evo_cfg.actions), device, dtype),
        torch.tensor([float(tau)], device=device, dtype=dtype),
        onehot(regime_index, T, device, dtype),
        _step_frac(step_index, evo_cfg.dynamics.episode_length, device, dtype),
    ])

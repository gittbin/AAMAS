"""Semantic, deterministic inter-agent message vectors (MVP).

No trainable / differentiable communication in the MVP — each agent emits a fixed,
interpretable feature vector padded/truncated to ``message_dim``. Contents follow
CLAUDE_CODE_TASKS Sec 13; every value is finite and derived from *estimated* /
local quantities only (no true-channel leak).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..physics.csi import EstimatedChannels

_EPS = 1e-30


@dataclass
class MessageContext:
    """Everything needed to build all three agents' messages at a step."""

    estimated: EstimatedChannels     # current estimated components (1,1,K,M)
    theta: torch.Tensor              # (1,L) RIS phases
    positions: torch.Tensor          # (1,M,3) MA layout
    centers: torch.Tensor            # (M,3) cell centres
    tau: float
    power_action_index: int
    power_codebook_size: int
    outage_count: int
    K: int
    d_min: float
    half_extent: float
    ma_update_countdown: int = 0
    ma_update_period: int = 1
    ris_last_changed_groups: int = 0
    ris_group_count: int = 1
    ris_last_sign: float = 0.0


def _ref(ctx: MessageContext):
    agg = ctx.estimated.aggregate
    return agg.device, agg.real.dtype


def _fit(vec: torch.Tensor, message_dim: int) -> torch.Tensor:
    """Pad with zeros / truncate ``vec`` to exactly ``message_dim``."""
    n = vec.shape[0]
    if n == message_dim:
        return vec
    if n > message_dim:
        return vec[:message_dim].contiguous()
    pad = torch.zeros(message_dim - n, dtype=vec.dtype, device=vec.device)
    return torch.cat([vec, pad])


def _min_pairwise_distance(P: torch.Tensor) -> float:
    M = P.shape[1]
    if M < 2:
        return float("inf")
    diff = P[:, :, None, :] - P[:, None, :, :]
    dist = torch.linalg.norm(diff, dim=-1)
    eye = torch.eye(M, dtype=torch.bool, device=P.device).unsqueeze(0)
    return float(dist.masked_fill(eye, float("inf")).min())


def bs_message(ctx: MessageContext, message_dim: int) -> torch.Tensor:
    dev, rdtype = _ref(ctx)
    agg = ctx.estimated.aggregate[0, 0]                     # (K,M)
    norms = agg.abs().pow(2).sum(-1).sqrt()                 # (K,)
    feats = torch.stack([
        torch.log10(norms.min() + _EPS),
        torch.log10(norms.mean() + _EPS),
        torch.tensor(ctx.power_action_index / max(1, ctx.power_codebook_size - 1),
                     dtype=rdtype, device=dev),
        torch.tensor(float(ctx.tau), dtype=rdtype, device=dev),
        torch.tensor(min(1.0, ctx.outage_count / max(1, ctx.K)), dtype=rdtype, device=dev),
    ]).to(rdtype)
    return _fit(feats, message_dim)


def ris_message(ctx: MessageContext, message_dim: int) -> torch.Tensor:
    dev, rdtype = _ref(ctx)
    theta = ctx.theta[0]                                    # (L,)
    mean_cos = torch.cos(theta).mean()
    mean_sin = torch.sin(theta).mean()
    coherence = torch.sqrt(mean_cos ** 2 + mean_sin ** 2)
    rp = ctx.estimated.reflected.abs().pow(2).mean()
    dp = ctx.estimated.direct.abs().pow(2).mean().clamp_min(_EPS)
    log_ratio = torch.log10((rp / dp).clamp_min(_EPS))
    last_move = torch.tensor(
        ctx.ris_last_sign * ctx.ris_last_changed_groups / max(1, ctx.ris_group_count),
        dtype=rdtype, device=dev,
    )
    feats = torch.stack([
        mean_cos, mean_sin, coherence, log_ratio, last_move,
        torch.tensor(float(ctx.tau), dtype=rdtype, device=dev),
    ]).to(rdtype)
    return _fit(feats, message_dim)


def ma_message(ctx: MessageContext, message_dim: int) -> torch.Tensor:
    dev, rdtype = _ref(ctx)
    disp = (ctx.positions[0] - ctx.centers).norm(dim=-1).mean() / max(ctx.half_extent, _EPS)
    min_sp = _min_pairwise_distance(ctx.positions)
    margin = torch.tensor((min_sp - ctx.d_min) / max(ctx.d_min, _EPS), dtype=rdtype, device=dev)
    mag = ctx.estimated.direct[0, 0].abs()                 # (K,M)
    cv = (mag.std(dim=-1) / mag.mean(dim=-1).clamp_min(_EPS)).mean()
    countdown = torch.tensor(
        ctx.ma_update_countdown / max(1, ctx.ma_update_period), dtype=rdtype, device=dev
    )
    feats = torch.stack([disp.to(rdtype), margin, cv.to(rdtype), countdown]).to(rdtype)
    return _fit(feats, message_dim)


def build_agent_messages(ctx: MessageContext, message_dim: int) -> dict[str, torch.Tensor]:
    """Return ``{agent_name: message_vector}`` for all three agents."""
    return {
        "bs_agent": bs_message(ctx, message_dim),
        "ris_agent": ris_message(ctx, message_dim),
        "ma_agent": ma_message(ctx, message_dim),
    }


def message_priority(tau: float, outage_indicator: float, violation_indicator: float = 0.0) -> float:
    """Budget-ranking priority: CSI uncertainty + recent constraint/outage urgency."""
    return float(tau) + float(outage_indicator) + float(violation_indicator)

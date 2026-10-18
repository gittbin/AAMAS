"""Low-level feature extractors for building agent observations.

Compact, fixed-size, finite summaries of complex channel tensors and RIS phases —
never the raw flattened channel unless explicitly projected. All helpers are pure
(no RNG, no mutation).
"""

from __future__ import annotations

import torch

from ..normalization import FeatureProjection
from ..communication.budget import ReceivedMessage

_EPS = 1e-30


def log_mag_stats(x: torch.Tensor) -> torch.Tensor:
    """[mean, std, min, max] of log10(|x|) over all entries. Shape (4,)."""
    lm = torch.log10(x.abs() + _EPS)
    return torch.stack([lm.mean(), lm.std(unbiased=False), lm.min(), lm.max()])


def per_user_norms(x_km: torch.Tensor) -> torch.Tensor:
    """log10 per-user channel norm of a (K,M) complex tensor. Shape (K,)."""
    norms = x_km.abs().pow(2).sum(-1).sqrt()
    return torch.log10(norms + _EPS)


def complex_to_features(
    x_km: torch.Tensor, mode: str = "summary", projection: FeatureProjection | None = None
) -> torch.Tensor:
    """Fixed-size real feature vector summarising a (K,M) complex channel.

    mode="summary": [log-mag mean/std/min/max (4)] + [per-user log-norms (K)]
    (+ [fixed random projection of re/im (proj.out_dim)] if a projection is given).
    """
    if mode != "summary":
        raise ValueError(f"unknown mode {mode!r}; only 'summary' is supported")
    parts = [log_mag_stats(x_km), per_user_norms(x_km)]
    if projection is not None:
        parts.append(projection.project(x_km))
    return torch.cat(parts)


def grouped_phase_sincos(theta: torch.Tensor, group_map: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Per-group mean cos/sin of the RIS phases. Shape (2*n_groups,)."""
    th = theta.reshape(-1)
    cos, sin = torch.cos(th), torch.sin(th)
    device, dtype = th.device, th.dtype
    gmap = group_map.to(device=device, dtype=torch.long)
    cnt = torch.zeros(n_groups, device=device, dtype=dtype).index_add_(0, gmap, torch.ones_like(cos))
    cnt = cnt.clamp_min(1.0)
    csum = torch.zeros(n_groups, device=device, dtype=dtype).index_add_(0, gmap, cos)
    ssum = torch.zeros(n_groups, device=device, dtype=dtype).index_add_(0, gmap, sin)
    return torch.cat([csum / cnt, ssum / cnt])


def min_pairwise_distance(P: torch.Tensor) -> float:
    """Minimum pairwise antenna distance of ``P`` (1,M,3); inf if M<2."""
    M = P.shape[1]
    if M < 2:
        return float("inf")
    diff = P[:, :, None, :] - P[:, None, :, :]
    dist = torch.linalg.norm(diff, dim=-1)
    eye = torch.eye(M, dtype=torch.bool, device=P.device).unsqueeze(0)
    return float(dist.masked_fill(eye, float("inf")).min())


def onehot(index, size: int, device, dtype) -> torch.Tensor:
    v = torch.zeros(size, device=device, dtype=dtype)
    if index is not None and 0 <= int(index) < size:
        v[int(index)] = 1.0
    return v


def message_slot(messages, budget: int, message_dim: int, device, dtype) -> torch.Tensor:
    """Fixed-size (budget*message_dim) buffer of delivered messages, zero-padded."""
    slot = torch.zeros(budget * message_dim, device=device, dtype=dtype)
    for i, msg in enumerate(messages[:budget]):
        slot[i * message_dim:(i + 1) * message_dim] = msg.to(device=device, dtype=dtype)
    return slot


def inbox_features(messages, budget: int, message_dim: int, device, dtype) -> torch.Tensor:
    """Each slot is [availability, sender one-hot (3), payload]."""
    width = message_dim + 4
    slots = torch.zeros(budget, width, device=device, dtype=dtype)
    for i, message in enumerate(messages[:budget]):
        if not isinstance(message, ReceivedMessage):
            raise TypeError("inbox requires ReceivedMessage with sender metadata")
        slots[i, 0] = 1.0
        slots[i, 1:4] = onehot(message.sender, 3, device, dtype)
        slots[i, 4:] = message.payload.to(device=device, dtype=dtype)
    return slots.reshape(-1)

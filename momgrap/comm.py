"""Communication: BD precoding + EVM-aware SINR + rate.  [spec Sec 2.2; INHERITED]

Operates on the aggregated channel rows ``Hh = h_k^H`` of shape ``(N,S,K,M)``
(complex).  Block-diagonalisation (BD) precoding nulls inter-user interference
via a batched SVD null-space projection. Realized SINR retains inter-user
interference when the precoder was built from a different channel estimate.
"""

from __future__ import annotations

import torch

from .config import Config


def bd_precoding(Hh: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Block-diagonalisation precoders ``W``, ``(N,S,M,K)`` complex, unit-norm columns.

    For each user ``k`` the precoder lies in the null space of the other users'
    channels (so ``h_j^H w_k = 0`` for ``j != k``) and is aligned with ``h_k``:
    ``w_k = (V_null V_null^H h_k) / || . ||``.
    """
    N, S, K, M = Hh.shape
    if M < K:
        raise ValueError("single-stream BD requires M >= K")
    h = Hh.conj()                                   # h_k (column) = conj(h_k^H), (N,S,K,M)
    W = torch.empty(N, S, M, K, dtype=Hh.dtype, device=Hh.device)

    idx_all = torch.arange(K, device=Hh.device)
    for k in range(K):
        others = idx_all[idx_all != k]
        H_minus = Hh.index_select(2, others)        # (N,S,K-1,M)
        # full_matrices=True -> Vh is (N,S,M,M); null space = last M-(K-1) cols of V.
        _, _, Vh = torch.linalg.svd(H_minus, full_matrices=True)
        V = Vh.mH                                   # (N,S,M,M)
        V_null = V[..., :, (K - 1):]                # (N,S,M, M-K+1)
        h_k = h[:, :, k, :]                         # (N,S,M)
        # project h_k onto the null space
        coeff = torch.einsum("nsmr,nsm->nsr", V_null.conj(), h_k)   # (N,S,R)
        w = torch.einsum("nsmr,nsr->nsm", V_null, coeff)            # (N,S,M)
        norm = torch.linalg.norm(w, dim=-1, keepdim=True)
        # Degenerate desired channels still need a unit beam to preserve the
        # transmit-power trace. Such a beam has zero useful desired gain.
        W[:, :, :, k] = torch.where(norm > 1e-12, w / norm.clamp_min(1e-12), V_null[..., 0])
    return W


def power_allocation(Hh: torch.Tensor, W: torch.Tensor, cfg: Config) -> torch.Tensor | None:
    """Per-user transmit power ``P_k`` ``(N,S,K)`` for the configured policy.

    ``"equal"`` returns ``None`` (the caller uses the scalar ``P_max/K`` fast path,
    numerically identical to the original model).  ``"inv_gain"`` returns a
    fairness-oriented allocation ``P_k ∝ 1/|h_k^H w_k|^2`` renormalised to
    ``sum_k P_k = P_max`` (weak users get more power).
    """
    if cfg.power_policy == "equal":
        return None
    hkwk = torch.einsum("nskm,nsmk->nsk", Hh, W)            # (N,S,K)
    gain = (hkwk.abs() ** 2).clamp_min(1e-12)              # effective desired gain
    inv = 1.0 / gain
    weights = inv / inv.sum(dim=-1, keepdim=True)           # sum_k = 1
    return cfg.p_max_watt * weights                         # (N,S,K), sums to P_max


def evm_sinr_rate(Hh: torch.Tensor, W: torch.Tensor, cfg: Config,
                  p_user: torch.Tensor | None = None) -> torch.Tensor:
    """EVM-aware per-user rate ``R_{n,s,k}``, ``(N,S,K)`` real (bps/Hz).

    ``SINR_k = (1-k^2) P_k |h_k^H w_k|^2 / (sigma^2 + I_k + EVM_k)``
    where ``I_k = (1-k^2) sum_{j != k} P_j |h_k^H w_j|^2``.
    with ``E|x_m|^2 = sum_j P_j |w_{j,m}|^2``.  ``p_user`` (N,S,K) supplies a per-user
    power allocation; ``None`` (default) uses uniform ``P_k = P_max/K`` (scalar path,
    unchanged from the spec model).
    """
    kappa2 = cfg.evm_kappa ** 2
    sigma2 = cfg.noise_power_watt

    # desired signal: h_k^H w_k  (Hh already is h^H)
    hkwk = torch.einsum("nskm,nsmk->nsk", Hh, W)             # (N,S,K)
    if p_user is None:
        p_k = cfg.p_per_user                                # scalar P_max/K
        signal = (1.0 - kappa2) * p_k * (hkwk.abs() ** 2)   # (N,S,K)
        ex_m = p_k * (W.abs() ** 2).sum(dim=-1)             # (N,S,M)
    else:
        signal = (1.0 - kappa2) * p_user * (hkwk.abs() ** 2)          # (N,S,K)
        # E|x_m|^2 = sum_j P_j |w_{j,m}|^2  -> weight columns of W by per-user power
        ex_m = torch.einsum("nsmk,nsk->nsm", (W.abs() ** 2), p_user)  # (N,S,M)
    # EVM distortion seen by user k: k^2 * sum_m |h_{k,m}|^2 E|x_m|^2
    h_abs2 = Hh.abs() ** 2                                   # (N,S,K,M)
    evm = kappa2 * torch.einsum("nskm,nsm->nsk", h_abs2, ex_m)  # (N,S,K)

    # Mask diagonal gains before summation: total-minus-desired can lose
    # small residual interference through float32 cancellation.
    cross_gain = torch.einsum("nskm,nsmj->nskj", Hh, W).abs().square()
    diagonal = torch.eye(Hh.shape[-2], device=Hh.device, dtype=torch.bool)
    cross_gain = cross_gain.masked_fill(diagonal, 0.0)
    if p_user is None:
        interference = (1.0 - kappa2) * cfg.p_per_user * cross_gain.sum(-1)
    else:
        interference = (1.0 - kappa2) * torch.einsum("nskj,nsj->nsk", cross_gain, p_user)
    sinr = signal / (sigma2 + interference + evm)
    rate = torch.log2(1.0 + sinr)
    return rate

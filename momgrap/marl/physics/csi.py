"""Imperfect CSI: true-vs-estimated channel separation + actor delay buffer.

The MARL environment must never let the actor or the block-diagonalisation (BD)
precoder see the *true* channel directly. This module provides:

* :func:`apply_complex_csi_error` — the per-tensor estimation-error model
  ``Hhat = sqrt(1-tau^2) H + tau Z_H`` with a circular-complex-Gaussian ``Z_H``
  whose per-entry power is scaled to the channel RMS (so unit-variance noise does
  not swamp the very small path-loss magnitudes);
* :func:`estimate_channel_components` — apply it to the direct/reflected split;
* :class:`CSIHistoryBuffer` — the delayed-observation ring the actor reads from.

Error-model note (used by the tests): with the RMS scaling and ``Z_H`` independent
of ``H`` the normalised MSE has the closed form ``NMSE(tau) = 2 (1 - sqrt(1-tau^2))``
(0 at ``tau=0``, 2 at ``tau=1``), monotonically increasing in ``tau``.

Data-flow contract (enforced by the backend in Task 07, guarded here by tests):

    W = bd_precoding(H_est_for_beamformer, cfg)   # BD uses the ESTIMATE
    R = evm_sinr_rate(H_true, W, cfg, p_user)      # realized rate uses the TRUTH

Never ``bd_precoding(H_true, cfg)`` in the default imperfect-CSI environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .components import CommChannelComponents

_VALID_SCALING = ("rms", "unit")


@dataclass
class EstimatedChannels:
    """Estimated (noisy) channel components handed to the actor / BD precoder.

    Intentionally holds **no** reference to any true-channel tensor — the actor
    getter must never be able to reach ground truth through this object.

    Shapes mirror :class:`~momgrap.channels.CommChannelComponents` (complex)::

        direct     (N, S, K, M)
        reflected  (N, S, K, M)
        aggregate  (N, S, K, M)   (== direct + reflected)
    """

    direct: torch.Tensor
    reflected: torch.Tensor
    aggregate: torch.Tensor
    tau: float
    reference_positions: torch.Tensor | None = None
    reference_theta: torch.Tensor | None = None
    step_index: int | None = None


def apply_complex_csi_error(
    true_tensor: torch.Tensor,
    tau: float,
    generator: torch.Generator,
    scaling: str = "rms",
) -> torch.Tensor:
    """Return a noisy estimate ``Hhat = sqrt(1-tau^2) H + tau Z_H`` of a channel.

    ``Z_H`` is circular complex Gaussian with ``E|Z|^2 = 1``; with ``scaling="rms"``
    it is multiplied by ``sqrt(mean(|H|^2))`` so its power matches the channel
    (avoids unit-variance noise dominating tiny path-loss magnitudes). ``tau=0``
    returns an exact (cloned) copy of the truth. dtype and device are preserved.
    """
    if not torch.is_tensor(true_tensor) or not true_tensor.is_complex():
        raise ValueError("apply_complex_csi_error expects a complex tensor")
    tau = float(tau)
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau={tau} must be in [0,1]")
    if scaling not in _VALID_SCALING:
        raise ValueError(f"unknown scaling {scaling!r}; expected one of {_VALID_SCALING}")

    if tau == 0.0:
        return true_tensor.clone()

    real_dtype = true_tensor.real.dtype
    device = true_tensor.device
    shape = true_tensor.shape
    a = torch.randn(shape, generator=generator, device=device, dtype=real_dtype)
    b = torch.randn(shape, generator=generator, device=device, dtype=real_dtype)
    z = torch.complex(a, b) * (1.0 / math.sqrt(2.0))     # E|z|^2 = 1

    if scaling == "rms":
        rms = true_tensor.abs().pow(2).mean().clamp_min(1e-30).sqrt()
        z = z * rms

    coeff = math.sqrt(1.0 - tau * tau)
    return coeff * true_tensor + tau * z


def estimate_channel_components(
    true_components: CommChannelComponents,
    tau: float,
    generator: torch.Generator,
    scaling: str = "rms",
    *,
    positions: torch.Tensor | None = None,
    theta: torch.Tensor | None = None,
    step_index: int | None = None,
) -> EstimatedChannels:
    """Apply the CSI error model to the direct and reflected components.

    The aggregate estimate is the sum of the two estimated components, so the BD
    input (``aggregate``) and the actor observations (``direct``/``reflected``)
    share one coherent noise realisation. At ``tau=0`` every field equals truth.
    """
    direct_hat = apply_complex_csi_error(true_components.direct, tau, generator, scaling)
    reflected_hat = apply_complex_csi_error(true_components.reflected, tau, generator, scaling)
    aggregate_hat = direct_hat + reflected_hat
    return EstimatedChannels(
        direct=direct_hat,
        reflected=reflected_hat,
        aggregate=aggregate_hat,
        tau=float(tau),
        reference_positions=positions.clone() if positions is not None else None,
        reference_theta=theta.clone() if theta is not None else None,
        step_index=step_index,
    )


class CSIHistoryBuffer:
    """Ring of past :class:`EstimatedChannels` for delayed actor observations.

    ``get_delayed(d)`` returns the estimate from ``d`` steps ago (``d=0`` = latest).
    When the history is shorter than ``d`` it repeats the *first* stored estimate
    (never zero-padding), so early-episode observations stay physically plausible.
    """

    def __init__(self, maxlen: int | None = None):
        if maxlen is not None and maxlen <= 0:
            raise ValueError(f"maxlen={maxlen} must be > 0 or None")
        self._maxlen = maxlen
        self._buf: list[EstimatedChannels] = []

    def reset(self, initial: EstimatedChannels) -> None:
        if not isinstance(initial, EstimatedChannels):
            raise TypeError("CSIHistoryBuffer stores EstimatedChannels")
        self._buf = [initial]

    def append(self, value: EstimatedChannels) -> None:
        if not self._buf:
            raise RuntimeError("call reset() before append()")
        if not isinstance(value, EstimatedChannels):
            raise TypeError("CSIHistoryBuffer stores EstimatedChannels")
        self._buf.append(value)
        if self._maxlen is not None and len(self._buf) > self._maxlen:
            self._buf = self._buf[-self._maxlen:]

    def get_delayed(self, delay_steps: int) -> EstimatedChannels:
        if not self._buf:
            raise RuntimeError("CSI history is empty; call reset() first")
        if delay_steps < 0:
            raise ValueError(f"delay_steps={delay_steps} must be >= 0")
        idx = len(self._buf) - 1 - delay_steps
        if idx < 0:
            idx = 0                      # repeat first estimate (no zero-pad)
        return self._buf[idx]

    def __len__(self) -> int:
        return len(self._buf)

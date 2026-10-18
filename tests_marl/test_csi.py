"""Tests for imperfect CSI (error model + delay buffer + no-leak) — Task 03."""

from __future__ import annotations

import dataclasses
import math
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch

from momgrap import smoke_config
from momgrap.channels import build_comm_channel_components, sample_environment
from momgrap.comm import bd_precoding, evm_sinr_rate
from momgrap.genotype import decode
from momgrap.marl.physics.csi import (
    CSIHistoryBuffer,
    EstimatedChannels,
    apply_complex_csi_error,
    estimate_channel_components,
)


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def _true_components(regime_index: int = 0, seed: int = 7, n: int = 48, s: int = 4):
    cfg = smoke_config(device_str="cpu")
    g = _gen(seed)
    pop = torch.rand(n, cfg.D, generator=g, device="cpu", dtype=cfg.dtype_real)
    P_ant, theta = decode(pop, cfg)
    env = sample_environment(cfg.regimes[regime_index], s, g, cfg)
    return cfg, build_comm_channel_components(P_ant, theta, env, cfg)


def _avg_nmse(H: torch.Tensor, tau: float, draws: int = 96) -> float:
    if tau == 0.0:
        return 0.0
    true_pow = float(H.abs().pow(2).mean())
    acc = 0.0
    for s in range(draws):
        Hhat = apply_complex_csi_error(H, tau, _gen(10_000 + s), scaling="rms")
        acc += float((Hhat - H).abs().pow(2).mean()) / true_pow
    return acc / draws


# ---------------------------------------------------------------- error model
def test_tau_zero_returns_truth():
    cfg, comp = _true_components()
    est = estimate_channel_components(comp, 0.0, _gen(0))
    assert torch.equal(est.direct, comp.direct)
    assert torch.equal(est.reflected, comp.reflected)
    assert torch.equal(est.aggregate, comp.aggregate)
    assert est.tau == 0.0


def test_nmse_matches_closed_form():
    """Averaged NMSE ~ 2(1 - sqrt(1-tau^2)) for the per-tensor RMS error model."""
    _cfg, comp = _true_components(n=64, s=4)
    H = comp.direct
    for tau in (0.05, 0.1, 0.2, 0.3):
        nmse = _avg_nmse(H, tau)
        expected = 2.0 * (1.0 - math.sqrt(1.0 - tau * tau))
        assert abs(nmse - expected) < 0.15 * expected + 0.01, (tau, nmse, expected)


def test_relative_error_increases_with_tau():
    _cfg, comp = _true_components()
    taus = [0.0, 0.05, 0.1, 0.2, 0.3]
    for field in (comp.direct, comp.reflected, comp.aggregate):
        nmses = [_avg_nmse(field, t) for t in taus]
        for i in range(len(nmses) - 1):
            assert nmses[i] < nmses[i + 1], (field.shape, nmses)


def test_deterministic_same_seed():
    _cfg, comp = _true_components()
    H = comp.direct
    a = apply_complex_csi_error(H, 0.2, _gen(42))
    b = apply_complex_csi_error(H, 0.2, _gen(42))
    assert torch.equal(a, b)
    c = apply_complex_csi_error(H, 0.2, _gen(43))
    assert not torch.equal(a, c)


def test_complex_dtype_preserved():
    _cfg, comp = _true_components()
    H = comp.direct
    assert H.dtype == torch.complex64
    out = apply_complex_csi_error(H, 0.2, _gen(1))
    assert out.dtype == torch.complex64

    H128 = H.to(torch.complex128)
    out128 = apply_complex_csi_error(H128, 0.2, _gen(1))
    assert out128.dtype == torch.complex128


def test_invalid_arguments():
    _cfg, comp = _true_components()
    H = comp.direct
    with pytest.raises(ValueError):
        apply_complex_csi_error(H, 1.5, _gen(0))
    with pytest.raises(ValueError):
        apply_complex_csi_error(H, -0.1, _gen(0))
    with pytest.raises(ValueError):
        apply_complex_csi_error(H, 0.2, _gen(0), scaling="bogus")
    with pytest.raises(ValueError):
        apply_complex_csi_error(H.real, 0.2, _gen(0))  # non-complex


# ---------------------------------------------------------------- delay buffer
def test_history_buffer_delay():
    _cfg, comp = _true_components()
    g = _gen(0)
    e0 = estimate_channel_components(comp, 0.1, g)
    buf = CSIHistoryBuffer()
    buf.reset(e0)
    assert len(buf) == 1
    assert buf.get_delayed(0) is e0
    assert buf.get_delayed(3) is e0          # not enough history -> repeat first

    e1 = estimate_channel_components(comp, 0.1, g)
    e2 = estimate_channel_components(comp, 0.1, g)
    buf.append(e1)
    buf.append(e2)
    assert buf.get_delayed(0) is e2
    assert buf.get_delayed(1) is e1
    assert buf.get_delayed(2) is e0
    assert buf.get_delayed(9) is e0          # clamp to first, no zero-pad


def test_history_buffer_errors():
    buf = CSIHistoryBuffer()
    with pytest.raises(RuntimeError):
        buf.get_delayed(0)                   # empty
    with pytest.raises(RuntimeError):
        buf.append(estimate_channel_components(_true_components()[1], 0.1, _gen(0)))
    _cfg, comp = _true_components()
    buf.reset(estimate_channel_components(comp, 0.1, _gen(0)))
    with pytest.raises(ValueError):
        buf.get_delayed(-1)


def test_history_buffer_maxlen():
    _cfg, comp = _true_components()
    buf = CSIHistoryBuffer(maxlen=2)
    ests = [estimate_channel_components(comp, 0.1, _gen(i)) for i in range(4)]
    buf.reset(ests[0])
    for e in ests[1:]:
        buf.append(e)
    assert len(buf) == 2
    assert buf.get_delayed(0) is ests[3]
    assert buf.get_delayed(1) is ests[2]


# ---------------------------------------------------------------- no leak
def test_estimated_channels_has_no_true_field():
    fields = {f.name for f in dataclasses.fields(EstimatedChannels)}
    assert fields == {"direct", "reflected", "aggregate", "tau",
                      "reference_positions", "reference_theta", "step_index"}
    assert not any("true" in name or "environment" in name for name in fields)


def test_actor_getter_returns_noisy_estimate_only():
    _cfg, comp = _true_components()
    buf = CSIHistoryBuffer()
    buf.reset(estimate_channel_components(comp, 0.2, _gen(1)))
    buf.append(estimate_channel_components(comp, 0.2, _gen(2)))
    delayed = buf.get_delayed(1)
    assert isinstance(delayed, EstimatedChannels)
    # the delayed estimate is noisy: not equal to the true channel it estimates
    assert not torch.equal(delayed.aggregate, comp.aggregate)


# ---------------------------------------------------------------- integration
def test_bd_uses_estimate_rate_uses_truth():
    cfg, comp = _true_components(n=8, s=3)
    est = estimate_channel_components(comp, 0.15, _gen(3))

    W_est = bd_precoding(est.aggregate, cfg)      # BD on the ESTIMATE
    W_true = bd_precoding(comp.aggregate, cfg)
    assert not torch.equal(W_est, W_true)         # proves BD did not use truth

    R = evm_sinr_rate(comp.aggregate, W_est, cfg)  # realized rate on the TRUTH
    assert torch.isfinite(R).all()
    assert float(R.min()) >= 0.0

    # tau=0: estimate == truth -> identical precoder
    est0 = estimate_channel_components(comp, 0.0, _gen(3))
    W0 = bd_precoding(est0.aggregate, cfg)
    assert torch.equal(W0, W_true)

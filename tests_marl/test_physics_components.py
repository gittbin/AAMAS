"""Tests for the backward-compatible channel component API (Task 02)."""

from __future__ import annotations

import pathlib
import sys

# Make the repository root importable regardless of pytest's import mode / cwd.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from momgrap import smoke_config
from momgrap.channels import (
    build_comm_channel_components,
    build_comm_channels,
    sample_environment,
)
from momgrap.genotype import decode

# The MARL-namespace re-export must be the very same objects.
from momgrap.marl.physics import components as marl_components

N_IND = 6   # N > 1
S_MC = 3    # S > 1


def _setup(regime_index: int = 0, seed: int = 123):
    """Deterministic CPU population + environment for one regime."""
    cfg = smoke_config(device_str="cpu")
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    pop = torch.rand(N_IND, cfg.D, generator=g, device="cpu", dtype=cfg.dtype_real)
    P_ant, theta = decode(pop, cfg)
    env = sample_environment(cfg.regimes[regime_index], S_MC, g, cfg)
    return cfg, P_ant, theta, env


def test_aggregate_matches_legacy_wrapper():
    """components.aggregate is bit-for-bit the legacy build_comm_channels output."""
    for regime_index in range(3):
        cfg, P_ant, theta, env = _setup(regime_index)
        comp = build_comm_channel_components(P_ant, theta, env, cfg)
        legacy = build_comm_channels(P_ant, theta, env, cfg)
        assert torch.equal(comp.aggregate, legacy)


def test_aggregate_is_direct_plus_reflected():
    for regime_index in range(3):
        cfg, P_ant, theta, env = _setup(regime_index)
        comp = build_comm_channel_components(P_ant, theta, env, cfg)
        assert torch.equal(comp.aggregate, comp.direct + comp.reflected)


def test_component_shapes():
    cfg, P_ant, theta, env = _setup()
    comp = build_comm_channel_components(P_ant, theta, env, cfg)
    M, L, K = cfg.M, cfg.L, cfg.K
    assert comp.direct.shape == (N_IND, S_MC, K, M)
    assert comp.bs_ris.shape == (N_IND, S_MC, L, M)
    assert comp.ris_user.shape == (S_MC, K, L)
    assert comp.reflected.shape == (N_IND, S_MC, K, M)
    assert comp.aggregate.shape == (N_IND, S_MC, K, M)


def test_components_finite_and_complex():
    for regime_index in range(3):
        cfg, P_ant, theta, env = _setup(regime_index)
        comp = build_comm_channel_components(P_ant, theta, env, cfg)
        for t in (comp.direct, comp.bs_ris, comp.ris_user, comp.reflected, comp.aggregate):
            assert torch.isfinite(t).all()
            assert t.is_complex()


def test_components_on_cpu():
    cfg, P_ant, theta, env = _setup()
    comp = build_comm_channel_components(P_ant, theta, env, cfg)
    for t in (comp.direct, comp.bs_ris, comp.ris_user, comp.reflected, comp.aggregate):
        assert t.device.type == "cpu"


def test_deterministic_no_added_randomness():
    """Two calls on the same env/genotype return identical tensors (no RNG use)."""
    cfg, P_ant, theta, env = _setup()
    a = build_comm_channel_components(P_ant, theta, env, cfg)
    b = build_comm_channel_components(P_ant, theta, env, cfg)
    assert torch.equal(a.aggregate, b.aggregate)
    assert torch.equal(a.direct, b.direct)
    assert torch.equal(a.reflected, b.reflected)


def test_marl_reexport_identity():
    """momgrap.marl.physics re-exports the exact core symbols (no re-implementation)."""
    assert marl_components.build_comm_channel_components is build_comm_channel_components
    from momgrap.channels import CommChannelComponents as CoreComponents
    assert marl_components.CommChannelComponents is CoreComponents

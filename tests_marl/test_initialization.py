"""Tests for RAMP offline-archive initialization (Task 12)."""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch

from momgrap import smoke_config
from momgrap.channels import bs_cell_centers
from momgrap.marl.actions.safety import _positions_feasible
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.physics.initialization import (
    RAMP_MODES,
    build_ramp_initializations,
    check_compatibility,
    initial_config_from,
    load_initialization,
    save_initialization,
)

SMOKE = _REPO_ROOT / "configs_marl" / "smoke.yaml"


@pytest.fixture(scope="module")
def ramp():
    evo = load_evocomm_config(SMOKE)
    init = build_ramp_initializations(evo.physical, seed=0)
    return init, evo


# ---------------------------------------------------------------- build
def test_build_shapes_and_selection(ramp):
    init, evo = ramp
    cfg = evo.physical
    P = init.n_points
    assert P >= 1
    assert init.front.shape == (P, 2)
    assert init.genotypes.shape == (P, cfg.D)
    assert init.positions.shape == (P, cfg.M, 3)
    assert init.theta.shape == (P, cfg.L)
    assert set(init.selected_indices) == {"knee", "comm_extreme", "sensing_extreme"}
    # extremes are actually extreme
    assert init.selected_indices["comm_extreme"] == int(init.front[:, 0].argmax())
    assert init.selected_indices["sensing_extreme"] == int(init.front[:, 1].argmax())


def test_deterministic_selection():
    evo = load_evocomm_config(SMOKE)
    a = build_ramp_initializations(evo.physical, seed=0)
    b = build_ramp_initializations(evo.physical, seed=0)
    assert torch.equal(a.front, b.front)
    assert a.selected_indices == b.selected_indices
    assert a.spaced_indices == b.spaced_indices
    assert torch.equal(a.positions, b.positions)


# ---------------------------------------------------------------- feasibility
def test_decoded_configs_feasible(ramp):
    init, evo = ramp
    cfg = evo.physical
    centers = bs_cell_centers(cfg).to(dtype=cfg.dtype_real)
    for idx in init.selected_indices.values():
        P = init.positions[idx:idx + 1]
        assert _positions_feasible(P, centers, cfg, 1e-6 * cfg.d_min, 1e-6)


# ---------------------------------------------------------------- save/load
def test_save_load_roundtrip(ramp, tmp_path):
    init, _evo = ramp
    pt_path, json_path = save_initialization(init, tmp_path / "init")
    assert pathlib.Path(pt_path).exists() and pathlib.Path(json_path).exists()
    loaded = load_initialization(pt_path)
    assert loaded.seed == init.seed
    assert (loaded.M, loaded.L, loaded.K, loaded.Q) == (init.M, init.L, init.K, init.Q)
    assert torch.equal(loaded.front, init.front)
    assert torch.equal(loaded.positions, init.positions)
    assert torch.equal(loaded.theta, init.theta)
    assert loaded.selected_indices == init.selected_indices


# ---------------------------------------------------------------- compatibility
def test_compatibility_ok_and_rejects_wrong_dims(ramp):
    init, evo = ramp
    check_compatibility(init, evo.physical)                 # same cfg -> OK
    with pytest.raises(ValueError):
        check_compatibility(init, smoke_config())           # default M=16/L=64 -> mismatch


# ---------------------------------------------------------------- modes
def test_initial_config_from_modes(ramp):
    init, evo = ramp
    cfg = evo.physical
    g = torch.Generator().manual_seed(0)
    for mode in RAMP_MODES:
        pos, th = initial_config_from(init, mode, generator=g)
        assert pos.shape == (1, cfg.M, 3)
        assert th.shape == (1, cfg.L)
    # knee vs comm-extreme differ when the front has >1 distinct point
    if init.selected_indices["knee"] != init.selected_indices["comm_extreme"]:
        pk, _ = initial_config_from(init, "ramp_knee")
        pc, _ = initial_config_from(init, "ramp_comm_extreme")
        assert not torch.equal(pk, pc)


def test_env_uses_ramp_knee_init(ramp):
    init, evo = ramp
    env = MOMGRAPParallelEnv(evo, seed=0, init_mode="ramp_knee", ramp_init=init)
    env.reset(seed=0)
    idx = init.selected_indices["knee"]
    expected = init.positions[idx:idx + 1].to(env._pstate.positions.dtype)
    assert torch.allclose(env._pstate.positions.cpu(), expected, atol=1e-5)


def test_env_ramp_mode_requires_init(ramp):
    _init, evo = ramp
    env = MOMGRAPParallelEnv(evo, seed=0, init_mode="ramp_knee", ramp_init=None)
    with pytest.raises(ValueError):
        env.reset(seed=0)

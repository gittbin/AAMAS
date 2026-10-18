"""Tests for the EvoComm-MOMGRAP configuration layer (Task 01)."""

from __future__ import annotations

import dataclasses
import pathlib
import sys

# Make the repository root importable regardless of pytest's import mode / cwd.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch
import yaml

from momgrap.config import Config, default_config
from momgrap.marl.config import (
    EvoCommConfig,
    evocomm_config_to_dict,
    load_evocomm_config,
    resolve_dtype,
    save_resolved_config,
)

CONFIGS = _REPO_ROOT / "configs_marl"
SMOKE = CONFIGS / "smoke.yaml"
MVP = CONFIGS / "mvp.yaml"


# ---------------------------------------------------------------- happy paths
def test_load_smoke_values():
    cfg = load_evocomm_config(SMOKE)
    assert isinstance(cfg, EvoCommConfig)

    # physical block -> a real momgrap Config
    assert isinstance(cfg.physical, Config)
    assert (cfg.physical.M, cfg.physical.L, cfg.physical.K, cfg.physical.Q) == (4, 16, 2, 1)
    assert cfg.physical.device_str == "cpu"
    assert cfg.physical.D == 2 * 4 + 16  # derived property still works

    assert cfg.dynamics.episode_length == 16
    assert cfg.dynamics.regime_sampling == "uniform_episode"
    assert cfg.csi.tau_eval == 0.05
    assert cfg.csi.actor_delay_steps == 1
    assert cfg.csi.noise_scaling == "rms"
    assert cfg.actions.power_codebook_size == 7
    assert cfg.actions.ris_group_count == 4
    assert cfg.communication.graph_type == "full"
    assert cfg.communication.delivery_delay_steps == 1
    assert cfg.reward.w_min_rate == 1.0
    assert cfg.rmappo.num_envs == 2
    assert cfg.rmappo.hidden_size == 32


def test_load_mvp_values():
    cfg = load_evocomm_config(MVP)
    assert (cfg.physical.M, cfg.physical.L, cfg.physical.K, cfg.physical.Q) == (16, 64, 4, 3)
    assert cfg.actions.ris_group_count == 8
    assert cfg.physical.L % cfg.actions.ris_group_count == 0
    assert cfg.dynamics.episode_length == 100


def test_physical_grid_consistency():
    for path in (SMOKE, MVP):
        cfg = load_evocomm_config(path)
        p = cfg.physical
        assert p.bs_grid_rows * p.bs_grid_cols == p.M
        assert p.ris_grid_rows * p.ris_grid_cols == p.L


# ---------------------------------------------------------------- overrides
def test_overrides_are_deep_merged():
    cfg = load_evocomm_config(
        SMOKE, overrides={"dynamics": {"episode_length": 32}, "csi": {"tau_eval": 0.2}}
    )
    assert cfg.dynamics.episode_length == 32
    assert cfg.csi.tau_eval == 0.2
    # unrelated fields untouched
    assert cfg.dynamics.gauss_markov_rho == 0.90
    # file itself is not mutated: a fresh load has the original value
    assert load_evocomm_config(SMOKE).dynamics.episode_length == 16


def test_physical_override():
    cfg = load_evocomm_config(SMOKE, overrides={"physical": {"log_every": 5}})
    assert cfg.physical.log_every == 5
    assert cfg.physical.M == 4  # other physical fields preserved


# ---------------------------------------------------------------- dtype handling
def test_dtype_real_string():
    assert resolve_dtype("float64") is torch.float64
    assert resolve_dtype("float32") is torch.float32
    assert resolve_dtype("torch.float64") is torch.float64
    assert resolve_dtype(torch.float32) is torch.float32

    cfg = load_evocomm_config(SMOKE, overrides={"physical": {"dtype_real": "float64"}})
    assert cfg.physical.dtype_real is torch.float64
    assert cfg.physical.dtype_complex is torch.complex128


def test_dtype_real_invalid():
    with pytest.raises(ValueError):
        load_evocomm_config(SMOKE, overrides={"physical": {"dtype_real": "float16"}})


# ---------------------------------------------------------------- immutability
def test_configs_are_frozen():
    cfg = load_evocomm_config(SMOKE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.dynamics.episode_length = 999  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.rmappo.seed = 7  # type: ignore[misc]


def test_global_config_not_mutated():
    before = default_config().M
    load_evocomm_config(SMOKE)  # sets physical M=4 internally
    after = default_config().M
    assert before == after == 16


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "overrides",
    [
        {"csi": {"tau_eval": 1.5}},                       # tau out of [0,1]
        {"csi": {"tau_train_min": -0.1}},                 # tau out of [0,1]
        {"csi": {"tau_train_min": 0.3, "tau_train_max": 0.1}},  # min > max
        {"csi": {"actor_delay_steps": -1}},               # negative delay
        {"dynamics": {"gauss_markov_rho": 1.0}},          # rho not in [0,1)
        {"dynamics": {"gauss_markov_rho": -0.1}},         # rho not in [0,1)
        {"dynamics": {"episode_length": 0}},              # episode_length <= 0
        {"dynamics": {"regime_sampling": "bogus"}},       # unknown regime sampling
        {"actions": {"ma_update_period": 0}},             # ma_update_period <= 0
        {"actions": {"ris_group_count": 0}},              # ris_group_count <= 0
        {"actions": {"ris_group_count": 5}},              # L(16) % 5 != 0
        {"communication": {"max_messages_per_step": -1}},  # negative budget
        {"communication": {"graph_type": "mesh"}},        # unknown graph preset
        {"physical": {"M": 5}},                           # grid 2x2 != M -> ValueError
    ],
)
def test_validation_rejects(overrides):
    with pytest.raises(ValueError):
        load_evocomm_config(SMOKE, overrides=overrides)


def test_missing_section_rejected(tmp_path):
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    del raw["reward"]
    bad = tmp_path / "no_reward.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_evocomm_config(bad)


def test_unknown_key_rejected(tmp_path):
    raw = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    raw["dynamics"]["not_a_field"] = 1
    bad = tmp_path / "extra_key.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_evocomm_config(bad)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_evocomm_config(CONFIGS / "does_not_exist.yaml")


# ---------------------------------------------------------------- serialisation
def test_serializable_roundtrip(tmp_path):
    cfg = load_evocomm_config(SMOKE)
    d = evocomm_config_to_dict(cfg)

    # dtype is a string, ris_position a list, regimes plain dicts -> YAML-dumpable
    assert d["physical"]["dtype_real"] == "float32"
    assert isinstance(d["physical"]["ris_position"], list)
    assert all(set(reg) == {"name", "k_factor_db"} for reg in d["physical"]["regimes"])

    text = yaml.safe_dump(d)  # must not raise
    reloaded = yaml.safe_load(text)
    assert reloaded["dynamics"]["episode_length"] == 16
    assert reloaded["rmappo"]["hidden_size"] == 32

    out = save_resolved_config(cfg, tmp_path / "resolved.yaml")
    assert pathlib.Path(out).exists()
    on_disk = yaml.safe_load(pathlib.Path(out).read_text(encoding="utf-8"))
    assert on_disk["physical"]["M"] == 4

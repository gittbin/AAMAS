"""EvoComm-MOMGRAP configuration layer.

Adds a set of frozen dataclasses describing the MARL extension (dynamics, CSI
error, action codebooks, communication, reward shaping, RMAPPO) *around* the
existing ``momgrap.config.Config`` without modifying it. A YAML loader assembles
an :class:`EvoCommConfig`, validating every field with explicit ``ValueError``s
(not bare ``assert``s) so misconfiguration fails loudly and early.

Design rules (CLAUDE_CODE_TASKS Sec 6):
* the physical block is turned into a ``Config`` via ``default_config(**physical)``
  so the momgrap defaults/validation still apply — the global config is never mutated;
* ``dtype_real`` may be given as a string (e.g. ``"float32"``);
* the resolved config is convertible to a plain, YAML/JSON-serialisable dict.
"""

from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import dataclass
from typing import Any

import torch
import yaml

from ..config import Config, default_config

# ---------------------------------------------------------------- allowed values
VALID_REGIME_SAMPLING = ("fixed_los", "fixed_rayleigh", "fixed_rician", "uniform_episode")
VALID_NOISE_SCALING = ("rms", "unit")
VALID_GRAPH_TYPES = ("none", "full", "ring", "star_bs", "custom")

_DTYPE_ALIASES = {
    "float32": torch.float32, "float": torch.float32, "fp32": torch.float32,
    "torch.float32": torch.float32,
    "float64": torch.float64, "double": torch.float64, "fp64": torch.float64,
    "torch.float64": torch.float64,
}


# ---------------------------------------------------------------- dataclasses
@dataclass(frozen=True)
class DynamicsConfig:
    episode_length: int
    dt_seconds: float
    regime_sampling: str
    gauss_markov_rho: float
    mobility_enabled: bool
    user_speed_min_mps: float
    user_speed_max_mps: float
    resample_scatter_directions_each_episode: bool


@dataclass(frozen=True)
class CSIConfig:
    tau_train_min: float
    tau_train_max: float
    tau_eval: float
    actor_delay_steps: int
    beamformer_uses_current_estimate: bool
    expose_tau_to_actor: bool
    noise_scaling: str
    feature_mode: str = "normal"


@dataclass(frozen=True)
class ActionConfig:
    power_codebook_size: int
    ris_group_count: int
    ris_phase_step_deg: float
    ma_step_fraction_of_cell: float
    ma_update_period: int


@dataclass(frozen=True)
class CommunicationConfig:
    enabled: bool
    graph_type: str
    max_messages_per_step: int
    message_dim: int
    message_cost: float
    delivery_delay_steps: int


@dataclass(frozen=True)
class RewardConfig:
    rate_scale: float
    sensing_offset_db: float
    sensing_scale_db: float
    w_min_rate: float
    w_sum_rate: float
    w_sensing: float
    w_outage: float
    w_message: float
    w_reconfiguration: float
    w_violation: float
    rate_target_bpshz: float


@dataclass(frozen=True)
class RMAPPOConfig:
    seed: int
    total_timesteps: int
    num_envs: int
    rollout_length: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    update_epochs: int
    num_minibatches: int
    hidden_size: int
    sequence_length: int
    eval_interval: int
    save_interval: int
    recurrent_actor: bool = True
    comm_entropy_coef: float | None = None
    centralized_actor_hidden_size: int | None = None


@dataclass(frozen=True)
class EvoCommConfig:
    physical: Config
    dynamics: DynamicsConfig
    csi: CSIConfig
    actions: ActionConfig
    communication: CommunicationConfig
    reward: RewardConfig
    rmappo: RMAPPOConfig


# ---------------------------------------------------------------- dtype helpers
def resolve_dtype(value: Any) -> torch.dtype:
    """Map a YAML dtype (string or already-a-dtype) to a ``torch.dtype``."""
    if isinstance(value, torch.dtype):
        return value
    key = str(value).strip().lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(
            f"unknown dtype_real {value!r}; expected one of {sorted(set(_DTYPE_ALIASES))}"
        )
    return _DTYPE_ALIASES[key]


def dtype_to_str(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


# ---------------------------------------------------------------- builders
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins), non-mutating."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _build_physical(raw: dict) -> Config:
    section = raw.get("physical", {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("config section 'physical' must be a mapping")
    overrides = dict(section)
    if "dtype_real" in overrides:
        overrides["dtype_real"] = resolve_dtype(overrides["dtype_real"])
    try:
        # default_config runs Config.validate() (grid==M/L etc.); convert its
        # AssertionError into an explicit ValueError as required by the spec.
        return default_config(**overrides)
    except AssertionError as exc:
        raise ValueError(f"invalid 'physical' config: {exc}") from exc
    except TypeError as exc:
        raise ValueError(f"invalid 'physical' config (unknown field?): {exc}") from exc


def _build_section(cls, raw: dict, key: str):
    section = raw.get(key, None)
    if section is None:
        raise ValueError(f"missing required config section '{key}'")
    if not isinstance(section, dict):
        raise ValueError(f"config section '{key}' must be a mapping, got {type(section).__name__}")
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(section) - field_names)
    if unknown:
        raise ValueError(f"unknown keys in '{key}': {unknown}; allowed: {sorted(field_names)}")
    required = {f.name for f in dataclasses.fields(cls)
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING}
    missing = sorted(required - set(section))
    if missing:
        raise ValueError(f"missing keys in '{key}': {missing}")
    try:
        return cls(**section)
    except TypeError as exc:
        raise ValueError(f"invalid '{key}' config: {exc}") from exc


def _validate_evocomm(cfg: EvoCommConfig) -> None:
    """Explicit cross-field validation; collects every problem into one ValueError."""
    errs: list[str] = []
    phys, d, c, a = cfg.physical, cfg.dynamics, cfg.csi, cfg.actions
    comm, r, rm = cfg.communication, cfg.reward, cfg.rmappo

    # --- CSI error levels ---
    for name, v in (("tau_train_min", c.tau_train_min),
                    ("tau_train_max", c.tau_train_max),
                    ("tau_eval", c.tau_eval)):
        if not (0.0 <= float(v) <= 1.0):
            errs.append(f"csi.{name}={v} must be in [0,1]")
    if c.tau_train_min > c.tau_train_max:
        errs.append(f"csi.tau_train_min={c.tau_train_min} > tau_train_max={c.tau_train_max}")
    if c.actor_delay_steps < 0:
        errs.append(f"csi.actor_delay_steps={c.actor_delay_steps} must be >= 0")
    if c.noise_scaling not in VALID_NOISE_SCALING:
        errs.append(f"csi.noise_scaling={c.noise_scaling!r} not in {VALID_NOISE_SCALING}")
    if c.feature_mode not in ("normal", "masked"):
        errs.append("csi.feature_mode must be normal or masked")
    if rm.comm_entropy_coef is not None and rm.comm_entropy_coef < 0:
        errs.append("rmappo.comm_entropy_coef must be nonnegative")
    if rm.centralized_actor_hidden_size is not None and rm.centralized_actor_hidden_size <= 0:
        errs.append("centralized actor hidden size must be positive")

    # --- dynamics ---
    if not (0.0 <= d.gauss_markov_rho < 1.0):
        errs.append(f"dynamics.gauss_markov_rho={d.gauss_markov_rho} must be in [0,1)")
    if d.episode_length <= 0:
        errs.append(f"dynamics.episode_length={d.episode_length} must be > 0")
    if d.dt_seconds <= 0:
        errs.append(f"dynamics.dt_seconds={d.dt_seconds} must be > 0")
    if d.regime_sampling not in VALID_REGIME_SAMPLING:
        errs.append(f"dynamics.regime_sampling={d.regime_sampling!r} not in {VALID_REGIME_SAMPLING}")
    if d.user_speed_min_mps < 0.0:
        errs.append(f"dynamics.user_speed_min_mps={d.user_speed_min_mps} must be >= 0")
    if d.user_speed_max_mps < d.user_speed_min_mps:
        errs.append(
            f"dynamics.user_speed_max_mps={d.user_speed_max_mps} < user_speed_min_mps={d.user_speed_min_mps}"
        )

    # --- actions ---
    if a.power_codebook_size <= 0:
        errs.append(f"actions.power_codebook_size={a.power_codebook_size} must be > 0")
    if a.ma_update_period <= 0:
        errs.append(f"actions.ma_update_period={a.ma_update_period} must be > 0")
    if a.ris_group_count <= 0:
        errs.append(f"actions.ris_group_count={a.ris_group_count} must be > 0")
    elif phys.L % a.ris_group_count != 0:
        errs.append(
            f"physical.L={phys.L} must be divisible by actions.ris_group_count={a.ris_group_count} (MVP)"
        )
    if a.ris_phase_step_deg <= 0:
        errs.append(f"actions.ris_phase_step_deg={a.ris_phase_step_deg} must be > 0")
    if not (0.0 < a.ma_step_fraction_of_cell <= 1.0):
        errs.append(f"actions.ma_step_fraction_of_cell={a.ma_step_fraction_of_cell} must be in (0,1]")

    # --- communication ---
    if comm.graph_type not in VALID_GRAPH_TYPES:
        errs.append(f"communication.graph_type={comm.graph_type!r} not in {VALID_GRAPH_TYPES}")
    if comm.max_messages_per_step < 0:
        errs.append(f"communication.max_messages_per_step={comm.max_messages_per_step} must be >= 0")
    if comm.message_dim <= 0:
        errs.append(f"communication.message_dim={comm.message_dim} must be > 0")
    if comm.delivery_delay_steps < 0:
        errs.append(f"communication.delivery_delay_steps={comm.delivery_delay_steps} must be >= 0")

    # --- rmappo ---
    for name, v in (("total_timesteps", rm.total_timesteps), ("num_envs", rm.num_envs),
                    ("rollout_length", rm.rollout_length), ("update_epochs", rm.update_epochs),
                    ("num_minibatches", rm.num_minibatches), ("hidden_size", rm.hidden_size),
                    ("sequence_length", rm.sequence_length)):
        if v <= 0:
            errs.append(f"rmappo.{name}={v} must be > 0")
    if rm.sequence_length > rm.rollout_length:
        errs.append(
            f"rmappo.sequence_length={rm.sequence_length} must be <= rollout_length={rm.rollout_length}"
        )
    if rm.learning_rate <= 0:
        errs.append(f"rmappo.learning_rate={rm.learning_rate} must be > 0")
    if not (0.0 < rm.gamma <= 1.0):
        errs.append(f"rmappo.gamma={rm.gamma} must be in (0,1]")
    if not (0.0 <= rm.gae_lambda <= 1.0):
        errs.append(f"rmappo.gae_lambda={rm.gae_lambda} must be in [0,1]")

    if errs:
        raise ValueError("invalid EvoCommConfig:\n  - " + "\n  - ".join(errs))


# ---------------------------------------------------------------- public loader
def load_evocomm_config(path: str | pathlib.Path, overrides: dict | None = None) -> EvoCommConfig:
    """Load and validate an :class:`EvoCommConfig` from a YAML file.

    ``overrides`` is a nested dict deep-merged over the file contents (so partial
    overrides like ``{"dynamics": {"episode_length": 32}}`` work). Every failure
    raises ``ValueError`` (or ``FileNotFoundError`` for a missing path). The global
    momgrap config is never mutated — each call builds a fresh ``Config``.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")
    if overrides:
        if not isinstance(overrides, dict):
            raise ValueError("overrides must be a dict")
        raw = _deep_merge(raw, overrides)

    cfg = EvoCommConfig(
        physical=_build_physical(raw),
        dynamics=_build_section(DynamicsConfig, raw, "dynamics"),
        csi=_build_section(CSIConfig, raw, "csi"),
        actions=_build_section(ActionConfig, raw, "actions"),
        communication=_build_section(CommunicationConfig, raw, "communication"),
        reward=_build_section(RewardConfig, raw, "reward"),
        rmappo=_build_section(RMAPPOConfig, raw, "rmappo"),
    )
    _validate_evocomm(cfg)
    return cfg


# ---------------------------------------------------------------- serialisation
def _physical_to_dict(phys: Config) -> dict:
    out: dict[str, Any] = {}
    for f in dataclasses.fields(phys):
        v = getattr(phys, f.name)
        if f.name == "dtype_real":
            v = dtype_to_str(v)
        elif f.name == "ris_position":
            v = list(v)
        elif f.name == "regimes":
            v = [{"name": r.name, "k_factor_db": r.k_factor_db} for r in v]
        out[f.name] = v
    return out


def evocomm_config_to_dict(cfg: EvoCommConfig) -> dict:
    """Return a plain, YAML/JSON-serialisable nested dict of the resolved config."""
    return {
        "physical": _physical_to_dict(cfg.physical),
        "dynamics": dataclasses.asdict(cfg.dynamics),
        "csi": dataclasses.asdict(cfg.csi),
        "actions": dataclasses.asdict(cfg.actions),
        "communication": dataclasses.asdict(cfg.communication),
        "reward": dataclasses.asdict(cfg.reward),
        "rmappo": dataclasses.asdict(cfg.rmappo),
    }


def save_resolved_config(cfg: EvoCommConfig, path: str | pathlib.Path) -> str:
    """Dump the resolved config to YAML (for run provenance). Returns the path."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(evocomm_config_to_dict(cfg), f, sort_keys=False)
    return str(path)

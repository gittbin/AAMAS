"""Reuse the offline MO-MFEA archive to seed the MARL episode's MA/RIS layout.

Runs the existing ``run_mfea`` optimiser, takes the unified robust Pareto front (via
``archive.unified_robust_front``) — front + genotypes — and selects meaningful
starting points (knee, communication-extreme, sensing-extreme, evenly-spaced). The
selected configs are decoded to feasible MA positions / RIS phases and exported.

We deliberately do NOT read genotypes from ``results/experiment.pkl`` (that file
stores objective fronts, not a guaranteed genotype archive).
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import torch

from ...config import Config
from ...genotype import decode
from ...mfea import run_mfea
from ..config import _physical_to_dict

RAMP_MODES = ("ramp_knee", "ramp_comm_extreme", "ramp_sensing_extreme", "ramp_random_front_point")


@dataclass
class RampInitialization:
    seed: int
    M: int
    L: int
    K: int
    Q: int
    front: torch.Tensor            # (P,2) robust objectives [F_com, F_sen_db]
    genotypes: torch.Tensor        # (P,D)
    positions: torch.Tensor        # (P,M,3) decoded
    theta: torch.Tensor            # (P,L) decoded
    selected_indices: dict         # {"knee":i, "comm_extreme":j, "sensing_extreme":k}
    spaced_indices: list           # evenly-spaced indices along the front
    physical_config: dict          # serialized Config (provenance)

    @property
    def n_points(self) -> int:
        return self.front.shape[0]


# ---------------------------------------------------------------- selection
def select_max_comm(front: torch.Tensor) -> int:
    return int(front[:, 0].argmax())


def select_max_sensing(front: torch.Tensor) -> int:
    return int(front[:, 1].argmax())


def select_knee(front: torch.Tensor) -> int:
    """Elbow point: max perpendicular distance to the chord between the two extremes."""
    P = front.shape[0]
    f = front.to(torch.float64)
    mn = f.min(0).values
    rng = (f.max(0).values - mn).clamp_min(1e-12)
    fn = (f - mn) / rng                                   # normalise both objectives to [0,1]
    if P <= 2:
        return int((fn[:, 0] + fn[:, 1]).argmax())       # most balanced point
    a = fn[int(fn[:, 0].argmax())]
    b = fn[int(fn[:, 1].argmax())]
    ab = b - a
    denom = ab.norm().clamp_min(1e-12)
    ap = fn - a
    cross = (ab[0] * ap[:, 1] - ab[1] * ap[:, 0]).abs() / denom
    return int(cross.argmax())


def select_evenly_spaced(front: torch.Tensor, n: int) -> list[int]:
    P = front.shape[0]
    if n >= P:
        return list(range(P))
    order = torch.argsort(front[:, 0])
    picks = torch.linspace(0, P - 1, n).round().long()
    return [int(order[i]) for i in picks]


# ---------------------------------------------------------------- build / io
def build_ramp_initializations(
    cfg: Config, seed: int = 0, s_eval: int | None = None, n_spaced: int = 5
) -> RampInitialization:
    cfg.validate()
    res = run_mfea(cfg, seed=seed)
    gen = torch.Generator(device=cfg.device)
    gen.manual_seed(int(seed) + 777)
    front, genos = res.archive.unified_robust_front(gen, s_eval or cfg.mc_samples)
    if front.shape[0] == 0:
        raise RuntimeError("MO-MFEA produced an empty robust front")

    positions, theta = decode(genos, cfg)                # (P,M,3), (P,L)
    selected = {
        "knee": select_knee(front),
        "comm_extreme": select_max_comm(front),
        "sensing_extreme": select_max_sensing(front),
    }
    return RampInitialization(
        seed=int(seed), M=cfg.M, L=cfg.L, K=cfg.K, Q=cfg.Q,
        front=front.detach().cpu(), genotypes=genos.detach().cpu(),
        positions=positions.detach().cpu(), theta=theta.detach().cpu(),
        selected_indices=selected, spaced_indices=select_evenly_spaced(front, n_spaced),
        physical_config=_physical_to_dict(cfg),
    )


def check_compatibility(init: RampInitialization, cfg: Config) -> None:
    """Raise if the initialization geometry does not match the target config."""
    for name, a, b in (("M", init.M, cfg.M), ("L", init.L, cfg.L),
                       ("K", init.K, cfg.K), ("Q", init.Q, cfg.Q)):
        if a != b:
            raise ValueError(f"RAMP init {name}={a} incompatible with config {name}={b}")


def initial_config_from(
    init: RampInitialization, mode: str, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(positions (1,M,3), theta (1,L))`` for a RAMP initialization mode."""
    if mode == "ramp_knee":
        idx = init.selected_indices["knee"]
    elif mode == "ramp_comm_extreme":
        idx = init.selected_indices["comm_extreme"]
    elif mode == "ramp_sensing_extreme":
        idx = init.selected_indices["sensing_extreme"]
    elif mode == "ramp_random_front_point":
        if generator is None:
            idx = 0
        else:
            idx = int(torch.randint(0, init.n_points, (1,), generator=generator).item())
    else:
        raise ValueError(f"unknown RAMP mode {mode!r}; expected {RAMP_MODES}")
    return init.positions[idx:idx + 1].clone(), init.theta[idx:idx + 1].clone()


def save_initialization(init: RampInitialization, path_stem: str | pathlib.Path) -> tuple[str, str]:
    stem = pathlib.Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pt_path = stem.with_suffix(".pt")
    json_path = stem.with_suffix(".json")

    torch.save({
        "physical_config": init.physical_config,
        "seed": init.seed, "M": init.M, "L": init.L, "K": init.K, "Q": init.Q,
        "front": init.front, "genotypes": init.genotypes,
        "positions": init.positions, "theta": init.theta,
        "selected_indices": init.selected_indices, "spaced_indices": init.spaced_indices,
    }, pt_path)

    json_path.write_text(json.dumps({
        "seed": init.seed, "M": init.M, "L": init.L, "K": init.K, "Q": init.Q,
        "physical_config": init.physical_config,
        "n_points": init.n_points,
        "front": init.front.tolist(),
        "selected_indices": init.selected_indices,
        "selected_objectives": {k: init.front[v].tolist() for k, v in init.selected_indices.items()},
        "spaced_indices": init.spaced_indices,
    }, indent=2), encoding="utf-8")
    return str(pt_path), str(json_path)


def load_initialization(pt_path: str | pathlib.Path) -> RampInitialization:
    # our own artifact -> weights_only=False (contains a plain config dict alongside tensors)
    p = torch.load(pt_path, weights_only=False)
    return RampInitialization(
        seed=p["seed"], M=p["M"], L=p["L"], K=p["K"], Q=p["Q"],
        front=p["front"], genotypes=p["genotypes"],
        positions=p["positions"], theta=p["theta"],
        selected_indices=p["selected_indices"], spaced_indices=p["spaced_indices"],
        physical_config=p["physical_config"],
    )

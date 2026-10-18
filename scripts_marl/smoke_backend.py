"""Backend smoke: exercise every physical building block the MARL layer reuses.

Runs the exact forward chain the future environment will call
(decode -> build_comm_channels -> bd_precoding -> power_allocation ->
evm_sinr_rate -> effective_steering -> beampattern) on a tiny population and
prints the resulting tensor shapes. Both power policies are shown so the
``power_allocation`` return contract (None for "equal", (N,S,K) for "inv_gain")
is visible.

Run: ``python scripts_marl/smoke_backend.py``
"""

from __future__ import annotations

import pathlib
import sys

# Make the repository root importable regardless of the invocation directory.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Windows consoles default to cp1252; force UTF-8 so unicode prints don't crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch

from momgrap import smoke_config
from momgrap.channels import build_comm_channels, sample_environment
from momgrap.comm import bd_precoding, evm_sinr_rate, power_allocation
from momgrap.genotype import decode
from momgrap.sensing import beampattern, effective_steering


def _shape(x) -> str:
    if x is None:
        return "None"
    return f"{tuple(x.shape)} {str(x.dtype).replace('torch.', '')} {x.device}"


def _run_once(cfg, label: str, n: int = 4) -> None:
    print(f"\n--- backend chain [{label}]  device={cfg.device}  "
          f"M={cfg.M} L={cfg.L} K={cfg.K} Q={cfg.Q} S={cfg.mc_samples} ---")

    g = torch.Generator(device=cfg.device)
    g.manual_seed(0)
    pop = torch.rand(n, cfg.D, generator=g, device=cfg.device, dtype=cfg.dtype_real)

    P_ant, theta = decode(pop, cfg)
    env = sample_environment(cfg.regimes[0], cfg.mc_samples, g, cfg)  # LoS

    Hh = build_comm_channels(P_ant, theta, env, cfg)
    W = bd_precoding(Hh, cfg)
    p_user = power_allocation(Hh, W, cfg)
    R = evm_sinr_rate(Hh, W, cfg, p_user)

    a_eff = effective_steering(P_ant, theta, cfg, env.sector_center_deg)
    B = beampattern(a_eff, W, cfg, p_user)

    print(f"  decode.P_ant        : {_shape(P_ant)}")
    print(f"  decode.theta        : {_shape(theta)}")
    print(f"  build_comm_channels : {_shape(Hh)}")
    print(f"  bd_precoding W      : {_shape(W)}")
    print(f"  power_allocation    : {_shape(p_user)}  (policy={cfg.power_policy})")
    print(f"  evm_sinr_rate R     : {_shape(R)}   "
          f"min={float(R.min()):.3f} mean={float(R.mean()):.3f} bps/Hz")
    print(f"  effective_steering  : {_shape(a_eff)}")
    print(f"  beampattern B       : {_shape(B)}   "
          f"min={float(B.min()):.3e} mean={float(B.mean()):.3e}")

    assert torch.isfinite(Hh).all() and torch.isfinite(W).all()
    assert torch.isfinite(R).all() and torch.isfinite(B).all()
    if p_user is not None:
        assert torch.isfinite(p_user).all()


def main() -> None:
    cfg_equal = smoke_config()                          # power_policy="equal" -> None
    cfg_inv = smoke_config(power_policy="inv_gain")     # -> (N,S,K) allocation

    _run_once(cfg_equal, "equal power")
    _run_once(cfg_inv, "inverse-gain power")

    print("\nBACKEND SMOKE OK")


if __name__ == "__main__":
    main()

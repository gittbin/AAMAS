"""Temporal channel dynamics: turn the momgrap ``Environment`` (an i.i.d. MC
snapshot) into a time-correlated single-realisation (S=1) process.

Two coupled dynamics, both seeded from a per-episode generator (never the global
RNG):

* **User mobility** — constant-velocity motion ``p_{t+1} = p_t + v_t dt`` in the
  z=0 plane, kept inside the radial band ``[r_min, r_max]`` and the angular sector
  by velocity-reflection (plus a final clamp for a hard region guarantee). After
  moving the users, the dependent geometry fields (``user_dir``/``user_amp``/
  ``g_los_dir``/``g_amp``) are recomputed so nothing contradicts ``user_pos``.
* **Small-scale fading** — a Gauss–Markov walk ``X_{t+1} = rho X_t +
  sqrt(1-rho^2) eps`` on the complex fading tensors (``scat_gain_bu``, ``G_nlos``,
  ``g_nlos``); the CN(0,1) marginal is preserved so statistics stay stationary.

Fixed within an episode: the regime, the scatter *directions*, and the sensing
target geometry (config-driven). ``advance`` never mutates its input state.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from ...channels import Environment, _cn, _pathloss_amp, sample_environment
from ...config import Config
from ..config import DynamicsConfig
from ..types import DynamicPhysicalState

_FIXED_REGIME_NAME = {
    "fixed_los": "LoS",
    "fixed_rayleigh": "Rayleigh",
    "fixed_rician": "Rician",
}
_FIXED_REGIME_FALLBACK = {"fixed_los": 0, "fixed_rayleigh": 1, "fixed_rician": 2}


class DynamicChannelProcess:
    """Per-episode temporal process over a momgrap ``Environment`` (S=1)."""

    def __init__(self, cfg: Config, dynamics: DynamicsConfig):
        self.cfg = cfg
        self.dyn = dynamics
        self._gen: torch.Generator | None = None
        self._fixed_scatter: torch.Tensor | None = None

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        seed: int,
        initial_positions: torch.Tensor,
        initial_theta: torch.Tensor,
        initial_power: torch.Tensor | None = None,
    ) -> DynamicPhysicalState:
        cfg = self.cfg
        gen = torch.Generator(device=cfg.device)
        gen.manual_seed(int(seed))
        self._gen = gen

        regime_index = self._resolve_regime_index(gen)
        env = sample_environment(cfg.regimes[regime_index], 1, gen, cfg)  # S=1

        if not self.dyn.resample_scatter_directions_each_episode:
            if self._fixed_scatter is None:
                self._fixed_scatter = env.scat_dir_bu
            else:
                env = dataclasses.replace(
                    env, scat_dir_bu=self._fixed_scatter.to(env.scat_dir_bu.device)
                )

        velocity = self._init_velocity(gen)

        positions = torch.as_tensor(initial_positions, dtype=cfg.dtype_real, device=cfg.device)
        theta = torch.as_tensor(initial_theta, dtype=cfg.dtype_real, device=cfg.device)
        if initial_power is None:
            power_alloc = torch.full(
                (1, 1, cfg.K), cfg.p_per_user, dtype=cfg.dtype_real, device=cfg.device
            )
        else:
            power_alloc = torch.as_tensor(initial_power, dtype=cfg.dtype_real, device=cfg.device)

        return DynamicPhysicalState(
            step_index=0,
            regime_index=regime_index,
            environment=env,
            user_velocity=velocity,
            positions=positions,
            theta=theta,
            power_alloc=power_alloc,
            previous_physical_actions={},
        )

    # ---------------------------------------------------------------- advance
    def advance(self, state: DynamicPhysicalState) -> DynamicPhysicalState:
        if self._gen is None:
            raise RuntimeError("call reset() before advance()")
        cfg = self.cfg
        gen = self._gen
        env = state.environment
        rho = float(self.dyn.gauss_markov_rho)

        # --- user mobility + dependent geometry recompute ---
        if self.dyn.mobility_enabled:
            new_pos, new_vel = self._advance_users(
                env.user_pos, state.user_velocity, env.sector_center_deg
            )
            user_dir, user_amp, g_los_dir, g_amp = self._recompute_geometry(new_pos)
        else:
            new_pos, new_vel = env.user_pos, state.user_velocity
            user_dir, user_amp = env.user_dir, env.user_amp
            g_los_dir, g_amp = env.g_los_dir, env.g_amp

        # --- Gauss-Markov small-scale fading (directions stay fixed) ---
        scat_gain_bu = self._gauss_markov(env.scat_gain_bu, rho, gen)
        G_nlos = self._gauss_markov(env.G_nlos, rho, gen)
        g_nlos = self._gauss_markov(env.g_nlos, rho, gen)

        new_env = dataclasses.replace(
            env,
            user_pos=new_pos,
            user_dir=user_dir,
            user_amp=user_amp,
            g_los_dir=g_los_dir,
            g_amp=g_amp,
            scat_gain_bu=scat_gain_bu,
            G_nlos=G_nlos,
            g_nlos=g_nlos,
        )

        return DynamicPhysicalState(
            step_index=state.step_index + 1,
            regime_index=state.regime_index,
            environment=new_env,
            user_velocity=new_vel,
            positions=state.positions,
            theta=state.theta,
            power_alloc=state.power_alloc,
            previous_physical_actions=dict(state.previous_physical_actions),
        )

    # ------------------------------------------------------------- internals
    def _resolve_regime_index(self, gen: torch.Generator) -> int:
        s = self.dyn.regime_sampling
        if s == "uniform_episode":
            return int(torch.randint(0, self.cfg.T, (1,), generator=gen, device=self.cfg.device).item())
        if s not in _FIXED_REGIME_NAME:
            raise ValueError(f"unknown regime_sampling {s!r}")
        name = _FIXED_REGIME_NAME[s]
        for i, reg in enumerate(self.cfg.regimes):
            if reg.name == name:
                return i
        return _FIXED_REGIME_FALLBACK[s]

    def _init_velocity(self, gen: torch.Generator) -> torch.Tensor:
        cfg, dyn = self.cfg, self.dyn
        K = cfg.K
        if not dyn.mobility_enabled:
            return torch.zeros(K, 3, dtype=cfg.dtype_real, device=cfg.device)
        span = dyn.user_speed_max_mps - dyn.user_speed_min_mps
        speed = dyn.user_speed_min_mps + torch.rand(
            K, generator=gen, dtype=cfg.dtype_real, device=cfg.device
        ) * span
        ang = torch.rand(K, generator=gen, dtype=cfg.dtype_real, device=cfg.device) * (2.0 * math.pi)
        vx = speed * torch.cos(ang)
        vy = speed * torch.sin(ang)
        vz = torch.zeros_like(vx)
        return torch.stack([vx, vy, vz], dim=-1)

    def _advance_users(
        self, user_pos: torch.Tensor, velocity: torch.Tensor, sector_center_deg: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg, dyn = self.cfg, self.dyn
        pos = user_pos[0]                                   # (K,3), S=1
        tentative = pos + velocity * dyn.dt_seconds         # (K,3)
        x, y = tentative[:, 0], tentative[:, 1]
        r = torch.sqrt(x * x + y * y).clamp_min(1e-9)
        az = torch.atan2(y, x)

        hw = math.radians(cfg.user_sector_width_deg) / 2.0
        az_c = math.radians(sector_center_deg)
        r_ref, r_flip = self._reflect(r, cfg.user_radius_min, cfg.user_radius_max)
        az_ref, az_flip = self._reflect(az, az_c - hw, az_c + hw)

        # decompose the incoming velocity into radial/tangential and bounce
        cos_a, sin_a = torch.cos(az), torch.sin(az)
        zeros = torch.zeros_like(cos_a)
        rhat = torch.stack([cos_a, sin_a, zeros], dim=-1)     # (K,3)
        that = torch.stack([-sin_a, cos_a, zeros], dim=-1)    # (K,3)
        v_r = (velocity * rhat).sum(-1)
        v_t = (velocity * that).sum(-1)
        v_r = torch.where(r_flip, -v_r, v_r)
        v_t = torch.where(az_flip, -v_t, v_t)
        new_vel = v_r.unsqueeze(-1) * rhat + v_t.unsqueeze(-1) * that

        cos_ar, sin_ar = torch.cos(az_ref), torch.sin(az_ref)
        new_pos = torch.stack(
            [r_ref * cos_ar, r_ref * sin_ar, torch.zeros_like(r_ref)], dim=-1
        )[None]                                               # (1,K,3)
        return new_pos, new_vel

    @staticmethod
    def _reflect(val: torch.Tensor, lo: float, hi: float) -> tuple[torch.Tensor, torch.Tensor]:
        below = val < lo
        above = val > hi
        val = torch.where(above, 2.0 * hi - val, val)
        val = torch.where(below, 2.0 * lo - val, val)
        val = torch.clamp(val, lo, hi)          # hard region guarantee
        return val, (below | above)

    def _recompute_geometry(
        self, user_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        device = user_pos.device
        # BS at origin: direct link
        r = torch.linalg.norm(user_pos, dim=-1)                     # (1,K)
        user_dir = user_pos / r.clamp_min(1e-6).unsqueeze(-1)       # (1,K,3)
        user_amp = _pathloss_amp(r, cfg)                            # (1,K)
        # RIS -> user link (float64 geometry as in sample_environment)
        p_ris = torch.tensor(cfg.ris_position, dtype=torch.float64, device=device)
        ris_to_user = user_pos.to(torch.float64) - p_ris            # (1,K,3)
        d_ru = torch.linalg.norm(ris_to_user, dim=-1)              # (1,K)
        g_los_dir = (ris_to_user / d_ru.clamp_min(1e-6).unsqueeze(-1)).to(user_dir.dtype)
        g_amp = _pathloss_amp(d_ru, cfg).to(user_amp.dtype)
        return user_dir, user_amp, g_los_dir, g_amp

    @staticmethod
    def _gauss_markov(x: torch.Tensor, rho: float, gen: torch.Generator) -> torch.Tensor:
        eps = _cn(x.shape, gen, x.device, x.dtype)
        return rho * x + math.sqrt(max(0.0, 1.0 - rho * rho)) * eps

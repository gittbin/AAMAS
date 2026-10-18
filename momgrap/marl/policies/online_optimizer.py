"""Online (myopic) re-optimizer baseline.

A strong *non-learning adaptive* baseline: at every control interval it re-selects the
power / RIS / MA action that maximizes the *estimated* one-step scalar objective, using
only a delayed, imperfect CSI estimate (never the realized true rate). It searches the
**same discrete, incremental action space** the RL agents use---one increment per step per
actuator---by coordinate ascent, so the comparison isolates *learning a temporal policy*
from *greedily re-optimizing each step*.

Fairness: candidates are ranked by the objective computed on the *estimated* channel
(the optimizer's belief); the environment then realizes the chosen action on the *true*
channel, so the optimizer pays the same CSI-mismatch penalty the learned policy does. It
uses a private RNG so it never perturbs the environment's own stochastic stream. The
message cost is excluded (it sends nothing); reconfiguration and constraint penalties are
included, so it will not thrash the antenna panel needlessly.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ...comm import bd_precoding, evm_sinr_rate
from ...sensing import beampattern, effective_steering
from ..actions.codebooks import (
    ma_action_count,
    power_action_count,
    ris_action_count,
)
from ..actions.decoder import decode_ma_action, decode_power_action, decode_ris_action
from ..actions.safety import project_to_safety
from ..envs.reward import ObjectiveVector, compute_reward
from ..physics.backend import _DB_EPS
from ..physics.surrogate import candidate_estimate
from .baselines import Policy


class OnlineOptimizerPolicy(Policy):
    """Greedy per-step re-optimizer over the discrete action codebooks (estimate-only)."""

    name = "online_optimizer"

    def __init__(self, seed: int = 0, rounds: int = 2):
        self._seed = int(seed)
        self.rounds = int(rounds)
        self._gen = None  # created lazily on the env device in act()
        self.last_candidate_evaluations = 0

    def reset(self) -> None:
        self._gen = None

    def _ensure_gen(self, device) -> None:
        if self._gen is None or self._gen.device != torch.device(device):
            self._gen = torch.Generator(device=device).manual_seed(self._seed)

    # ---------------------------------------------------------------- scoring
    def _tau(self, env) -> float:
        # Deterministic in eval_mode (tau_eval); honor an explicit override otherwise.
        ov = getattr(env, "_tau_override", None)
        if ov is not None:
            return float(ov)
        return float(env.evo.csi.tau_eval)

    def _estimated_reward(self, env, tau, power_idx, ris_idx, ma_idx, ma_allowed) -> float:
        cfg, evo = env.cfg, env.evo
        positions, theta = env.decision_configuration()
        theta_cand, n_changed = decode_ris_action(theta, ris_idx, env.group_map, env.phase_step)
        if ma_allowed:
            pos_cand = decode_ma_action(positions, ma_idx, cfg, evo.actions)
        else:
            pos_cand = positions
        initial_power = torch.full((1, 1, cfg.K), cfg.p_per_user, device=positions.device, dtype=positions.dtype)
        proj = project_to_safety(pos_cand, theta_cand, initial_power, cfg,
                                 previous_positions=positions)
        positions_new, theta_new = proj.positions, proj.theta
        viol_before = (proj.boundary_violation_before
                       + max(0.0, cfg.d_min - proj.min_spacing_before)
                       + proj.power_violation_before)

        est = candidate_estimate(env.current_estimate(), positions_new, theta_new, cfg)
        W = bd_precoding(est.aggregate, cfg)
        p_user = decode_power_action(power_idx, est.aggregate, W, cfg, evo.actions)
        # Rates on the ESTIMATE (belief), never the truth.
        rates = evm_sinr_rate(est.aggregate, W, cfg, p_user)[0, 0]        # (K,)
        a_eff = effective_steering(positions_new, theta_new, cfg, cfg.user_sector_center_deg)
        B = beampattern(a_eff, W, cfg, p_user)[0, 0]                      # (Q,)
        min_sens_db = float(10.0 * torch.log10(B.min() + _DB_EPS))

        ma_disp = (float((positions_new - positions).norm(dim=-1).mean()) / max(env._half, 1e-12)
                   if ma_allowed else 0.0)
        reconfig = ma_disp + n_changed / max(1, evo.actions.ris_group_count)
        obj = ObjectiveVector(
            min_user_rate=float(rates.min()), sum_rate=float(rates.sum()),
            min_sensing_illumination_db=min_sens_db,
            outage_count=int((rates < evo.reward.rate_target_bpshz).sum()),
            delivered_messages=0, reconfiguration_cost=reconfig,
            constraint_violation_before=viol_before,
        )
        r, _ = compute_reward(obj, evo.reward)
        if not math.isfinite(r):
            return -1e9
        return r

    # ---------------------------------------------------------------- act
    def act(self, observations, env) -> dict:
        self._ensure_gen(env.device)
        tau = self._tau(env)
        n_pow = power_action_count(env.evo.actions)
        n_ris = ris_action_count(env.evo.actions)
        n_ma = ma_action_count(env.evo.actions)
        ma_allowed = (env.step_count % env.evo.actions.ma_update_period == 0)

        best = [0, 0, 0]  # [power, ris, ma], all no-op to start
        counts = [n_pow, n_ris, (n_ma if ma_allowed else 1)]
        self.last_candidate_evaluations = 0

        def score(cand):
            self.last_candidate_evaluations += 1
            return self._estimated_reward(env, tau, cand[0], cand[1], cand[2], ma_allowed)

        best_score = score(best)
        for _ in range(self.rounds):
            improved = False
            for dim in range(3):
                if dim == 2 and not ma_allowed:
                    continue
                for idx in range(counts[dim]):
                    if idx == best[dim]:
                        continue
                    cand = list(best)
                    cand[dim] = idx
                    s = score(cand)
                    if s > best_score:
                        best_score, best[dim], improved = s, idx, True
            if not improved:
                break

        out = {}
        for a, phys in zip(("bs_agent", "ris_agent", "ma_agent"), best):
            if a in env.agents:
                out[a] = np.array([phys, 0], dtype=np.int64)
        return out

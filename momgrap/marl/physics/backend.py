"""One control-interval physical backend for the MARL environment.

Packages the exact forward chain for a single control step *without* calling
``evaluate_single_regime`` (which aggregates over S MC snapshots and hides the
true/estimated CSI split). The imperfect-CSI timing is load-bearing:

    true   = build_comm_channel_components(post-action positions/theta, env)   # step 4
    est    = estimate_channel_components(true, tau, generator)                 # step 5 (current CSI)
    W      = bd_precoding(est.aggregate, cfg)                                   # step 6 (BD on ESTIMATE)
    p_user = decode_power_action(action, est.aggregate, W, cfg, actions)       # step 6
    rates  = evm_sinr_rate(true.aggregate, W, cfg, p_user)                     # step 7-8 (rate on TRUTH)
    B      = beampattern(effective_steering(...), W, cfg, p_user)              # step 9 (same W)

Never: ``bd_precoding(true.aggregate)`` by default, estimated channel for realized
rates, ``Config.power_policy`` as the action, or MC-mean aggregation (S=1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ...channels import Environment, build_comm_channel_components
from ...comm import bd_precoding, evm_sinr_rate
from ...config import Config
from ...sensing import beampattern, effective_steering
from ..actions.decoder import decode_power_action
from .components import CommChannelComponents
from .csi import EstimatedChannels, estimate_channel_components

# Match momgrap.objectives._DB_EPS so the sensing dB is comparable to the evaluator.
_DB_EPS = 1e-12


@dataclass
class PhysicalStepResult:
    true_components: CommChannelComponents
    estimated_components: EstimatedChannels
    precoder: torch.Tensor            # W (1,1,M,K)
    p_user: torch.Tensor              # (1,1,K)
    user_rates: torch.Tensor          # (K,) realized on the TRUE channel
    min_user_rate: float
    sum_rate: float
    sensing_per_target: torch.Tensor  # (Q,) beampattern illumination (linear)
    min_sensing_illumination_db: float
    outage_count: int


def evaluate_control_interval(
    environment: Environment,
    positions: torch.Tensor,
    theta: torch.Tensor,
    power_action: int,
    tau: float,
    cfg: Config,
    marl_cfg,
    generator: torch.Generator,
) -> PhysicalStepResult:
    """Evaluate one control interval; returns raw physical metrics (no reward shaping).

    ``positions`` (1,M,3) and ``theta`` (1,L) are the post-action, safety-projected
    physical config; ``power_action`` is the BS power codebook index. ``environment``
    is an S=1 snapshot (the current step of the dynamic process).
    """
    # 4. post-action TRUE channel components
    true_components = build_comm_channel_components(positions, theta, environment, cfg)

    # 5. current imperfect CSI for the low-level BD stage
    estimated = estimate_channel_components(
        true_components, tau, generator, marl_cfg.csi.noise_scaling,
        positions=positions, theta=theta,
    )

    # 6. BD precoder from the ESTIMATE (never the truth) + power action from estimate
    W = bd_precoding(estimated.aggregate, cfg)
    p_user = decode_power_action(power_action, estimated.aggregate, W, cfg, marl_cfg.actions)

    # 7-8. realized communication rate on the TRUE channel
    rates = evm_sinr_rate(true_components.aggregate, W, cfg, p_user)      # (1,1,K)

    # 9. sensing illumination from the SAME precoder / power
    a_eff = effective_steering(positions, theta, cfg, environment.sector_center_deg)
    B = beampattern(a_eff, W, cfg, p_user)                               # (1,1,Q)

    # 10. raw metrics (S=1: no MC mean)
    user_rates = rates[0, 0]                                             # (K,)
    sensing_per_target = B[0, 0]                                         # (Q,)
    min_sens_lin = sensing_per_target.min()
    min_sensing_illumination_db = float(10.0 * torch.log10(min_sens_lin + _DB_EPS))

    rate_target = float(marl_cfg.reward.rate_target_bpshz)
    outage_count = int((user_rates < rate_target).sum())

    return PhysicalStepResult(
        true_components=true_components,
        estimated_components=estimated,
        precoder=W,
        p_user=p_user,
        user_rates=user_rates,
        min_user_rate=float(user_rates.min()),
        sum_rate=float(user_rates.sum()),
        sensing_per_target=sensing_per_target,
        min_sensing_illumination_db=min_sensing_illumination_db,
        outage_count=outage_count,
    )

"""Candidate-channel surrogate from delayed estimates and public geometry only.

Direct-path column phases use fixed reference user bearings spanning the configured
sector. The reflected path uses the known BS--RIS bearing and a uniform-element
RIS phase approximation. It cannot query hypothetical true simulator channels.
"""
import math

import torch

from .csi import EstimatedChannels


def candidate_estimate(estimate, positions, theta, cfg) -> EstimatedChannels:
    if estimate.reference_positions is None or estimate.reference_theta is None:
        raise ValueError("surrogate requires the estimate's acquisition geometry")
    delta = positions[0] - estimate.reference_positions[0]
    angles = torch.linspace(-0.5, 0.5, cfg.K, device=positions.device, dtype=positions.dtype)
    angles = (cfg.user_sector_center_deg + cfg.user_sector_width_deg * angles) * math.pi / 180
    user_dirs = torch.stack((angles.cos(), angles.sin(), torch.zeros_like(angles)), dim=-1)
    # Steering convention follows channels._steering: exp(+j k0 P . direction).
    direct_phase = torch.exp(1j * cfg.k0 * (user_dirs @ delta.T))
    ris_dir = torch.as_tensor(cfg.ris_position, device=positions.device, dtype=positions.dtype)
    ris_dir = ris_dir / ris_dir.norm().clamp_min(1e-12)
    reflected_phase = torch.exp(1j * cfg.k0 * (delta @ ris_dir))
    phase_ratio = torch.exp(1j * (theta - estimate.reference_theta)).mean()
    direct = estimate.direct * direct_phase
    reflected = estimate.reflected * reflected_phase * phase_ratio
    return EstimatedChannels(direct, reflected, direct + reflected, estimate.tau,
                             positions.clone(), theta.clone(), estimate.step_index)

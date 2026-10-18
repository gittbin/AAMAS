"""Overflow-only inverse-gain normalization, preserving every finite legacy result.

The original gain clamp, codebook, power floor and physical model are unchanged.
Float64 can represent gain**(-2) at the existing float32 lower bound of 1e-30.
This correction is confined to the exploratory perfect-CSI grid, after final
opening. Training, checkpoint selection and primary comparisons are untouched.
"""
from __future__ import annotations
import types
import torch
from experiments_paper_completion.engine import env_for as original_env_for
from momgrap.marl.actions.decoder import decode_power_action as legacy_decode, _enforce_floor
from momgrap.marl.actions.codebooks import power_profile_specs, power_action_count

VERSION = "perfect-csi-inverse-power-overflow-erratum-1"


def corrected_decoder(events):
    def decode(action_index, H_est, W_est, cfg, action_cfg, min_power_fraction=0.05):
        result = legacy_decode(action_index, H_est, W_est, cfg, action_cfg, min_power_fraction)
        if torch.isfinite(result).all():
            return result
        if not torch.isfinite(H_est).all() or not torch.isfinite(W_est).all():
            raise RuntimeError("Numerical erratum cannot repair a nonfinite channel or precoder")
        gain = (torch.einsum("nskm,nsmk->nsk", H_est, W_est).abs().square()
                .mean((0, 1)).clamp_min(1e-30).to(cfg.dtype_real))
        kind, *parameters = power_profile_specs(int(cfg.K), power_action_count(action_cfg))[action_index]
        if not torch.isfinite(gain).all() or kind not in ("invgain", "blend"):
            raise RuntimeError("Nonfinite output outside the declared normalization correction")
        alpha = float(parameters[0]) if kind == "invgain" else 1.0
        inv = gain.double().pow(-alpha)
        weights = inv / inv.sum()
        if kind == "blend":
            weights = 0.5 / int(cfg.K) + 0.5 * weights
        weights = weights.to(cfg.dtype_real)
        pmax = float(cfg.p_max_watt)
        result = _enforce_floor(weights * pmax, pmax, float(min_power_fraction) * pmax / int(cfg.K)).reshape(1, 1, int(cfg.K))
        if not torch.isfinite(result).all():
            raise RuntimeError("Overflow correction failed")
        events.append({"power_action":int(action_index), "gain":gain.cpu().tolist(),
                       "inverse_exponent":alpha, "legacy_nonfinite":True,
                       "corrected_power":result.flatten().cpu().tolist()})
        return result
    return decode


def corrected_env_for(cfg, condition, graph=None, *, events, **kwargs):
    env = original_env_for(cfg, condition, graph=graph, **kwargs)
    original_step = env._base_step.__func__
    backend = original_step.__globals__["evaluate_control_interval"]
    decode = corrected_decoder(events)
    def recorded_decode(*args, **opts):
        before = len(events)
        result = decode(*args, **opts)
        for event in events[before:]:
            event.update(scenario_seed=int(env._seed), step=int(env.step_count))
        return result
    corrected_backend = types.FunctionType(backend.__code__, dict(backend.__globals__,
        decode_power_action=recorded_decode), argdefs=backend.__defaults__)
    corrected_backend.__kwdefaults__ = backend.__kwdefaults__
    step = types.FunctionType(original_step.__code__, dict(original_step.__globals__,
        evaluate_control_interval=corrected_backend), argdefs=original_step.__defaults__)
    env._base_step = types.MethodType(step, env)
    return env


def corrected_rollout(*args, events, **kwargs):
    from scripts_marl.run_paper_completion import rollout
    factory = lambda cfg, condition, **opts: corrected_env_for(cfg, condition, events=events, **opts)
    function = types.FunctionType(rollout.__code__, dict(rollout.__globals__, env_for=factory),
                                  argdefs=rollout.__defaults__)
    return function(*args, **kwargs)

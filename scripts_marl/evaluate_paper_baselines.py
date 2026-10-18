"""Fresh non-learning references and uncontended CPU decision-kernel timing."""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts_marl.evaluate_paper_v2 import CONFIG, PROTOCOL, load_policy, make_env, rollout, write_json
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.policies.baselines import NoOpPolicy, RandomPolicy, HeuristicNoCommunicationPolicy
from momgrap.marl.policies.online_optimizer import OnlineOptimizerPolicy
from momgrap.marl.physics.initialization import build_ramp_initializations, load_initialization, save_initialization
from momgrap.marl.envs import ramp_parallel_env as env_module
from momgrap.marl.physics import backend


def latency(cfg, policy, measurements=500, warmup=50):
    env = make_env(cfg, {})
    obs, _ = env.reset(seed=10200)
    policy.reset()
    samples = {"actor_or_optimizer_ms": [], "routing_shield_ms": [], "low_level_bd_power_ms": [], "decision_kernels_ms": []}
    component_times = {"routing_shield_ms": 0.0, "low_level_bd_power_ms": 0.0}
    def wrapper(function, key):
        def timed(*args, **kwargs):
            start = time.perf_counter()
            result = function(*args, **kwargs)
            component_times[key] += (time.perf_counter() - start) * 1000
            return result
        return timed
    from contextlib import ExitStack
    with ExitStack() as stack:
        for module, name, key in [
            (env_module, "decode_ma_action", "routing_shield_ms"), (env_module, "decode_ris_action", "routing_shield_ms"),
            (env_module, "project_to_safety", "routing_shield_ms"), (env_module, "bd_precoding", "routing_shield_ms"),
            (env_module, "evm_sinr_rate", "routing_shield_ms"), (env_module, "build_agent_messages", "routing_shield_ms"),
            (backend, "bd_precoding", "low_level_bd_power_ms"), (backend, "decode_power_action", "low_level_bd_power_ms")]:
            stack.enter_context(patch.object(module, name, wrapper(getattr(module, name), key)))
        stack.enter_context(patch.object(env.router, "submit", wrapper(env.router.submit, "routing_shield_ms")))
        for index in range(warmup + measurements):
            component_times.update(routing_shield_ms=0.0, low_level_bd_power_ms=0.0)
            start = time.perf_counter()
            actions = policy.act(obs, env)
            actor_ms = (time.perf_counter() - start) * 1000
            obs = env.step(actions)[0]
            if index >= warmup:
                samples["actor_or_optimizer_ms"].append(actor_ms)
                for key in component_times:
                    samples[key].append(component_times[key])
                samples["decision_kernels_ms"].append(actor_ms + sum(component_times.values()))
            if not env.agents:
                obs, _ = env.reset(seed=10201 + index)
                policy.reset()
    return {"n": measurements, "warmup": warmup, "batch_size": 1, "device": "cpu", "dtype": "float32/complex64",
            "torch_threads": 1, "torch_version": torch.__version__,
            "scope": "Actor/optimizer, selected routing/shield kernels, common current-estimate BD and power decoding. Excludes channel acquisition/estimation, simulator channel construction, dynamics, observation feature construction, and communication transport. This is not measured deployed end-to-end latency.",
            "samples": samples, "summary": {key: {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
                                             for key, values in samples.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="outputs_marl/paper_v2/campaign")
    parser.add_argument("--scenarios", type=int, default=50)
    parser.add_argument("--skip-archive", action="store_true", help="Smoke check only; incomplete primary reference set")
    parser.add_argument("--latency-samples", type=int, default=500)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = load_evocomm_config(CONFIG, overrides={"communication": {"enabled": False, "graph_type": "none"}})
    campaign = Path(args.campaign)
    out = campaign / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(PROTOCOL["heldout_seed_start"], PROTOCOL["heldout_seed_start"] + args.scenarios))
    manifest = {"complete": False, "references": {}, "offline_archive": None}
    policies = {"noop": NoOpPolicy(), "random": RandomPolicy(8501),
                "heuristic": HeuristicNoCommunicationPolicy(),
                "online_optimizer": OnlineOptimizerPolicy(rounds=PROTOCOL["online_optimizer_rounds"])}
    for name, policy in policies.items():
        manifest["references"][name] = {"episodes": rollout(cfg, policy, {}, seeds, trace_path=out / f"{name}.csv")}
        write_json(out / "results.json", manifest)
        print(f"Baseline complete: {name}", flush=True)
    if not args.skip_archive:
        artifact = out / "ramp_archive.pt"
        if artifact.exists():
            ramp = load_initialization(artifact)
        else:
            start = time.perf_counter()
            ramp = build_ramp_initializations(cfg.physical, seed=PROTOCOL["offline_optimizer_seed"])
            save_initialization(ramp, artifact.with_suffix(""))
            write_json(out / "archive_runtime.json", {"seconds": time.perf_counter() - start})
        manifest["offline_archive"] = {"seed": ramp.seed, "front_points": ramp.n_points, "selected_indices": ramp.selected_indices,
                                       "selection_information": "Offline perfect-CSI evolutionary surrogate on independent simulator samples; final dynamic evaluation uses paper-v2 imperfect CSI"}
        for mode in ("ramp_knee", "ramp_comm_extreme", "ramp_sensing_extreme"):
            manifest["references"][mode] = {"episodes": rollout(cfg, NoOpPolicy(), {"init_mode": mode}, seeds,
                                                                trace_path=out / f"{mode}.csv", ramp=ramp)}
            write_json(out / "results.json", manifest)
            print(f"Baseline complete: {mode}", flush=True)
    latency_results = {"online_optimizer": latency(cfg, policies["online_optimizer"], args.latency_samples)}
    for condition in ("full_nocomm", "full_comm", "centralized"):
        trained_cfg, policy = load_policy(campaign / "train" / condition / "seed_0")
        latency_results[condition] = latency(trained_cfg, policy, args.latency_samples)
    write_json(out / "latency.json", latency_results)
    manifest.update(complete=not args.skip_archive, smoke_only=args.skip_archive)
    write_json(out / "results.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Held-out paper-v2 evaluation, raw per-step evidence, and independent replicates.

No legacy checkpoint is accepted. Development selection never uses held-out IDs.
Learned actor execution receives local observations (or their union) only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from momgrap.marl.algorithms.rmappo.centralized import CentralizedActor
from momgrap.marl.algorithms.rmappo.networks import build_actors
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.normalization import RunningMeanStd
from momgrap.marl.observations.builder import observation_sizes
from momgrap.marl.policies.baselines import NoOpPolicy, RandomPolicy, HeuristicNoCommunicationPolicy, TrainedActorPolicy
from momgrap.marl.policies.online_optimizer import OnlineOptimizerPolicy
from momgrap.marl.physics.initialization import load_initialization
from momgrap.marl.protocol import EVALUATOR_VERSION, OBSERVATION_SCHEMA_VERSION
from momgrap.config import RegimeSpec

PROTOCOL = json.loads((ROOT / "configs_marl/paper_v2_protocol.json").read_text())
CONFIG = ROOT / "configs_marl/paper_v2.yaml"
EDGE_ORDER = PROTOCOL["graph_edge_order"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class CentralPolicy:
    def __init__(self, actor, rms):
        self.actor, self.rms = actor, rms
        self.reset()
    def reset(self):
        self.hidden = self.actor.zero_hidden(1)
    @torch.no_grad()
    def act(self, observations, env):
        union = np.concatenate([observations[a] for a in env.possible_agents])
        obs = self.rms.normalize(torch.as_tensor(union).unsqueeze(0))
        masks = {head: env.physical_action_mask(agent).unsqueeze(0)
                 for head, agent in zip(("power", "ris", "ma"), env.possible_agents)}
        actions, _, self.hidden = self.actor.act(obs, self.hidden, torch.ones(1, 1),
                                               deterministic=True, action_masks=masks)
        return {agent: np.array([int(actions[head][0]), 0])
                for head, agent in zip(("power", "ris", "ma"), env.possible_agents)}


def load_policy(folder):
    ck = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
    if ck.get("evaluator_version") != EVALUATOR_VERSION or ck.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("Legacy checkpoint cannot be used for paper-v2")
    restored_config = dict(ck["config"])
    restored_config["physical"] = dict(restored_config["physical"])
    restored_config["physical"]["regimes"] = [RegimeSpec(**value) for value in restored_config["physical"]["regimes"]]
    cfg = load_evocomm_config(CONFIG, overrides=restored_config)
    sizes = observation_sizes(cfg)
    if "actors" in ck:
        actors = build_actors(cfg, sizes)
        rms = {}
        for agent in actors:
            actors[agent].load_state_dict(ck["actors"][agent])
            actors[agent].eval()
            rms[agent] = RunningMeanStd(sizes[agent])
            rms[agent].load_state_dict(ck["obs_rms"][agent])
        policy = TrainedActorPolicy(actors, rms, active_agents=ck["active_agents"])
    else:
        actor = CentralizedActor(sum(sizes.values()), 9, 19, 9,
                                 cfg.rmappo.centralized_actor_hidden_size or cfg.rmappo.hidden_size)
        actor.load_state_dict(ck["actor"])
        actor.eval()
        rms = RunningMeanStd(sum(sizes.values()))
        rms.load_state_dict(ck["actor_obs_rms"])
        policy = CentralPolicy(actor, rms)
    return cfg, policy


def changed_config(cfg, overrides):
    # Clone nested dataclasses instead of mutating the training configuration.
    parts = {}
    for section, values in overrides.items():
        parts[section] = dataclasses.replace(getattr(cfg, section), **values)
    return dataclasses.replace(cfg, **parts)


def make_env(cfg, treatment, reference=None, initial=None, ramp=None):
    overrides = {}
    if "tau" in treatment:
        overrides["csi"] = {"tau_eval": treatment["tau"]}
    if "delay" in treatment:
        overrides.setdefault("csi", {})["actor_delay_steps"] = treatment["delay"]
    if "dynamics" in treatment:
        setting = PROTOCOL["dynamics_settings"][treatment["dynamics"]]
        overrides["dynamics"] = {"user_speed_min_mps": setting["speed_min"],
                                 "user_speed_max_mps": setting["speed_max"],
                                 "gauss_markov_rho": setting["rho"]}
    evo = changed_config(cfg, overrides)
    env = MOMGRAPParallelEnv(evo, eval_mode=True, init_mode=treatment.get("init_mode", "uspa_zero_phase"), ramp_init=ramp)
    if "comm_mode" in treatment:
        env.set_communication_mode(treatment["comm_mode"])
    if "corruption" in treatment:
        env.set_message_corruption(treatment["corruption"])
    if "graph" in treatment:
        adjacency = torch.zeros(3, 3, dtype=torch.bool)
        for bit, (source, target) in enumerate(EDGE_ORDER):
            adjacency[source, target] = bool(treatment["graph"] & (1 << bit))
        env.set_adjacency(adjacency)
    if "csi_features" in treatment:
        env.set_csi_features(treatment["csi_features"], reference)
    if treatment.get("pricing") == "requested":
        env.set_message_pricing("requested")
    if initial is not None:
        positions, phases, power = initial
        env._initial_config = lambda: (positions.clone(), phases.clone())
    return env


def rollout(cfg, policy, treatment, seeds, trace_path=None, reference=None, initial=None, ramp=None, replay=None):
    env = make_env(cfg, treatment, reference, initial, ramp)
    result = []
    trace_handle = trace_path.open("w", newline="", encoding="utf-8") if trace_path else None
    writer = None
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            policy.reset()
            if initial is not None:
                # Static learned configuration fixes the profile rule, not realized powers.
                env._frozen = True
                env._frozen_power_action = int(initial[2])
            rows = []
            for step in range(cfg.dynamics.episode_length):
                if step == treatment.get("freeze_step", -1):
                    env.freeze_configuration()
                start = time.perf_counter()
                actions = policy.act(obs, env)
                if replay is not None:
                    for agent in env.possible_agents:
                        actions[agent][1] = replay[(seed, step)][agent]
                decision_ms = (time.perf_counter() - start) * 1000
                # Random action baseline uses the same actuator schedule and sends no messages.
                if isinstance(policy, RandomPolicy):
                    for agent in actions:
                        if not bool(env.physical_action_mask(agent)[int(actions[agent][0])]):
                            actions[agent][0] = 0
                        actions[agent][1] = 0
                obs, rewards, _, _, infos = env.step(actions)
                info = infos["bs_agent"]
                reward = float(rewards["bs_agent"])
                charged = info["message_requested"] if treatment.get("pricing") == "requested" else info["message_delivered"]
                before_cost = reward + cfg.reward.w_message * charged
                positions, _ = env.decision_configuration()
                offsets = (positions[0] - env.centers).norm(dim=-1).mean()
                row = {"scenario_seed": seed, "step": step, "regime": info["regime"], "net_reward": reward,
                       "reward_before_message_cost": before_cost,
                       "reward_at_default_price": before_cost - 0.01 * info["message_delivered"],
                       "min_user_rate": info["min_user_rate"], "sum_rate": info["sum_rate"],
                       "illumination_db": info["min_sensing_illumination_db"], "outage_fraction": info["outage_count"] / cfg.physical.K,
                       "requested": info["message_requested"], "eligible": info["message_eligible"],
                       "delivered": info["message_delivered"], "budget_dropped": info["message_dropped"],
                       "graph_rejected": info["graph_rejected"], "reconfiguration": info["reconfiguration_cost"],
                       "violation_before": info["violation_before"], "violation_after": info["violation_after"],
                       "power_action": info["power_action"], "ris_action": info["ris_action"], "ma_action": info["ma_action"],
                       **{f"comm_action_{agent}": int(actions[agent][1]) for agent in env.possible_agents},
                       "layout_offset_wavelengths": float(offsets / cfg.physical.wavelength),
                       "decision_ms_exploratory": decision_ms,
                       "candidate_evaluations": getattr(policy, "last_candidate_evaluations", 0)}
                rows.append(row)
                if trace_handle:
                    if writer is None:
                        writer = csv.DictWriter(trace_handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
            # Fixed common window for adaptation; whole-episode return reported separately.
            post = rows[PROTOCOL["primary_freeze_step"]:]
            primary_metrics = ["min_user_rate", "sum_rate", "illumination_db", "outage_fraction", "requested", "eligible", "delivered",
                               "budget_dropped", "graph_rejected", "reconfiguration", "violation_before", "violation_after"]
            episode = {"scenario_seed": seed, "regime": rows[0]["regime"], "steps": len(rows),
                       "net_return": sum(r["net_reward"] for r in rows),
                       "return_before_message_cost": sum(r["reward_before_message_cost"] for r in rows),
                       "return_at_default_price": sum(r["reward_at_default_price"] for r in rows),
                       "prefix25_sha256": hashlib.sha256(np.asarray([[r["net_reward"], r["power_action"], r["ris_action"], r["ma_action"]]
                                                                   for r in rows[:25]], dtype=np.float64).tobytes()).hexdigest()}
            for key in primary_metrics:
                episode[key] = float(np.mean([r[key] for r in rows]))
                episode["post25_" + key] = float(np.mean([r[key] for r in post]))
            episode["post25_return"] = sum(r["net_reward"] for r in post)
            result.append(episode)
    finally:
        if trace_handle:
            trace_handle.close()
        env.close()
    return result


def development_features(cfg):
    env = make_env(cfg, {})
    samples = {agent: [] for agent in env.possible_agents}
    lengths = {"bs_agent": 4 + cfg.physical.K + 8, "ris_agent": 5 + cfg.physical.K + 8, "ma_agent": 4 + cfg.physical.K + 8}
    noop = NoOpPolicy()
    for seed in PROTOCOL["development_seeds"]:
        obs, _ = env.reset(seed=seed)
        while env.agents:
            for agent in samples:
                samples[agent].append(obs[agent][:lengths[agent]].copy())
            obs = env.step(noop.act(obs, env))[0]
    return {agent: np.asarray(values) for agent, values in samples.items()}


def select_static_learned(cfg, policy, out):
    candidates = []
    env = make_env(cfg, {})
    for seed in PROTOCOL["development_seeds"]:
        obs, _ = env.reset(seed=seed)
        policy.reset()
        for _ in range(PROTOCOL["primary_freeze_step"]):
            obs = env.step(policy.act(obs, env))[0]
        positions, phases = env.decision_configuration()
        candidates.append((positions, phases, env._prev_actions["bs_agent"]))
    scores = []
    for initial in candidates:
        values = rollout(cfg, NoOpPolicy(), {}, PROTOCOL["development_seeds"], initial=initial)
        scores.append(float(np.mean([r["net_return"] for r in values])))
    winner = int(np.argmax(scores))
    torch.save({"positions": candidates[winner][0], "theta": candidates[winner][1], "power_profile": candidates[winner][2],
                "development_scores": scores, "selected_index": winner, "development_seeds": PROTOCOL["development_seeds"]}, out / "static_learned.pt")
    return candidates[winner]


def worker(args):
    torch.set_num_threads(1)
    folder = Path(args.campaign) / "train" / args.condition / f"seed_{args.seed}"
    status = json.loads((folder / "status.json").read_text())
    if not status["complete"]:
        raise RuntimeError("Evaluation requires a completed training run")
    cfg, policy = load_policy(folder)
    out = Path(args.campaign) / "evaluation" / args.condition / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    base = {"pricing": "requested"} if args.condition == "request_price" else {}
    checkpoint_hash = hashlib.sha256((folder / "checkpoint.pt").read_bytes()).hexdigest()
    manifest = {"complete": False, "condition": args.condition, "training_seed": args.seed,
                "checkpoint_sha256": checkpoint_hash, "source_sha256": status["source_sha256"],
                "evaluator_version": EVALUATOR_VERSION, "treatments": {}}
    treatments = {"normal": base}
    if args.condition == "full_nocomm":
        treatments.update({"frozen25": {"freeze_step": 25}, "frozen10": {"freeze_step": 10}})
    if args.condition == "full_comm":
        treatments.update({"comm_off": {"comm_mode": "none"}, "comm_random": {"comm_mode": "random"},
                           "comm_forced": {"comm_mode": "full"},
                           **{f"content_{mode}": {"corruption": mode} for mode in ("zero", "random", "permute")}})
    # Primary scenarios are identical across all conditions and independent policies.
    seeds = list(range(PROTOCOL["heldout_seed_start"], PROTOCOL["heldout_seed_start"] + args.scenarios))
    replay = None
    for name, treatment in treatments.items():
        if name.startswith("content_"):
            with (out / "normal.csv").open(newline="", encoding="utf-8") as handle:
                reference_rows = list(csv.DictReader(handle))
            replay = {(int(r["scenario_seed"]), int(r["step"])):
                      {agent: int(r[f"comm_action_{agent}"]) for agent in cfg_agents()}
                      for r in reference_rows}
            treatment = dict(treatment, delivery_schedule="replayed from normal treatment")
        values = rollout(cfg, policy, treatment, seeds, trace_path=out / f"{name}.csv",
                         replay=replay if name.startswith("content_") else None)
        if name.startswith("content_"):
            with (out / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
                altered_rows = list(csv.DictReader(handle))
            for reference_row, altered_row in zip(reference_rows, altered_rows):
                if any(reference_row[key] != altered_row[key] for key in ("requested", "eligible", "delivered", "budget_dropped", "graph_rejected")):
                    raise AssertionError("Payload intervention changed delivery accounting")
        manifest["treatments"][name] = {"settings": treatment, "episodes": values}
        write_json(out / "results.json", manifest)
    if args.condition == "full_nocomm":
        normal = manifest["treatments"]["normal"]["episodes"]
        frozen = manifest["treatments"]["frozen25"]["episodes"]
        if any(a["prefix25_sha256"] != b["prefix25_sha256"] for a, b in zip(normal, frozen)):
            raise AssertionError("Adaptive and frozen runs did not share their first 25 decisions")
        initial = select_static_learned(cfg, policy, out)
        manifest["treatments"]["static_learned"] = {"settings": {"selected_on": "development_only"},
            "episodes": rollout(cfg, NoOpPolicy(), {}, seeds, trace_path=out / "static_learned.csv", initial=initial)}
    if args.condition in ("full_nocomm", "feedforward", "masked_csi") and not args.core_only:
        exploratory = seeds[:PROTOCOL["heldout_scenarios_exploratory"]]
        reference_pool = development_features(cfg) if args.condition == "full_nocomm" else None
        settings = {**{f"tau_{value}": {"tau": value} for value in PROTOCOL["tau_grid"]},
                    **{f"delay_{value}": {"delay": value} for value in PROTOCOL["delay_grid"]},
                    **{f"dynamics_{name}": {"dynamics": name} for name in ("static", "slow", "fast")}}
        if reference_pool is not None:
            settings.update({"csi_fixed": {"csi_features": "fixed"}, "csi_shuffled": {"csi_features": "shuffled"}})
        for name, treatment in settings.items():
            reference = ({a: x.mean(axis=0) for a, x in reference_pool.items()} if name == "csi_fixed" else reference_pool)
            manifest["treatments"][name] = {"settings": treatment,
                "episodes": rollout(cfg, policy, treatment, exploratory, trace_path=out / f"{name}.csv", reference=reference)}
            if args.condition == "full_nocomm" and name.startswith("dynamics_"):
                frozen_treatment = dict(treatment, freeze_step=25)
                manifest["treatments"][name + "_frozen25"] = {"settings": frozen_treatment,
                    "episodes": rollout(cfg, policy, frozen_treatment, exploratory, trace_path=out / f"{name}_frozen25.csv")}
            write_json(out / "results.json", manifest)
    manifest["complete"] = True
    write_json(out / "results.json", manifest)
    print(f"Evaluation complete: {args.condition}/seed_{args.seed}", flush=True)


def cfg_agents():
    return ("bs_agent", "ris_agent", "ma_agent")


def orchestrate(args):
    campaign = Path(args.campaign)
    statuses = sorted((campaign / "train").glob("*/seed_*/status.json"))
    jobs = []
    for path in statuses:
        status = json.loads(path.read_text())
        if args.conditions and status["condition"] not in args.conditions:
            continue
        if not status["complete"]:
            raise RuntimeError(f"Training is incomplete: {path}")
        jobs.append((status["condition"], status["training_seed"]))
    if args.conditions:
        jobs.sort(key=lambda job: (args.conditions.index(job[0]), job[1]))
    def run(job):
        condition, seed = job
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--campaign", str(campaign),
                   "--condition", condition, "--seed", str(seed), "--scenarios", str(args.scenarios)]
        if args.core_only:
            command.append("--core-only")
        with (campaign / f"eval_{condition}_seed_{seed}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        return {"condition": condition, "seed": seed, "returncode": result.returncode}
    state = {"complete": False, "scope": args.conditions or "all conditions", "core_only": args.core_only, "jobs": []}
    status_path = campaign / Path(args.status_name).name
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            result = future.result()
            state["jobs"].append(result)
            write_json(status_path, state)
            print(json.dumps(result), flush=True)
    state["complete"] = all(job["returncode"] == 0 for job in state["jobs"])
    write_json(status_path, state)
    if not state["complete"]:
        return 1
    if args.skip_followups:
        return 0
    # Baselines and reporting are separate inspectable scripts, never legacy results.
    for script in ("evaluate_paper_baselines.py", "report_paper_v2.py"):
        result = subprocess.run([sys.executable, "-u", str(ROOT / "scripts_marl" / script), "--campaign", str(campaign)], cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="outputs_marl/paper_v2/campaign")
    parser.add_argument("--condition")
    parser.add_argument("--conditions", nargs="+", help="Evaluate selected completed conditions before the full campaign finishes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--scenarios", type=int, default=50)
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--skip-followups", action="store_true")
    parser.add_argument("--status-name", default="evaluation_status.json")
    args = parser.parse_args()
    if args.condition:
        worker(args)
        return 0
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())

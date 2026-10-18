"""Train, select on development, assess and report isolated paper-v3 pilot runs."""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments_marl.paper_v3 import (VERSION, CONDITIONS, TRAIN_BASE, TRAIN_STRIDE,
    ImprovementEnv, ImprovementTrainer, make_config, settings, source_digest,
    save_experiment_checkpoint, load_experiment_policy, training_seed)
from momgrap.marl.config import evocomm_config_to_dict, save_resolved_config

DEFAULT_PROTOCOL = ROOT / "configs_marl/paper_v3_screening.json"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    payload = dict(value)
    if "config" in payload:
        payload["config"] = json_config(payload["config"])
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def json_config(config):
    # Pure LoS uses an infinite Rician K-factor. Encode this physical metadata
    # explicitly while still rejecting nonfinite losses/returns/status metrics.
    config = copy.deepcopy(config)
    for regime in config["physical"]["regimes"]:
        value = float(regime["k_factor_db"])
        if np.isinf(value):
            regime["k_factor_db"] = "Infinity" if value > 0 else "-Infinity"
    return config


def validate_split(protocol):
    from scripts_marl.paper_seed_audit import training_scenario_ids
    dev, assessment = set(protocol["development_seeds"]), set(protocol["assessment_seeds"])
    final = set(range(protocol["reserved_final_seed_start"], protocol["reserved_final_seed_start"] + protocol["reserved_final_scenarios"]))
    if any(a & b for a, b in [(dev, assessment), (dev, final), (assessment, final)]):
        raise ValueError("Development/assessment/final seed sets overlap")
    maximum_episodes = protocol["steps_per_run"] // 800 + 2
    seen = set()
    for seed in protocol["training_seeds"]:
        ids = {training_seed(seed, i, e, 8) for e in range(maximum_episodes) for i in range(8)}
        if ids & (dev | assessment | final | seen):
            raise ValueError("New training seed namespace overlaps another split/replicate")
        seen |= ids
    old_ids = set()
    for path in (ROOT / "outputs_marl/paper_v2/campaign/train/full_comm").glob("seed_*/status.json"):
        old_ids |= training_scenario_ids(json.loads(path.read_text()))
    if (dev | assessment | final | seen) & old_ids:
        raise ValueError("Seed split overlaps legacy campaign training")
    return {"passed": True, "training_seeds_enumerated": len(seen), "legacy_training_seeds_checked": len(old_ids),
            "development_scenarios": len(dev), "assessment_scenarios": len(assessment), "final_scenarios_reserved": len(final)}


def train_worker(out, condition, seed, protocol, digest):
    torch.set_num_threads(1)
    if source_digest() != digest:
        raise RuntimeError("Source changed after protocol lock")
    cfg = make_config(condition, seed, protocol["steps_per_run"])
    folder = out / "train" / condition / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    status_path = folder / "status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get("training_complete") and previous["source_sha256"] == digest and previous["config"] == json_config(evocomm_config_to_dict(cfg)):
            return
        raise RuntimeError(f"Existing incomplete/incompatible run at {folder}; preserve it and use a new output directory")
    spec = settings(condition)
    status = {"experiment_version": VERSION, "condition": condition, "training_seed": seed,
              "source_sha256": digest, "complete": False, "training_complete": False, "config": evocomm_config_to_dict(cfg),
              "settings": spec, "pid": os.getpid(), "stage": "training", "actual_transitions": 0,
              "best_development_return": None, "selection_criterion": protocol["checkpoint_selection"]}
    write_json(status_path, status)
    save_resolved_config(cfg, folder / "config_resolved.yaml")
    def factory(i):
        return ImprovementEnv(cfg, actor_projection=spec["actor_projection"], eval_mode=i == 10000)
    trainer = ImprovementTrainer(cfg, factory, condition, log_dir=folder, seed=seed)
    start = time.perf_counter()
    best = -float("inf")
    next_eval = cfg.rmappo.eval_interval
    while trainer.global_step < cfg.rmappo.total_timesteps:
        metrics = trainer.collect_rollout()
        diagnostics = trainer.update()
        if not all(np.isfinite(x) for x in [*metrics.values(), *diagnostics.values()]):
            raise RuntimeError("Nonfinite training diagnostics")
        trainer._log(metrics, diagnostics)
        status.update(actual_transitions=trainer.global_step, updates=trainer.update_index,
                      training_seconds=time.perf_counter() - start, stage="training", latest_diagnostics=diagnostics)
        write_json(status_path, status)
        if trainer.global_step >= next_eval or trainer.global_step >= cfg.rmappo.total_timesteps:
            if source_digest() != digest:
                raise RuntimeError("Source/configuration changed during training; preserve this run and relock a new campaign")
            status["stage"] = "development_selection"
            write_json(status_path, status)
            ev = trainer.evaluate(protocol["development_seeds"])
            trainer._log_eval(ev)
            path = folder / f"checkpoint_{trainer.global_step}.pt"
            save_experiment_checkpoint(path, trainer)
            if ev["eval_return_mean"] > best:
                best = ev["eval_return_mean"]
                save_experiment_checkpoint(folder / "checkpoint_best.pt", trainer)
                status.update(best_development_return=best, best_step=trainer.global_step)
            status.update(last_development_return=ev["eval_return_mean"])
            write_json(status_path, status)
            next_eval += cfg.rmappo.eval_interval
        if trainer.update_index % 10 == 0:
            print(f"{condition}/seed_{seed} {trainer.global_step}/{cfg.rmappo.total_timesteps}", flush=True)
    save_experiment_checkpoint(folder / "checkpoint_final.pt", trainer)
    if trainer.writer is not None:
        trainer.writer.close()
    for env in trainer.envs:
        env.close()
    if trainer._eval_env is not None:
        trainer._eval_env.close()
    status.update(training_complete=True, training_seconds=time.perf_counter() - start,
                  stage="awaiting_assessment", checkpoint_best_sha256=hashlib.sha256((folder / "checkpoint_best.pt").read_bytes()).hexdigest(),
                  checkpoint_final_sha256=hashlib.sha256((folder / "checkpoint_final.pt").read_bytes()).hexdigest())
    write_json(status_path, status)


def assess_checkpoint(path, seeds, trace_path, comm_off=False):
    cfg, policy, ck = load_experiment_policy(path)
    env = ImprovementEnv(cfg, actor_projection=ck["experiment_settings"]["actor_projection"], eval_mode=True)
    env.projection.load_state_dict(ck["projection"])
    if comm_off:
        env.set_communication_mode("none")
    episodes = []
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            policy.reset()
            rows = []
            for step in range(cfg.dynamics.episode_length):
                actions = policy.act(obs, env)
                obs, rewards, _, _, infos = env.step(actions)
                info = infos["bs_agent"]
                row = {"scenario_seed": seed, "step": step, "regime": info["regime"], "net_reward": float(rewards["bs_agent"]),
                       "min_user_rate": info["min_user_rate"], "illumination_db": info["min_sensing_illumination_db"],
                       "outage_fraction": info["outage_count"] / cfg.physical.K, "delivered": info["message_delivered"],
                       "requested": info["message_requested"], "reconfiguration": info["reconfiguration_cost"],
                       "power_action": info["power_action"], "ris_action": info["ris_action"], "ma_action": info["ma_action"]}
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)
            episodes.append({"scenario_seed": seed, "regime": rows[0]["regime"], "steps": len(rows),
                             "net_return": sum(r["net_reward"] for r in rows),
                             **{key: float(np.mean([r[key] for r in rows])) for key in
                                ("min_user_rate", "illumination_db", "outage_fraction", "delivered", "requested", "reconfiguration")}})
    env.close()
    return episodes


def evaluate_worker(out, condition, seed, protocol, digest):
    torch.set_num_threads(1)
    if source_digest() != digest:
        raise RuntimeError("Source changed after lock")
    folder = out / "train" / condition / f"seed_{seed}"
    status = json.loads((folder / "status.json").read_text())
    if not status["training_complete"] or status["source_sha256"] != digest:
        raise RuntimeError("Assessment requires compatible completed training")
    evaluation = out / "evaluation" / condition / f"seed_{seed}"
    evaluation.mkdir(parents=True, exist_ok=True)
    status["stage"] = "assessment"
    write_json(folder / "status.json", status)
    result = {"experiment_version": VERSION, "condition": condition, "training_seed": seed, "complete": False,
              "scope": protocol["assessment_role"], "source_sha256": digest, "treatments": {},
              "checkpoint_best_sha256": status["checkpoint_best_sha256"], "checkpoint_final_sha256": status["checkpoint_final_sha256"]}
    for treatment in ["best", "final"] + (["best_comm_off"] if condition != "no_comm" else []):
        kind = "final" if treatment == "final" else "best"
        path = folder / f"checkpoint_{kind}.pt"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == status[f"checkpoint_{kind}_sha256"]
        result["treatments"][treatment] = {"episodes": assess_checkpoint(path, protocol["assessment_seeds"],
                                                                       evaluation / f"{treatment}.csv", treatment.endswith("comm_off"))}
    result["complete"] = True
    write_json(evaluation / "results.json", result)
    status.update(complete=True, stage="complete")
    write_json(folder / "status.json", status)


def summarize(out, protocol):
    records = []
    for path in sorted((out / "evaluation").glob("*/seed_*/results.json")):
        data = json.loads(path.read_text())
        if not data["complete"]:
            continue
        for treatment, entry in data["treatments"].items():
            eps = entry["episodes"]
            assert [e["scenario_seed"] for e in eps] == protocol["assessment_seeds"]
            records.append({"condition": data["condition"], "training_seed": data["training_seed"], "treatment": treatment,
                            **{key: float(np.mean([e[key] for e in eps])) for key in
                               ("net_return", "min_user_rate", "illumination_db", "outage_fraction", "delivered", "reconfiguration")}})
    report = out / "report"
    report.mkdir(exist_ok=True)
    write_json(report / "results.json", {"scope": "Development screening; insufficient for final claims.",
               "stage": protocol["stage"], "training_replicates": len(protocol["training_seeds"]), "per_seed": records})
    with (report / "per_seed.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ["# Kết quả screening cải thiện paper-v3", "", f"Stage: {protocol['stage']}. Đây là development screening; bộ final dự trữ chưa được mở. "
             f"Mỗi điều kiện chỉ có {len(protocol['training_seeds'])} training policy nên không kết luận superiority/significance. "
             f"Checkpoint best chọn trên {len(protocol['development_seeds'])} scenario development riêng, "
             f"bảng dưới dùng {len(protocol['assessment_seeds'])} assessment scenario khác; assessment vẫn là development nếu dùng để chọn cải tiến.", "",
             "| Điều kiện | Seed | Checkpoint/treatment | Return | Min-rate | Illumination dB | Delivered/interval |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['condition']} | {r['training_seed']} | {r['treatment']} | {r['net_return']:.3f} | {r['min_user_rate']:.3f} | {r['illumination_db']:.3f} | {r['delivered']:.3f} |")
    lines += ["", "Mean return giữa seed của checkpoint best:", ""]
    for condition in protocol["conditions"]:
        values = [r["net_return"] for r in records if r["condition"] == condition and r["treatment"] == "best"]
        lines.append(f"- {condition}: {np.mean(values):.3f}; từng seed {', '.join(f'{v:.3f}' for v in values)}.")
    (report / "KET_QUA_SCREENING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["campaign", "train", "evaluate"], default="campaign")
    parser.add_argument("--out", default="outputs_marl/paper_v3/screening")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.mode != "campaign":
        lock = json.loads((out / "protocol_locked.json").read_text())
        worker = train_worker if args.mode == "train" else evaluate_worker
        worker(out, args.condition, args.seed, lock["protocol"], lock["source_sha256"])
        return 0
    if (out / "campaign_status.json").exists():
        raise RuntimeError("Existing campaign output; use another directory to preserve its evidence")
    protocol = json.loads(DEFAULT_PROTOCOL.read_text())
    if args.smoke:
        protocol.update(stage="pipeline_smoke", steps_per_run=2048, training_seeds=[0], total_training_transitions=10240,
                        development_seeds=protocol["development_seeds"][:3], assessment_seeds=protocol["assessment_seeds"][:3])
    workers = args.workers or protocol["workers"]
    digest = source_digest()
    write_json(out / "protocol_locked.json", {"source_sha256": digest, "protocol": protocol, "seed_audit": validate_split(protocol)})
    paths = list((ROOT / "momgrap").rglob("*.py")) + list((ROOT / "experiments_marl").rglob("*.py"))
    paths += [ROOT / "scripts_marl/run_paper_improvements.py", ROOT / "scripts_marl/paper_seed_audit.py",
              DEFAULT_PROTOCOL, ROOT / "configs_marl/paper_v2.yaml"]
    for path in paths:
        target = out / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    state = {"experiment_version": VERSION, "pid": os.getpid(), "start_unix": time.time(), "complete": False,
             "stage": "training", "source_sha256": digest, "planned_runs": len(protocol["conditions"]) * len(protocol["training_seeds"]),
             "successful_training": [], "successful_evaluation": [], "failed": [], "final_scenarios_opened": False}
    write_json(out / "campaign_status.json", state)
    def run(mode, condition, seed):
        log_path = out / f"{mode}_{condition}_seed_{seed}.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", mode, "--condition", condition,
                   "--seed", str(seed), "--out", str(out)]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with log_path.open("a", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return {"condition": condition, "seed": seed, "returncode": result.returncode, "log": str(log_path)}
    jobs = [(c, s) for c in protocol["conditions"] for s in protocol["training_seeds"]]
    for mode, key in [("train", "successful_training"), ("evaluate", "successful_evaluation")]:
        state["stage"] = "training" if mode == "train" else "assessment"
        write_json(out / "campaign_status.json", state)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run, mode, c, s) for c, s in jobs]
            for future in as_completed(futures):
                result = future.result()
                state[key if result["returncode"] == 0 else "failed"].append(result)
                state["last_update_unix"] = time.time()
                write_json(out / "campaign_status.json", state)
                print(json.dumps(result), flush=True)
        if state["failed"]:
            state.update(complete=True, stage="failed", end_unix=time.time())
            write_json(out / "campaign_status.json", state)
            return 1
    summarize(out, protocol)
    state.update(complete=True, stage="complete", end_unix=time.time())
    write_json(out / "campaign_status.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fresh versioned development screening of payload and routing, with no final evaluation."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments_marl_v4.messages import (VERSION, CONDITIONS, MessageEnv, MessageTrainer, make_config,
                                         settings, source_digest, save_policy, load_policy)
from experiments_marl.paper_v3 import training_seed
from scripts_marl.run_paper_improvements import (write_json, validate_split, train_worker as old_train)
from scripts_marl.evaluate_paper_v2 import rollout as old_rollout
from scripts_marl.report_paper_v2 import nested_interval
from scripts_marl.run_paper_confirmation import sha, verify_sources

DEFAULT_PROTOCOL = ROOT / "configs_marl/paper_v4_messages.json"


def require_sources(lock):
    if lock["source_sha256"] != source_digest():
        raise RuntimeError("Frozen v4 source changed; preserve run and relock a fresh campaign")
    verify_sources(json.loads((ROOT / "outputs_marl/paper_v3/confirmation_v2/protocol_locked.json").read_text()))


def history_audit(protocol):
    final = set(range(protocol["reserved_final_seed_start"], protocol["reserved_final_seed_start"] + protocol["reserved_final_scenarios"]))
    count = protocol["steps_per_run"] // 800 + 2
    proposed = {training_seed(s, i, e, 8) for s in protocol["training_seeds"] for e in range(count) for i in range(8)}
    paths = list((ROOT / "outputs_marl").glob("paper_v*/**/episode_seeds.csv"))
    for path in paths:
        with path.open(newline="") as handle:
            actual = {int(row["scenario_seed"]) for row in csv.DictReader(handle)}
        if actual & (proposed | final):
            raise ValueError(f"Proposed training/final namespace already used: {path}")
    def scenarios(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "scenario_seed" and item in final:
                    raise ValueError("Proposed final scenario already evaluated")
                scenarios(item)
        elif isinstance(value, list):
            for item in value:
                scenarios(item)
    manifests = list((ROOT / "outputs_marl").glob("paper_v*/**/results.json"))
    for path in manifests:
        scenarios(json.loads(path.read_text(encoding="utf-8")))
    return {"passed": True, "training_history_files": len(paths), "result_manifests": len(manifests)}


def initialization_audit(protocol):
    torch.set_num_threads(1)
    records = {}
    for seed in protocol["training_seeds"]:
        hashes = []
        for condition in protocol["conditions"]:
            cfg = make_config(condition, seed, protocol["steps_per_run"])
            trainer = MessageTrainer(cfg, lambda i: MessageEnv(cfg, condition), condition, seed=seed)
            digest = hashlib.sha256()
            for module in [*trainer.actors.values(), trainer.critic]:
                for name, value in module.state_dict().items():
                    digest.update(name.encode())
                    digest.update(value.detach().numpy().tobytes())
            hashes.append(digest.hexdigest())
            for env in trainer.envs:
                env.close()
        if len(set(hashes)) != 1:
            raise ValueError("Arms do not share initial actor/critic parameters")
        records[str(seed)] = hashes[0]
    return {"passed": True, "initial_actor_critic_sha256": records}


def train(out, condition, seed, lock):
    require_sources(lock)
    # The frozen v3 training procedure is reused with instance-bound v4 providers.
    # No globals in the original module are modified.
    namespace = dict(old_train.__globals__)
    namespace.update(source_digest=source_digest, make_config=make_config, settings=settings,
                     ImprovementTrainer=MessageTrainer, VERSION=VERSION, save_experiment_checkpoint=save_policy,
                     ImprovementEnv=lambda cfg, actor_projection="raw", **kw: MessageEnv(cfg, condition, **kw))
    function = types.FunctionType(old_train.__code__, namespace)
    function(out, condition, seed, lock["protocol"], lock["source_sha256"])


def lock_checkpoints(out, lock):
    records, scenario_sets = {}, {}
    protected = set(lock["protocol"]["development_seeds"] + lock["protocol"]["assessment_seeds"])
    for condition in lock["protocol"]["conditions"]:
        for seed in lock["protocol"]["training_seeds"]:
            folder = out / "train" / condition / f"seed_{seed}"
            status = json.loads((folder / "status.json").read_text())
            if not status["training_complete"] or status["actual_transitions"] != lock["protocol"]["steps_per_run"]:
                raise ValueError("Incomplete training before assessment")
            with (folder / "episode_seeds.csv").open(newline="") as handle:
                ids = [int(r["scenario_seed"]) for r in csv.DictReader(handle)]
            if len(set(ids)) != len(ids) or protected & set(ids) or (seed in scenario_sets and scenario_sets[seed] != set(ids)):
                raise ValueError("Actual training seed/matching audit failed")
            scenario_sets[seed] = set(ids)
            records[f"{condition}/seed_{seed}"] = {"best_step": status["best_step"],
                "selected_checkpoint_sha256": sha(folder / "checkpoint_best.pt"), "final_checkpoint_sha256": sha(folder / "checkpoint_final.pt")}
            if records[f"{condition}/seed_{seed}"]["selected_checkpoint_sha256"] != status["checkpoint_best_sha256"]:
                raise ValueError("Selected hash mismatch")
    values = list(scenario_sets.values())
    if any(values[i] & values[j] for i in range(len(values)) for j in range(i)):
        raise ValueError("Training replicates not disjoint")
    path = out / "selected_checkpoints_locked.json"
    if path.exists():
        raise RuntimeError("Never overwrite selection lock")
    write_json(path, {"locked_unix": time.time(), "source_sha256": lock["source_sha256"], "final_scenarios_opened": False, "selected": records})
    return sha(path)


def assess(path, condition, seeds, trace, treatment=None, replay=None):
    cfg, policy, ck = load_policy(path, condition)
    def make_env(cfg, treatment, reference=None, initial=None, ramp=None):
        env = MessageEnv(cfg, condition, eval_mode=True)
        env.projection.load_state_dict(ck["projection"])
        if "comm_mode" in treatment:
            env.set_communication_mode(treatment["comm_mode"])
        if "corruption" in treatment:
            env.set_message_corruption(treatment["corruption"])
        return env
    namespace = dict(old_rollout.__globals__, make_env=make_env)
    function = types.FunctionType(old_rollout.__code__, namespace, argdefs=old_rollout.__defaults__)
    return function(cfg, policy, treatment or {}, seeds, trace, replay=replay)


def evaluate(out, condition, seed, lock):
    torch.set_num_threads(1)
    require_sources(lock)
    state = json.loads((out / "campaign_status.json").read_text())
    if sha(out / "selected_checkpoints_locked.json") != state["selection_manifest_sha256"]:
        raise ValueError("Assessment selection lock changed")
    entry = json.loads((out / "selected_checkpoints_locked.json").read_text())["selected"][f"{condition}/seed_{seed}"]
    folder = out / "train" / condition / f"seed_{seed}"
    target = out / "evaluation" / condition / f"seed_{seed}"
    target.mkdir(parents=True)
    for kind, key in (("best", "selected_checkpoint_sha256"), ("final", "final_checkpoint_sha256")):
        if sha(folder / f"checkpoint_{kind}.pt") != entry[key]:
            raise ValueError("Checkpoint changed")
    ids = lock["protocol"]["assessment_seeds"]
    selected = assess(folder / "checkpoint_best.pt", condition, ids, target / "selected.csv")
    treatments = {"selected": {"episodes": selected}}
    if condition != "no_comm":
        treatments["comm_off"] = {"episodes": assess(folder / "checkpoint_best.pt", condition, ids, target / "comm_off.csv", {"comm_mode": "none"})}
        with (target / "selected.csv").open(newline="") as handle:
            normal = list(csv.DictReader(handle))
        agents = ("bs_agent", "ris_agent", "ma_agent")
        replay = {(int(r["scenario_seed"]), int(r["step"])): {a: int(r[f"comm_action_{a}"]) for a in agents} for r in normal}
        treatments["content_zero"] = {"episodes": assess(folder / "checkpoint_best.pt", condition, ids, target / "content_zero.csv", {"corruption": "zero"}, replay)}
        with (target / "content_zero.csv").open(newline="") as handle:
            altered = list(csv.DictReader(handle))
        keys = ("scenario_seed", "step", "requested", "eligible", "delivered", "budget_dropped", "graph_rejected") + tuple(f"comm_action_{a}" for a in agents)
        if len(normal) != len(altered) or any(any(a[k] != b[k] for k in keys) for a, b in zip(normal, altered)):
            raise ValueError("Fixed-send zero intervention changed sends/delivery")
    treatments["final_checkpoint"] = {"episodes": assess(folder / "checkpoint_final.pt", condition, ids, target / "final_checkpoint.csv")}
    write_json(target / "results.json", {"complete": True, "experiment_version": VERSION, "condition": condition, "training_seed": seed,
               "source_sha256": lock["source_sha256"], "checkpoint_lock": entry, "scenario_ids": ids, "scope": "Development assessment only", "treatments": treatments})
    status = json.loads((folder / "status.json").read_text())
    status.update(complete=True, stage="complete")
    write_json(folder / "status.json", status)


def report(out, protocol):
    matrices, rows = {}, []
    metrics = ("net_return", "min_user_rate", "illumination_db", "outage_fraction", "delivered", "reconfiguration")
    for condition in protocol["conditions"]:
        for seed in protocol["training_seeds"]:
            data = json.loads((out / "evaluation" / condition / f"seed_{seed}/results.json").read_text())
            for name, item in data["treatments"].items():
                eps = item["episodes"]
                if [e["scenario_seed"] for e in eps] != protocol["assessment_seeds"]:
                    raise ValueError("Mismatched assessment scenarios")
                key = f"{condition}/{name}"
                for metric in metrics:
                    matrices.setdefault((key, metric), []).append([e[metric] for e in eps])
                rows.append({"condition": condition, "seed": seed, "treatment": name, **{m: float(np.mean([e[m] for e in eps])) for m in metrics}})
    estimates = {}
    for (key, metric), matrix in matrices.items():
        estimates.setdefault(key, {})[metric] = nested_interval(matrix, seed=protocol["statistics_seed"])
    comparisons, controls, candidates = {}, {}, []
    reference = np.array(matrices[("comm_reference/selected", "net_return")])
    for condition in protocol["conditions"]:
        if condition != "no_comm":
            normal = np.array(matrices[(f"{condition}/selected", "net_return")])
            controls[condition] = {name: nested_interval(normal, matrices[(f"{condition}/{name}", "net_return")], paired=True, seed=protocol["statistics_seed"]) for name in ("comm_off", "content_zero")}
        if condition in ("comm_payload", "comm_rotating_router"):
            normal = np.array(matrices[(f"{condition}/selected", "net_return")])
            effect = nested_interval(normal, reference, paired=True, seed=protocol["statistics_seed"])
            effect["positive_training_pairs"] = int(np.sum((normal - reference).mean(1) > 0))
            comparisons[condition] = effect
            if effect["mean"] > 0 and effect["positive_training_pairs"] >= 2 and all(v["mean"] > 0 for v in controls[condition].values()):
                candidates.append(condition)
    write_json(out / "report/results.json", {"complete": True, "scope": "Three-seed development screening, no final claims", "estimates": estimates,
               "comparisons_vs_comm_reference": comparisons, "communication_controls": controls, "descriptive_candidates_for_review": candidates,
               "per_seed": rows, "final_scenarios_opened": False})
    with (out / "report/per_seed.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Screening payload và router paper-v4", "", f"{len(protocol['training_seeds'])} fresh seed/condition, {protocol['steps_per_run']:,} transitions/run. Development assessment, không phải final superiority evidence. Final dự trữ chưa mở.", "",
             "| Điều kiện | Selected return | Min-rate | Normal − off | Normal − zero |", "|---|---:|---:|---:|---:|"]
    for condition in protocol["conditions"]:
        values = estimates[f"{condition}/selected"]
        control = controls.get(condition)
        off = f"{control['comm_off']['mean']:+.3f}" if control else "—"
        zero = f"{control['content_zero']['mean']:+.3f}" if control else "—"
        lines.append(f"| {condition} | {values['net_return']['mean']:.3f} | {values['min_user_rate']['mean']:.3f} | {off} | {zero} |")
    lines += ["", "## So với communication reference"]
    for condition, effect in comparisons.items():
        lines.append(f"- {condition}: {effect['mean']:+.3f}, CI95 [{effect['ci95'][0]:+.3f}, {effect['ci95'][1]:+.3f}], thắng {effect['positive_training_pairs']}/{len(protocol['training_seeds'])} cặp seed.")
    lines += ["", f"Nhánh đạt tiêu chí screening mô tả để xem xét: {', '.join(candidates) if candidates else 'chưa có'}. Cần xem trade-off sensing/outage, không tự động mở final hay kết luận superiority.", "",
              "Số liệu từng seed/treatment: per_seed.csv; uncertainty và controls: results.json. Checkpoint chọn trên development và khóa trước assessment; zero payload giữ lịch gửi/metadata/accounting."]
    (out / "report/KET_QUA_SCREENING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs_marl/paper_v4/screening")
    parser.add_argument("--mode", choices=("controller", "train", "evaluate"), default="controller")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.mode != "controller":
        lock = json.loads((out / "protocol_locked.json").read_text())
        require_sources(lock)
        (train if args.mode == "train" else evaluate)(out, args.condition, args.seed, lock)
        return 0
    if out.exists():
        raise RuntimeError("Preserve existing campaign; choose fresh output")
    protocol = json.loads(DEFAULT_PROTOCOL.read_text())
    if args.smoke:
        protocol.update(stage="pipeline_smoke", steps_per_run=2048, training_seeds=[97], total_training_transitions=8192,
                        development_seeds=protocol["development_seeds"][:3], assessment_seeds=protocol["assessment_seeds"][:3])
    lock = {"protocol": protocol, "source_sha256": source_digest(), "seed_audit": validate_split(protocol),
            "history_audit": history_audit(protocol), "matched_initialization": initialization_audit(protocol),
            "development_diagnostic_sha256": sha(ROOT / "outputs_marl/paper_v4/diagnostics/report/results.json")}
    require_sources(lock)
    out.mkdir(parents=True)
    write_json(out / "protocol_locked.json", lock)
    paths = list((ROOT / "momgrap").rglob("*.py")) + list((ROOT / "experiments_marl").rglob("*.py")) + list((ROOT / "experiments_marl_v4").glob("*.py"))
    paths += [DEFAULT_PROTOCOL, Path(__file__), ROOT / "scripts_marl/run_paper_improvements.py", ROOT / "scripts_marl/paper_seed_audit.py",
              ROOT / "scripts_marl/evaluate_paper_v2.py", ROOT / "scripts_marl/report_paper_v2.py", ROOT / "configs_marl/paper_v2.yaml", ROOT / "configs_marl/paper_v3_screening.json"]
    for path in paths:
        target = out / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    state = {"experiment_version": VERSION, "pid": os.getpid(), "start_unix": time.time(), "complete": False, "stage": "training",
             "planned_runs": len(protocol["conditions"]) * len(protocol["training_seeds"]), "source_sha256": lock["source_sha256"],
             "successful_training": [], "successful_evaluation": [], "failed": [], "final_scenarios_opened": False,
             "protocol_lock_sha256": sha(out / "protocol_locked.json")}
    def run(mode, condition, seed):
        path = out / f"{mode}_{condition}_seed_{seed}.log"
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with path.open("w", encoding="utf-8") as handle:
            code = subprocess.run([sys.executable, "-u", str(Path(__file__)), "--mode", mode, "--condition", condition, "--seed", str(seed), "--out", str(out)],
                                  cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0).returncode
        return {"condition": condition, "seed": seed, "returncode": code, "log": str(path)}
    jobs = [(c, s) for c in protocol["conditions"] for s in protocol["training_seeds"]]
    for mode, key in (("train", "successful_training"), ("evaluate", "successful_evaluation")):
        state["stage"] = "training" if mode == "train" else "assessment"
        if mode == "evaluate":
            require_sources(lock)
            state["selection_manifest_sha256"] = lock_checkpoints(out, lock)
        write_json(out / "campaign_status.json", state)
        with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
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
    require_sources(lock)
    report(out, protocol)
    state.update(complete=True, stage="complete", end_unix=time.time())
    write_json(out / "campaign_status.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

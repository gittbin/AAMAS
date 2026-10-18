"""Fresh multi-seed confirmation; freeze selected hashes before opening final data."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments_marl.paper_v3 import (source_digest, load_experiment_policy, training_seed,
                                     make_config, ImprovementEnv, ImprovementTrainer)
from momgrap.marl.normalization import FeatureProjection
from scripts_marl.run_paper_improvements import write_json, validate_split, train_worker
from scripts_marl.evaluate_paper_v2 import rollout
from scripts_marl.report_paper_v2 import nested_interval, exact_test, holm

DEFAULT_PROTOCOL = ROOT / "configs_marl/paper_v3_confirmation.json"
VERSION = "paper-v3-confirmation-2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confirmation_digest():
    digest = hashlib.sha256(source_digest().encode())
    for path in [DEFAULT_PROTOCOL, Path(__file__).resolve(), ROOT / "scripts_marl/evaluate_paper_v2.py",
                 ROOT / "scripts_marl/report_paper_v2.py"]:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_sources(lock):
    if lock["engine_sha256"] != source_digest() or lock["confirmation_sha256"] != confirmation_digest():
        raise RuntimeError("Locked confirmation source/config changed; preserve this campaign and start a new one")


def check_final_boundary(protocol):
    """Ensure all previously recorded v2/v3 data remain outside reserved final IDs."""
    final = set(range(protocol["reserved_final_seed_start"], protocol["reserved_final_seed_start"] + protocol["reserved_final_scenarios"]))
    paths = list((ROOT / "outputs_marl/paper_v2/campaign").rglob("results.json"))
    paths += list((ROOT / "outputs_marl/paper_v3").glob("*/evaluation/*/seed_*/results.json"))
    for path in paths:
        data = json.loads(path.read_text())
        def visit(value):
            if isinstance(value, dict):
                if "scenario_seed" in value and value["scenario_seed"] in final:
                    raise ValueError(f"Final scenario was already evaluated in {path}")
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(data)
    for path in (ROOT / "outputs_marl/paper_v3").glob("*/train/*/seed_*/episode_seeds.csv"):
        with path.open(newline="") as handle:
            if final & {int(row["scenario_seed"]) for row in csv.DictReader(handle)}:
                raise ValueError(f"Final scenario occurs in prior training log {path}")
    return {"passed": True, "prior_result_manifests_checked": len(paths), "reserved_final_ids": sorted(final)}


def check_training_history(protocol):
    count = protocol["steps_per_run"] // 800 + 2
    proposed = {training_seed(s, i, e, 8) for s in protocol["training_seeds"] for e in range(count) for i in range(8)}
    paths = list((ROOT / "outputs_marl/paper_v3").glob("*/train/*/seed_*/episode_seeds.csv"))
    for path in paths:
        with path.open(newline="") as handle:
            prior = {int(row["scenario_seed"]) for row in csv.DictReader(handle)}
        if proposed & prior:
            raise ValueError(f"New training replicate overlaps previous paper-v3 scenarios in {path}")
    return {"passed": True, "prior_training_logs_checked": len(paths), "new_training_ids_enumerated": len(proposed)}


def audit_matched_initialization(protocol):
    torch.set_num_threads(1)
    hashes = {}
    for seed in protocol["training_seeds"]:
        reference = None
        initial_seeds = None
        for condition in protocol["conditions"]:
            cfg = make_config(condition, seed, protocol["steps_per_run"])
            trainer = ImprovementTrainer(cfg, lambda i: ImprovementEnv(cfg), condition, seed=seed)
            digest = hashlib.sha256()
            for prefix, module in [(a, actor) for a, actor in trainer.actors.items()] + [("critic", trainer.critic)]:
                for name, value in module.state_dict().items():
                    digest.update(f"{prefix}/{name}".encode())
                    digest.update(value.detach().cpu().numpy().tobytes())
            value = digest.hexdigest()
            scenarios = [env._seed for env in trainer.envs]
            for env in trainer.envs:
                env.close()
            if reference is not None and (value != reference or scenarios != initial_seeds):
                raise ValueError("Declared matching fails initial network/scenario check")
            reference, initial_seeds = value, scenarios
        hashes[str(seed)] = {"initial_actor_critic_sha256": reference, "initial_scenario_seeds": initial_seeds}
    return {"passed": True, "independent_training_pairs": len(hashes), "pairs": hashes}


def lock_selected_checkpoints(out, lock):
    protocol = lock["protocol"]
    selected = {}
    actual_ids = {}
    protected = set(protocol["development_seeds"]) | set(protocol["assessment_seeds"])
    protected |= set(range(protocol["reserved_final_seed_start"], protocol["reserved_final_seed_start"] + protocol["reserved_final_scenarios"]))
    for condition in protocol["conditions"]:
        for seed in protocol["training_seeds"]:
            folder = out / "train" / condition / f"seed_{seed}"
            status = json.loads((folder / "status.json").read_text())
            if not status["training_complete"] or status["actual_transitions"] != protocol["steps_per_run"]:
                raise ValueError("Every fresh training run must finish before final selection is locked")
            if status["source_sha256"] != lock["engine_sha256"]:
                raise ValueError("Training source digest mismatch")
            with (folder / "episode_seeds.csv").open(newline="") as handle:
                seeds = [int(row["scenario_seed"]) for row in csv.DictReader(handle)]
            ids = set(seeds)
            if len(ids) != len(seeds) or ids & protected:
                raise ValueError("Actual training episode seeds overlap or repeat")
            if seed in actual_ids and actual_ids[seed] != ids:
                raise ValueError("Algorithm conditions did not receive matched training scenario seeds")
            actual_ids[seed] = ids
            entry = {"selected_step": status["best_step"], "development_return": status["best_development_return"],
                     "selected_checkpoint_sha256": sha(folder / "checkpoint_best.pt"),
                     "final_checkpoint_sha256": sha(folder / "checkpoint_final.pt")}
            if (entry["selected_checkpoint_sha256"] != status["checkpoint_best_sha256"] or
                    entry["final_checkpoint_sha256"] != status["checkpoint_final_sha256"]):
                raise ValueError("Checkpoint hash changed after training")
            selected[f"{condition}/seed_{seed}"] = entry
    values = list(actual_ids.values())
    if any(values[i] & values[j] for i in range(len(values)) for j in range(i)):
        raise ValueError("Different training replicates share scenario seeds")
    manifest = {"campaign_version": VERSION, "confirmation_sha256": lock["confirmation_sha256"],
                "locked_unix": time.time(), "final_scenarios_opened_at_lock": False,
                "actual_episode_seed_audit_passed": True, "selected": selected}
    path = out / "selected_checkpoints_locked.json"
    if path.exists():
        raise RuntimeError("Selected checkpoint manifest already exists; never overwrite final selection")
    write_json(path, manifest)
    return sha(path)


def load_selected(folder, kind, expected_hash):
    path = folder / f"checkpoint_{kind}.pt"
    if sha(path) != expected_hash:
        raise ValueError("Checkpoint hash differs from pre-final locked selection")
    cfg, policy, ck = load_experiment_policy(path, expected_condition=folder.parent.name)
    if ck["experiment_settings"]["actor_projection"] != "raw":
        raise ValueError("This confirmation supports the two selected raw-projection conditions only")
    projection = FeatureProjection(2 * cfg.physical.K * cfg.physical.M, 8, seed=0,
                                   device=cfg.physical.device, dtype=cfg.physical.dtype_real)
    if not torch.equal(projection.matrix.cpu(), ck["projection"]["matrix"].cpu()):
        raise ValueError("Evaluator projection differs from the selected checkpoint")
    return cfg, policy


def evaluate_worker(out, condition, seed, lock):
    torch.set_num_threads(1)
    verify_sources(lock)
    campaign = json.loads((out / "campaign_status.json").read_text())
    manifest_path = out / "selected_checkpoints_locked.json"
    if not campaign["final_scenarios_opened"] or sha(manifest_path) != campaign["selection_manifest_sha256"]:
        raise RuntimeError("Final evaluation requires the complete pre-final checkpoint lock")
    entry = json.loads(manifest_path.read_text())["selected"][f"{condition}/seed_{seed}"]
    protocol = lock["protocol"]
    final_ids = list(range(protocol["reserved_final_seed_start"], protocol["reserved_final_seed_start"] + protocol["reserved_final_scenarios"]))
    folder = out / "train" / condition / f"seed_{seed}"
    evaluation = out / "evaluation" / condition / f"seed_{seed}"
    evaluation.mkdir(parents=True, exist_ok=True)
    if (evaluation / "results.json").exists():
        raise RuntimeError("Existing final evaluation must not be overwritten/reselected")
    cfg, policy = load_selected(folder, "best", entry["selected_checkpoint_sha256"])
    treatments = {"selected": {"episodes": rollout(cfg, policy, {}, final_ids, evaluation / "selected.csv")}}
    if condition == "comm_reference":
        treatments["comm_off"] = {"episodes": rollout(cfg, policy, {"comm_mode": "none"}, final_ids, evaluation / "comm_off.csv")}
        with (evaluation / "selected.csv").open(newline="") as handle:
            normal = list(csv.DictReader(handle))
        replay = {(int(row["scenario_seed"]), int(row["step"])):
                  {a: int(row[f"comm_action_{a}"]) for a in ("bs_agent", "ris_agent", "ma_agent")} for row in normal}
        treatments["content_zero"] = {"episodes": rollout(cfg, policy, {"corruption": "zero"}, final_ids,
                                                           evaluation / "content_zero.csv", replay=replay)}
        with (evaluation / "content_zero.csv").open(newline="") as handle:
            altered = list(csv.DictReader(handle))
        accounting = ("scenario_seed", "step", "requested", "eligible", "delivered", "budget_dropped", "graph_rejected")
        if any(any(a[k] != b[k] for k in accounting) for a, b in zip(normal, altered)) or len(normal) != len(altered):
            raise ValueError("Payload intervention changed the locked sending/delivery pattern")
    cfg, final_policy = load_selected(folder, "final", entry["final_checkpoint_sha256"])
    treatments["final_checkpoint"] = {"episodes": rollout(cfg, final_policy, {}, final_ids, evaluation / "final_checkpoint.csv")}
    for treatment in treatments.values():
        if not all(np.isfinite(v) for e in treatment["episodes"] for v in e.values() if isinstance(v, (int, float))):
            raise ValueError("Nonfinite final evaluation episode")
    write_json(evaluation / "results.json", {"complete": True, "campaign_version": VERSION, "condition": condition,
        "training_seed": seed, "confirmation_sha256": lock["confirmation_sha256"], "checkpoint_lock": entry,
        "scenario_ids": final_ids, "treatments": treatments})
    status = json.loads((folder / "status.json").read_text())
    status.update(complete=True, stage="complete")
    write_json(folder / "status.json", status)


def build_report(out, protocol):
    data = {}
    rows = []
    metrics = ("net_return", "min_user_rate", "illumination_db", "outage_fraction", "delivered", "reconfiguration")
    for condition in protocol["conditions"]:
        data[condition] = []
        for seed in protocol["training_seeds"]:
            item = json.loads((out / "evaluation" / condition / f"seed_{seed}" / "results.json").read_text())
            if not item["complete"] or item["training_seed"] != seed or item["condition"] != condition:
                raise ValueError("Incomplete confirmation evaluation")
            data[condition].append(item)
            for name, treatment in item["treatments"].items():
                rows.append({"condition": condition, "seed": seed, "treatment": name,
                             **{metric: float(np.mean([e[metric] for e in treatment["episodes"]])) for metric in metrics}})
    def matrix(key, metric):
        condition, treatment = key.split("/")
        expected = list(range(protocol["reserved_final_seed_start"], protocol["reserved_final_seed_start"] + protocol["reserved_final_scenarios"]))
        eps = [item["treatments"][treatment]["episodes"] for item in data[condition]]
        if any([e["scenario_seed"] for e in values] != expected for values in eps):
            raise ValueError("Final scenario IDs differ across policies")
        return np.array([[e[metric] for e in values] for values in eps])
    estimates = {key: {metric: nested_interval(matrix(key, metric), seed=protocol["statistics_seed"]) for metric in metrics}
                 for condition in data for key in [f"{condition}/{t}" for t in data[condition][0]["treatments"]]}
    comparisons = []
    for hypothesis in protocol["primary_hypotheses"]:
        a, b = matrix(hypothesis["left"], hypothesis["metric"]), matrix(hypothesis["right"], hypothesis["metric"])
        paired = hypothesis["paired_training_policies"]
        comparisons.append({**hypothesis, **nested_interval(a, b, paired=paired, seed=protocol["statistics_seed"]),
                            "exact_p": exact_test(a, b, paired=paired)})
    for comparison, adjusted in zip(comparisons, holm([c["exact_p"] for c in comparisons])):
        comparison["holm_p"] = adjusted
    exploratory = {metric: nested_interval(matrix("comm_reference/selected", metric), matrix("comm_reference/content_zero", metric),
                    paired=True, seed=protocol["statistics_seed"]) for metric in metrics} if "comm_reference" in data else {}
    report = out / "report"
    report.mkdir(exist_ok=True)
    write_json(report / "results.json", {"campaign_version": VERSION, "scope": "Locked independent confirmation on untouched final scenarios",
        "training_policies_per_condition": len(protocol["training_seeds"]), "final_scenarios_per_policy": protocol["reserved_final_scenarios"],
        "estimates": estimates, "primary_comparisons": comparisons, "exploratory_selected_minus_content_zero": exploratory})
    with (report / "per_seed.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Kết quả xác nhận communication paper-v3", "",
             f"{len(protocol['training_seeds'])} training seed mới mỗi điều kiện, {protocol['steps_per_run']:,} transitions mỗi run; "
             f"checkpoint chọn trên development và khóa trước khi mở {protocol['reserved_final_scenarios']} scenario final. "
             "Hai giả thuyết chính đã định trước; Holm điều chỉnh trong cùng family. Kết quả không dùng để chọn lại checkpoint. "
             "Các CI và kiểm định là hai mô tả khác nhau; nonsignificance không phải equivalence.", "",
             "| Điều kiện/treatment | Return [CI95] | Min-rate [CI95] | Delivered/interval |", "|---|---:|---:|---:|"]
    for key, values in estimates.items():
        def fmt(metric):
            v = values[metric]
            return f"{v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
        lines.append(f"| {key} | {fmt('net_return')} | {fmt('min_user_rate')} | {values['delivered']['mean']:.3f} |")
    lines += ["", "## So sánh chính", ""]
    for c in comparisons:
        lines.append(f"- {c['name']}: chênh lệch {c['mean']:+.3f}, CI95 [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], "
                     f"exact p={c['exact_p']:.5f}, Holm p={c['holm_p']:.5f}.")
    lines += ["", "## Từng seed", "", "| Điều kiện | Seed | Treatment | Return | Min-rate |", "|---|---:|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['condition']} | {row['seed']} | {row['treatment']} | {row['net_return']:.3f} | {row['min_user_rate']:.3f} |")
    lines += ["", "Payload zero là đối chứng exploratory, giữ nguyên send actions/delivery accounting; checkpoint cuối cũng là exploratory. "
              "Không so trực tiếp return với các tập scenario hoặc budget train khác. Artifact giữ đầy đủ trace và hash; "
              "đợt này xác nhận communication, không xác nhận lại tất cả claims của paper."]
    (report / "KET_QUA_XAC_NHAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    keys = ["no_comm/selected", "comm_reference/selected", "comm_reference/comm_off", "comm_reference/content_zero"]
    labels = ["No communication", "Communication", "Same comm policy, off", "Same send pattern, zero payload"]
    for axis, metric, title in zip(axes, ["net_return", "min_user_rate"], ["Episode net return", "Minimum-user rate (bps/Hz)"]):
        for index, key in enumerate(keys):
            v = estimates[key][metric]
            axis.errorbar(v["mean"], index, xerr=[[max(0, v["mean"] - v["ci95"][0])],
                                                [max(0, v["ci95"][1] - v["mean"])]], fmt="o", capsize=3)
        axis.set_xlabel(title)
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(len(keys)), labels)
    axes[0].invert_yaxis()
    fig.suptitle(f"Locked confirmation: {len(protocol['training_seeds'])} training policies; {protocol['reserved_final_scenarios']} shared final scenarios")
    fig.text(0.5, 0.015, "95% nested bootstrap intervals; primary Holm-adjusted tests are reported separately; zero payload is exploratory.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(report / "confirmation_comparison.pdf")
    fig.savefig(report / "confirmation_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["campaign", "train", "evaluate"], default="campaign")
    parser.add_argument("--out", default="outputs_marl/paper_v3/confirmation_v2")
    parser.add_argument("--condition", choices=["no_comm", "comm_reference"])
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.mode != "campaign":
        lock = json.loads((out / "protocol_locked.json").read_text())
        state = json.loads((out / "campaign_status.json").read_text())
        if sha(out / "protocol_locked.json") != state["protocol_lock_sha256"]:
            raise ValueError("Actual protocol lock changed after campaign launch")
        verify_sources(lock)
        if args.mode == "train":
            train_worker(out, args.condition, args.seed, lock["protocol"], lock["engine_sha256"])
        else:
            evaluate_worker(out, args.condition, args.seed, lock)
        return 0
    if (out / "campaign_status.json").exists():
        raise RuntimeError("Existing confirmation campaign; preserve it and use a new directory")
    protocol = json.loads(DEFAULT_PROTOCOL.read_text())
    if args.smoke:
        protocol.update(stage="confirmation_pipeline_smoke", steps_per_run=2048, training_seeds=[92,93],
                        total_training_transitions=8192, development_seeds=protocol["development_seeds"][:3],
                        reserved_final_seed_start=9100000, reserved_final_scenarios=3)
    lock = {"protocol": protocol, "engine_sha256": source_digest(), "confirmation_sha256": confirmation_digest(),
            "seed_audit": validate_split(protocol), "prior_final_boundary_audit": check_final_boundary(protocol),
            "new_training_history_audit": check_training_history(protocol),
            "matched_initialization_audit": audit_matched_initialization(protocol)}
    write_json(out / "protocol_locked.json", lock)
    # Snapshot the exact engine and confirmation files without editing screening evidence.
    paths = list((ROOT / "momgrap").rglob("*.py")) + list((ROOT / "experiments_marl").rglob("*.py"))
    paths += [Path(__file__).resolve(), DEFAULT_PROTOCOL, ROOT / "scripts_marl/run_paper_improvements.py",
              ROOT / "scripts_marl/evaluate_paper_v2.py", ROOT / "scripts_marl/report_paper_v2.py",
              ROOT / "scripts_marl/paper_seed_audit.py", ROOT / "configs_marl/paper_v2.yaml",
              ROOT / "configs_marl/paper_v3_screening.json"]
    for path in paths:
        target = out / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    state = {"campaign_version": VERSION, "pid": os.getpid(), "start_unix": time.time(), "complete": False,
             "stage": "training", "planned_runs": len(protocol["conditions"]) * len(protocol["training_seeds"]),
             "successful_training": [], "successful_evaluation": [], "failed": [], "final_scenarios_opened": False,
             "protocol_lock_sha256": sha(out / "protocol_locked.json"),
             "confirmation_sha256": lock["confirmation_sha256"], "engine_sha256": lock["engine_sha256"]}
    write_json(out / "campaign_status.json", state)
    jobs = [(c, s) for c in protocol["conditions"] for s in protocol["training_seeds"]]
    def run(mode, condition, seed):
        path = out / f"{mode}_{condition}_seed_{seed}.log"
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", mode, "--out", str(out),
                   "--condition", condition, "--seed", str(seed)]
        with path.open("a", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return {"condition": condition, "seed": seed, "returncode": result.returncode, "log": str(path)}
    for mode, key in [("train", "successful_training"), ("evaluate", "successful_evaluation")]:
        if mode == "evaluate":
            verify_sources(lock)
            state["selection_manifest_sha256"] = lock_selected_checkpoints(out, lock)
            state.update(stage="final_evaluation", final_scenarios_opened=True, final_opened_unix=time.time())
            write_json(out / "campaign_status.json", state)
        with ThreadPoolExecutor(max_workers=args.workers or protocol["workers"]) as pool:
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
    verify_sources(lock)
    build_report(out, protocol)
    state.update(complete=True, stage="complete", end_unix=time.time())
    write_json(out / "campaign_status.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

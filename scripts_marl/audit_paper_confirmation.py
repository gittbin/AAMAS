"""Recheck locked confirmation artifacts without running or selecting policies."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts_marl.run_paper_confirmation import sha, verify_sources
from scripts_marl.report_paper_v2 import nested_interval


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-8)


def audit(out):
    state = read_json(out / "campaign_status.json")
    lock = read_json(out / "protocol_locked.json")
    manifest = read_json(out / "selected_checkpoints_locked.json")
    report = read_json(out / "report/results.json")
    protocol = lock["protocol"]
    verify_sources(lock)
    require(state["complete"] and state["stage"] == "complete" and not state["failed"], "Campaign incomplete/failed")
    require(sha(out / "protocol_locked.json") == state["protocol_lock_sha256"], "Protocol hash changed")
    require(sha(out / "selected_checkpoints_locked.json") == state["selection_manifest_sha256"], "Selection lock changed")
    require(not manifest["final_scenarios_opened_at_lock"] and manifest["locked_unix"] < state["final_opened_unix"], "Selection not locked before final")
    final = list(range(protocol["reserved_final_seed_start"], protocol["reserved_final_seed_start"] + protocol["reserved_final_scenarios"]))
    protected = set(final + protocol["development_seeds"] + protocol["assessment_seeds"])
    matrices, actual_ids = {}, {}
    transitions = episodes = trace_rows = train_cells = 0
    max_violation = 0.0
    metrics = {"net_return": "net_reward", "min_user_rate": "min_user_rate", "illumination_db": "illumination_db",
               "outage_fraction": "outage_fraction", "delivered": "delivered", "reconfiguration": "reconfiguration"}
    for condition in protocol["conditions"]:
        for seed in protocol["training_seeds"]:
            folder = out / "train" / condition / f"seed_{seed}"
            status = read_json(folder / "status.json")
            entry = manifest["selected"][f"{condition}/seed_{seed}"]
            require(status["complete"] and status["training_complete"] and status["actual_transitions"] == protocol["steps_per_run"], "Incomplete training")
            require(status["source_sha256"] == lock["engine_sha256"], "Training source mismatch")
            transitions += status["actual_transitions"]
            for kind, key in (("best", "selected_checkpoint_sha256"), ("final", "final_checkpoint_sha256")):
                require(sha(folder / f"checkpoint_{kind}.pt") == entry[key] == status[f"checkpoint_{kind}_sha256"], "Checkpoint changed")
            evaluations = read_csv(folder / "eval.csv")
            chosen = max(evaluations, key=lambda r: (float(r["eval_return_mean"]), -int(r["global_step"])))
            require(int(chosen["global_step"]) == entry["selected_step"] and close(chosen["eval_return_mean"], entry["development_return"]), "Selection does not match development criterion")
            resets = read_csv(folder / "episode_seeds.csv")
            ids = [int(r["scenario_seed"]) for r in resets]
            require(len(set(ids)) == len(ids) and not set(ids) & protected, "Actual train/test overlap or repeated resets")
            require(seed not in actual_ids or actual_ids[seed] == set(ids), "Training pairs have different scenarios")
            actual_ids[seed] = set(ids)
            for row in read_csv(folder / "train.csv"):
                require(all(math.isfinite(float(v)) for v in row.values()), "Nonfinite training log")
                train_cells += len(row)
            evaluation = out / "evaluation" / condition / f"seed_{seed}"
            data = read_json(evaluation / "results.json")
            require(data["complete"] and data["checkpoint_lock"] == entry and data["confirmation_sha256"] == lock["confirmation_sha256"], "Evaluation provenance mismatch")
            require(data["scenario_ids"] == final, "Final scenario mismatch")
            expected = {"selected", "final_checkpoint"} | ({"comm_off", "content_zero"} if condition == "comm_reference" else set())
            require(set(data["treatments"]) == expected, "Missing treatments")
            traces = {}
            for treatment, item in data["treatments"].items():
                eps = item["episodes"]
                require([e["scenario_seed"] for e in eps] == final, "Episode scenario mismatch")
                rows = read_csv(evaluation / f"{treatment}.csv")
                require([(int(r["scenario_seed"]), int(r["step"])) for r in rows] == [(s, t) for s in final for t in range(100)], "Trace missing/duplicated/out of order")
                for row in rows:
                    require(all(math.isfinite(float(v)) for k, v in row.items() if k != "regime"), "Nonfinite trace")
                    require(int(row["requested"]) == int(row["eligible"]) + int(row["graph_rejected"]), "Request accounting mismatch")
                    require(int(row["eligible"]) == int(row["delivered"]) + int(row["budget_dropped"]), "Delivery accounting mismatch")
                    max_violation = max(max_violation, float(row["violation_after"]))
                for index, ep in enumerate(eps):
                    require(ep["steps"] == 100 and all(math.isfinite(v) for v in ep.values() if isinstance(v, (int, float))), "Invalid episode")
                    chunk = rows[index * 100:(index + 1) * 100]
                    for metric, column in metrics.items():
                        total = math.fsum(float(r[column]) for r in chunk)
                        require(close(ep[metric], total if metric == "net_return" else total / 100), f"Episode/trace mismatch: {metric}")
                key = f"{condition}/{treatment}"
                for metric in metrics:
                    matrices.setdefault((key, metric), []).append([e[metric] for e in eps])
                traces[treatment] = rows
                episodes += len(eps)
                trace_rows += len(rows)
            if condition == "comm_reference":
                keys = ("scenario_seed", "step", "requested", "eligible", "delivered", "budget_dropped", "graph_rejected",
                        "comm_action_bs_agent", "comm_action_ris_agent", "comm_action_ma_agent")
                require(all(all(a[k] == b[k] for k in keys) for a, b in zip(traces["selected"], traces["content_zero"])), "Zero-payload changed sends/delivery")
    sets = list(actual_ids.values())
    require(all(not sets[i] & sets[j] for i in range(len(sets)) for j in range(i)), "Training replicates overlap")
    for (key, metric), values in matrices.items():
        require(close(report["estimates"][key][metric]["mean"], math.fsum(map(math.fsum, values)) / (len(values) * len(final))), "Report mean mismatch")
    comparisons, raw_p = [], []
    for recorded, hypothesis in zip(report["primary_comparisons"], protocol["primary_hypotheses"], strict=True):
        left = matrices[(hypothesis["left"], hypothesis["metric"])]
        right = matrices[(hypothesis["right"], hypothesis["metric"])]
        differences = [math.fsum(a - b for a, b in zip(x, y)) / len(final) for x, y in zip(left, right)]
        observed = abs(math.fsum(differences) / len(differences))
        permuted = [abs(math.fsum(d * s for d, s in zip(differences, signs)) / len(differences)) for signs in itertools.product((-1, 1), repeat=len(differences))]
        p = sum(v >= observed - 1e-12 for v in permuted) / len(permuted)
        ci = nested_interval(left, right, paired=True, seed=protocol["statistics_seed"])
        require(close(p, recorded["exact_p"]) and close(ci["mean"], recorded["mean"]) and all(close(a, b) for a, b in zip(ci["ci95"], recorded["ci95"])), "Primary statistics mismatch")
        raw_p.append(p)
        comparisons.append({"name": hypothesis["name"], "seed_differences": dict(zip(map(str, protocol["training_seeds"]), differences)), "positive_pairs": sum(d > 1e-10 for d in differences), "negative_pairs": sum(d < -1e-10 for d in differences)})
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(raw_p)), key=lambda i: raw_p[i])):
        previous = max(previous, min(1.0, (len(raw_p) - rank) * raw_p[index]))
        require(close(previous, report["primary_comparisons"][index]["holm_p"]), "Holm mismatch")
    require(episodes == 2400 and trace_rows == 240000 and transitions == protocol["total_training_transitions"], "Unexpected totals")
    return {"audit_passed": True, "checked_unix": time.time(), "campaign_version": state["campaign_version"],
            "training_runs": len(manifest["selected"]), "training_transitions": transitions, "training_numeric_cells_checked": train_cells,
            "evaluation_episodes": episodes, "trace_rows": trace_rows, "nonfinite_cells": 0,
            "source_and_checkpoint_hashes_valid": True, "checkpoint_selection_verified": True,
            "selection_locked_before_final": True, "actual_training_disjoint_from_development_and_final": True,
            "matched_training_scenarios_valid": True, "payload_zero_preserves_sends_and_delivery": True,
            "trace_episode_report_means_match": True, "primary_ci_exact_p_and_holm_verified": True,
            "max_violation_after": max_violation, "runtime_minutes": (state["end_unix"] - state["start_unix"]) / 60,
            "paired_comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs_marl/paper_v3/confirmation_v2")
    args = parser.parse_args()
    result = audit(args.out)
    (args.out / "report/audit_final.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

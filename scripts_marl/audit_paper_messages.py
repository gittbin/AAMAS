"""Validate diagnostic and message-screening evidence without rerunning policies."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts_marl.run_paper_confirmation import sha, verify_sources
from scripts_marl.run_paper_messages import require_sources
from scripts_marl.run_paper_improvements import write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require(value, message):
    if not value:
        raise ValueError(message)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-8)


def trace(path, episodes, ids):
    records = rows(path)
    require([(int(r["scenario_seed"]), int(r["step"])) for r in records] == [(s, t) for s in ids for t in range(100)], "Trace IDs/steps missing or duplicate")
    for row in records:
        require(all(math.isfinite(float(v)) for k, v in row.items() if k != "regime"), "Nonfinite trace")
        require(int(row["requested"]) == int(row["eligible"]) + int(row["graph_rejected"]), "Request accounting")
        require(int(row["eligible"]) == int(row["delivered"]) + int(row["budget_dropped"]), "Delivery accounting")
    require([e["scenario_seed"] for e in episodes] == ids, "Episode IDs")
    for i, ep in enumerate(episodes):
        chunk = records[i * 100:(i + 1) * 100]
        require(all(math.isfinite(v) for v in ep.values() if isinstance(v, (float, int))), "Nonfinite episode")
        require(close(ep["net_return"], math.fsum(float(r["net_reward"]) for r in chunk)), "Trace/episode return")
        for metric in ("min_user_rate", "delivered", "reconfiguration"):
            require(close(ep[metric], math.fsum(float(r[metric]) for r in chunk) / 100), "Trace/episode metric")
    return records


def accounting_matches(a, b):
    keys = ("scenario_seed", "step", "requested", "eligible", "delivered", "budget_dropped", "graph_rejected",
            "comm_action_bs_agent", "comm_action_ris_agent", "comm_action_ma_agent")
    require(len(a) == len(b) and all(all(x[k] == y[k] for k in keys) for x, y in zip(a, b)), "Zero-payload sends/delivery mismatch")


def audit_diagnostics(out):
    source = ROOT / "outputs_marl/paper_v3/confirmation_v2"
    verify_sources(read(source / "protocol_locked.json"))
    selected = read(source / "selected_checkpoints_locked.json")["selected"]
    report = read(out / "report/results.json")
    episodes = traces = observations = 0
    for seed in range(30, 38):
        folder = out / f"seed_{seed}"
        d = read(folder / "results.json")
        expected_hash = selected[f"comm_reference/seed_{seed}"]["selected_checkpoint_sha256"]
        require(d["complete"] and d["checkpoint_sha256"] == expected_hash == sha(source / "train/comm_reference" / f"seed_{seed}/checkpoint_best.pt"), "Diagnostic checkpoint")
        ids = list(range(1000000, 1000030))
        records = {}
        for treatment, eps in d["treatments"].items():
            records[treatment] = trace(folder / f"{treatment}.csv", eps, ids)
            episodes += len(eps)
            traces += len(records[treatment])
        accounting_matches(records["normal"], records["zero"])
        normal_mean = math.fsum(e["net_return"] for e in d["treatments"]["normal"]) / 30
        status = read(source / "train/comm_reference" / f"seed_{seed}/status.json")
        require(close(normal_mean, status["best_development_return"]), "Passive diagnostic does not reproduce selected development score")
        observed = rows(folder / "inbox.csv")
        require(len(observed) == 9000, "Incomplete inbox evidence")
        for row in observed:
            require(all(math.isfinite(float(v)) for k, v in row.items() if k != "receiver"), "Nonfinite inbox/sensitivity")
        observations += len(observed)
    require(episodes == report["episodes"] == 720, "Episode total")
    return {"audit_passed": True, "role": "development diagnostic", "episodes": episodes,
            "trace_rows": traces, "inbox_rows": observations, "nonfinite_cells": 0,
            "selected_development_returns_reproduced": True, "fixed_send_intervention_valid": True,
            "final_scenarios_opened": False}


def audit_screening(out):
    lock = read(out / "protocol_locked.json")
    require_sources(lock)
    protocol = lock["protocol"]
    state = read(out / "campaign_status.json")
    manifest = read(out / "selected_checkpoints_locked.json")
    report = read(out / "report/results.json")
    require(state["complete"] and state["stage"] == "complete" and not state["failed"], "Incomplete screening")
    require(sha(out / "protocol_locked.json") == state["protocol_lock_sha256"], "Protocol lock hash")
    require(sha(out / "selected_checkpoints_locked.json") == state["selection_manifest_sha256"], "Selection lock hash")
    protected = set(protocol["development_seeds"] + protocol["assessment_seeds"])
    protected |= set(range(protocol["reserved_final_seed_start"], protocol["reserved_final_seed_start"] + protocol["reserved_final_scenarios"]))
    resets, episodes, traces, transitions, cells = {}, 0, 0, 0, 0
    for condition in protocol["conditions"]:
        for seed in protocol["training_seeds"]:
            folder = out / "train" / condition / f"seed_{seed}"
            status = read(folder / "status.json")
            require(status["complete"] and status["actual_transitions"] == protocol["steps_per_run"], "Training incomplete")
            transitions += status["actual_transitions"]
            entry = manifest["selected"][f"{condition}/seed_{seed}"]
            for kind, key in (("best", "selected_checkpoint_sha256"), ("final", "final_checkpoint_sha256")):
                require(sha(folder / f"checkpoint_{kind}.pt") == entry[key] == status[f"checkpoint_{kind}_sha256"], "Checkpoint changed")
            logged = rows(folder / "episode_seeds.csv")
            ids = [int(r["scenario_seed"]) for r in logged]
            require(len(set(ids)) == len(ids) and not set(ids) & protected, "Train/dev/final overlap")
            require(seed not in resets or resets[seed] == set(ids), "Matching failed")
            resets[seed] = set(ids)
            development = max(rows(folder / "eval.csv"), key=lambda r: (float(r["eval_return_mean"]), -int(r["global_step"])))
            require(int(development["global_step"]) == entry["best_step"], "Selection criterion")
            for row in rows(folder / "train.csv"):
                require(all(math.isfinite(float(v)) for v in row.values()), "Nonfinite training")
                cells += len(row)
            evaluation = out / "evaluation" / condition / f"seed_{seed}"
            d = read(evaluation / "results.json")
            require(d["complete"] and d["checkpoint_lock"] == entry, "Evaluation provenance")
            records = {}
            for name, item in d["treatments"].items():
                eps = item["episodes"]
                records[name] = trace(evaluation / f"{name}.csv", eps, protocol["assessment_seeds"])
                episodes += len(eps)
                traces += len(records[name])
            if condition != "no_comm":
                accounting_matches(records["selected"], records["content_zero"])
            per_seed = next(r for r in report["per_seed"] if r["condition"] == condition and r["seed"] == seed and r["treatment"] == "selected")
            require(close(per_seed["net_return"], math.fsum(e["net_return"] for e in d["treatments"]["selected"]["episodes"]) / len(protocol["assessment_seeds"])), "Report mean")
    sets = list(resets.values())
    require(all(not sets[i] & sets[j] for i in range(len(sets)) for j in range(i)), "Training replicates overlap")
    require(transitions == protocol["total_training_transitions"], "Transition total")
    require(not state["final_scenarios_opened"] and not report["final_scenarios_opened"], "Screening opened final")
    return {"audit_passed": True, "role": protocol["stage"], "training_runs": state["planned_runs"],
            "training_transitions": transitions, "training_numeric_cells": cells, "episodes": episodes, "trace_rows": traces,
            "nonfinite_cells": 0, "checkpoint_and_protocol_hashes_valid": True, "actual_train_dev_final_disjoint": True,
            "matched_training_scenario_ids_valid": True, "fixed_send_intervention_valid": True, "final_scenarios_opened": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    result = (audit_diagnostics if args.diagnostics else audit_screening)(args.out)
    result["checked_unix"] = time.time()
    write_json(args.out / "report/audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

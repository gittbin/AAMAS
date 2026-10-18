"""Verify completed artifacts and summarize paired held-out findings."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts_marl.evaluate_paper_v2 import PROTOCOL, write_json
from scripts_marl.report_paper_v2 import nested_interval
from scripts_marl.train_paper_condition import CONDITIONS
from scripts_marl.paper_seed_audit import audit_seed_split, SEED_WARNING


def main():
    campaign = ROOT / "outputs_marl/paper_v2/campaign"
    expected_seeds = list(range(40000, 40050))
    report = json.loads((campaign / "report/results_summary.json").read_text())
    manifests, source_hashes = {}, set()
    episodes_checked, traces_checked, maximum_violation = 0, 0, 0.0
    for condition in CONDITIONS:
        manifests[condition] = []
        for seed in range(5):
            training = campaign / "train" / condition / f"seed_{seed}"
            status = json.loads((training / "status.json").read_text())
            assert status["complete"] and status["actual_transitions"] == 400384
            source_hashes.add(status["source_sha256"])
            folder = campaign / "evaluation" / condition / f"seed_{seed}"
            data = json.loads((folder / "results.json").read_text())
            assert data["complete"] and data["source_sha256"] == status["source_sha256"]
            assert data["checkpoint_sha256"] == hashlib.sha256((training / "checkpoint.pt").read_bytes()).hexdigest()
            assert [r["scenario_seed"] for r in data["treatments"]["normal"]["episodes"]] == expected_seeds
            for treatment in data["treatments"].values():
                for episode in treatment["episodes"]:
                    assert episode["steps"] == 100
                    assert all(np.isfinite(value) for value in episode.values() if isinstance(value, (int, float)))
                    episodes_checked += 1
            for name in data["treatments"]:
                path = folder / f"{name}.csv"
                assert path.exists()
                with path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                assert len(rows) == 100 * len(data["treatments"][name]["episodes"])
                for row in rows:
                    assert int(row["requested"]) == int(row["eligible"]) + int(row["graph_rejected"])
                    assert int(row["eligible"]) == int(row["delivered"]) + int(row["budget_dropped"])
                    maximum_violation = max(maximum_violation, float(row["violation_after"]))
                traces_checked += len(rows)
            if condition == "full_nocomm":
                pairs = zip(data["treatments"]["normal"]["episodes"], data["treatments"]["frozen25"]["episodes"])
                assert all(a["prefix25_sha256"] == b["prefix25_sha256"] for a, b in pairs)
            if condition == "full_comm":
                def delivery_rows(name):
                    with (folder / f"{name}.csv").open(newline="") as handle:
                        return [tuple(row[key] for key in ("scenario_seed", "step", "requested", "eligible", "delivered", "budget_dropped", "graph_rejected")) for row in csv.DictReader(handle)]
                original = delivery_rows("normal")
                assert all(delivery_rows(name) == original for name in ("content_zero", "content_random", "content_permute"))
            manifests[condition].append(data)
    assert len(source_hashes) == 1
    def matrix(condition, treatment, metric):
        return np.asarray([[episode[metric] for episode in data["treatments"][treatment]["episodes"]] for data in manifests[condition]])
    paired = {}
    for condition, treatment, control in [("full_nocomm", "normal", "frozen25"),
                                          *[("full_comm", "normal", mode) for mode in ("comm_off", "content_zero", "content_random", "content_permute")]]:
        paired[f"{condition}:{treatment}-{control}"] = {metric: nested_interval(matrix(condition, treatment, metric),
            matrix(condition, control, metric), paired=True) for metric in ("net_return", "min_user_rate", "illumination_db", "post25_min_user_rate")}
    seed_audit = audit_seed_split(campaign, expected_seeds)
    findings = {"audit_passed": seed_audit["evaluation_disjoint_from_training"] and seed_audit["development_disjoint_from_training"],
                "artifact_integrity_passed": True, "seed_split_audit": seed_audit,
                "interpretation_warning": SEED_WARNING, "training_runs": 85, "transitions": 85 * 400384,
                "heldout_episodes_checked": episodes_checked, "heldout_trace_rows_checked": traces_checked,
                "consistent_core_source_sha256": list(source_hashes)[0], "maximum_postrepair_violation": maximum_violation,
                "primary_comparisons": report["primary_comparisons"], "exploratory_paired_comparisons": paired}
    write_json(campaign / "report/audit.json", findings)
    selected = [("Full team, no comm", "full_nocomm/normal"), ("Full team, comm", "full_comm/normal"),
                ("Centralized", "centralized/normal"), ("Feed-forward", "feedforward/normal"), ("Masked actor CSI", "masked_csi/normal"),
                ("Power + MA", "power_ma/normal"), ("MA only", "ma/normal"), ("USPA no-op", "baseline/noop"),
                ("Static RAMP knee", "baseline/ramp_knee"), ("Online optimizer", "baseline/online_optimizer")]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), sharey=True)
    for axis, metric, label in zip(axes, ("min_user_rate", "net_return"), ("Mean minimum-user rate (bps/Hz)", "Episode net return")):
        for index, (name, key) in enumerate(selected):
            value = report["estimates"][key][metric]
            low, high = value["ci95"]
            axis.errorbar(value["mean"], index, xerr=[[value["mean"] - low], [high - value["mean"]]], fmt="o",
                          capsize=3, color="#c65d27" if key.startswith("baseline/") else "#2365a4", markersize=5)
        axis.set_xlabel(label)
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(len(selected)), [name for name, _ in selected])
    axes[0].invert_yaxis()
    fig.suptitle("Original evaluation: 50 shared scenarios; 5 trained policies")
    fig.text(0.5, 0.015, "Scenario 40000 overlaps training seeds 0/4: not a fully unseen test. 95% nested bootstrap intervals.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(campaign / "report/heldout_comparison.pdf")
    fig.savefig(campaign / "report/heldout_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({key: value for key, value in findings.items() if key not in ("primary_comparisons", "exploratory_paired_comparisons")}, indent=2))
    for name, values in paired.items():
        print(name, json.dumps({key: value for key, value in values.items() if key in ("net_return", "post25_min_user_rate")}))


if __name__ == "__main__":
    main()

"""Summarize development learning curves without claiming held-out results."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    campaign = ROOT / "outputs_marl/paper_v2/campaign"
    summary = {"scope": "Development evaluation only: three fixed scenarios 10000,10001,10002. Five training policies where completed; no significance claim.", "conditions": {}}
    for folder in sorted((campaign / "train").iterdir()):
        records = []
        for run in sorted(folder.glob("seed_*")):
            status = json.loads((run / "status.json").read_text())
            if not status.get("complete"):
                continue
            with (run / "eval.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            record = {"seed": status["training_seed"], "first_return": float(rows[0]["eval_return_mean"]),
                      "final_return": float(rows[-1]["eval_return_mean"]),
                      "last_four_returns": [float(row["eval_return_mean"]) for row in rows[-4:]]}
            train_path = run / "train.csv"
            with train_path.open(newline="") as handle:
                train = list(csv.DictReader(handle))
            if "diagnostics/communication_entropy" in train[0]:
                record["late_train"] = {key: float(np.mean([float(row[key]) for row in train[-20:]]))
                                        for key in ("train/messages", "train/min_rate", "train/outage", "train/reconfiguration",
                                                    "diagnostics/communication_entropy", "diagnostics/approx_kl", "diagnostics/explained_variance")}
            record["nonfinite_training_cells"] = sum(not np.isfinite(float(value)) for row in train for value in row.values())
            records.append(record)
        if not records:
            continue
        finals = np.array([r["final_return"] for r in records])
        summary["conditions"][folder.name] = {"completed_training_seeds": len(records),
            "first_development_return_mean": float(np.mean([r["first_return"] for r in records])),
            "final_development_return_mean": float(finals.mean()),
            "final_return_sd_across_training_seeds": float(finals.std(ddof=1)) if len(records) > 1 else None,
            "nonfinite_training_cells": sum(r["nonfinite_training_cells"] for r in records), "per_seed": records}
    (campaign / "development_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Development only; mean +/- sample SD across completed training policies")
    for condition, values in summary["conditions"].items():
        sd = values['final_return_sd_across_training_seeds']
        sd_text = f"{sd:.3f}" if sd is not None else "pending"
        print(f"{condition:20s} n={values['completed_training_seeds']} first={values['first_development_return_mean']:.3f} final={values['final_development_return_mean']:.3f} sd={sd_text} nonfinite={values['nonfinite_training_cells']}")
    print("Selected late-training communication diagnostics:")
    for name in ("full_nocomm", "full_comm", "price_free", "price_tenfold", "budget3_default", "budget3_free", "budget3_tenfold"):
        if name in summary["conditions"]:
            records = summary["conditions"][name]["per_seed"]
            metrics = {key: float(np.mean([record["late_train"][key] for record in records])) for key in records[0]["late_train"]}
            print(name, json.dumps(metrics))


if __name__ == "__main__":
    main()

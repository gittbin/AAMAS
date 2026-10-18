"""Descriptive diagnostics of completed data; no policy selection or training."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs_marl/paper_v3/confirmation_v2"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    return statistics.mean(values)


def main():
    protocol = json.loads((CAMPAIGN / "protocol_locked.json").read_text())["protocol"]
    result = {"role": "post-confirmation descriptive diagnostics; final data must not select new configurations",
              "seeds": [], "training_diagnostics": []}
    for seed in protocol["training_seeds"]:
        evaluation = CAMPAIGN / "evaluation/comm_reference" / f"seed_{seed}"
        data = json.loads((evaluation / "results.json").read_text())
        treatments = data["treatments"]
        selected = treatments["selected"]["episodes"]
        off = treatments["comm_off"]["episodes"]
        zero = treatments["content_zero"]["episodes"]
        rows = read_csv(evaluation / "selected.csv")
        counts = [0, 0, 0]
        destinations = [[0, 0, 0] for _ in range(3)]
        for row in rows:
            eligible = [i for i, agent in enumerate(("bs_agent", "ris_agent", "ma_agent")) if int(row[f"comm_action_{agent}"]) != 0]
            # Full graph, budget one, identical priorities => sender-index tie-break.
            accepted = eligible[:1]
            assert len(accepted) == int(row["delivered"])
            for i in accepted:
                action = int(row[f"comm_action_{('bs_agent', 'ris_agent', 'ma_agent')[i]}"])
                destination = [j for j in range(3) if j != i][action - 1]
                counts[i] += 1
                destinations[i][destination] += 1
        item = {"seed": seed,
                "on_minus_off_net": mean([a["net_return"] - b["net_return"] for a, b in zip(selected, off)]),
                "on_minus_off_before_message_cost": mean([a["return_before_message_cost"] - b["return_before_message_cost"] for a, b in zip(selected, off)]),
                "message_cost_per_episode": mean([a["return_before_message_cost"] - a["net_return"] for a in selected]),
                "on_minus_zero_net": mean([a["net_return"] - b["net_return"] for a, b in zip(selected, zero)]),
                "accepted_sender_share": {a: n / sum(counts) for a, n in zip(("BS", "RIS", "MA"), counts)},
                "accepted_sender_destination_counts": destinations,
                "regimes": {}}
        for regime in sorted({e["regime"] for e in selected}):
            pairs = [(a, b) for a, b in zip(selected, off) if a["regime"] == regime]
            item["regimes"][regime] = {"n_scenarios": len(pairs), "on_minus_off_net": mean([a["net_return"] - b["net_return"] for a, b in pairs])}
        result["seeds"].append(item)
        for condition in protocol["conditions"]:
            folder = CAMPAIGN / "train" / condition / f"seed_{seed}"
            rows = read_csv(folder / "train.csv")[-20:]
            status = json.loads((folder / "status.json").read_text())
            result["training_diagnostics"].append({"condition": condition, "seed": seed,
                "best_step": status["best_step"], "best_development_return": status["best_development_return"],
                "last_development_return": status["last_development_return"],
                "late_mean_communication_entropy": mean([float(r["diagnostics/communication_entropy"]) for r in rows]),
                "late_mean_train_delivered": mean([float(r["train/messages"]) for r in rows]),
                "late_mean_actor_clip_factor": mean([float(r["diagnostics/actor_clip_factor"]) for r in rows])})
    result["aggregate"] = {metric: mean([s[metric] for s in result["seeds"]]) for metric in
        ("on_minus_off_net", "on_minus_off_before_message_cost", "message_cost_per_episode", "on_minus_zero_net")}
    result["aggregate"]["sender_share_mean_over_policies"] = {agent: mean([s["accepted_sender_share"][agent] for s in result["seeds"]]) for agent in ("BS", "RIS", "MA")}
    result["aggregate"]["on_minus_zero_excluding_seed30_descriptive"] = mean([s["on_minus_zero_net"] for s in result["seeds"] if s["seed"] != 30])
    result["aggregate"]["late_communication_entropy"] = mean([s["late_mean_communication_entropy"] for s in result["training_diagnostics"] if s["condition"] == "comm_reference"])
    result["aggregate"]["late_train_delivered"] = mean([s["late_mean_train_delivered"] for s in result["training_diagnostics"] if s["condition"] == "comm_reference"])
    (CAMPAIGN / "report/next_steps_diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

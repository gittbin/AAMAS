"""Export policy-level results separately from evaluation-scenario variation."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts_marl.paper_seed_audit import audit_seed_split, SEED_WARNING
METRICS = ("net_return", "return_at_default_price", "return_before_message_cost",
           "min_user_rate", "sum_rate", "illumination_db", "outage_fraction",
           "requested", "eligible", "delivered", "budget_dropped", "graph_rejected",
           "reconfiguration", "post25_min_user_rate", "post25_return")
LABELS = {"full_nocomm": "Full team, không giao tiếp", "full_comm": "Full team, có giao tiếp",
          "centralized": "Centralized PPO", "feedforward": "Feed-forward", "masked_csi": "Actor che CSI",
          "power": "Power", "ris": "RIS", "ma": "MA", "power_ris": "Power + RIS", "power_ma": "Power + MA",
          "ris_ma": "RIS + MA", "price_free": "Miễn phí, budget 1", "price_tenfold": "Phí 10×, budget 1",
          "budget3_default": "Phí mặc định, budget 3", "budget3_free": "Miễn phí, budget 3",
          "budget3_tenfold": "Phí 10×, budget 3", "request_price": "Tính phí theo request"}
ORDER = list(LABELS)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=ROOT / "outputs_marl/paper_v2/campaign")
    args = parser.parse_args()
    report = json.loads((args.campaign / "report/results_summary.json").read_text(encoding="utf-8"))
    rows, adaptation = [], []
    for path in sorted((args.campaign / "evaluation").glob("*/seed_*/results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("complete"):
            raise ValueError(f"Incomplete evaluation: {path}")
        condition, seed = data["condition"], data["training_seed"]
        normal = data["treatments"]["normal"]["episodes"]
        assert [e["scenario_seed"] for e in normal] == list(range(40000, 40050))
        for name, treatment in data["treatments"].items():
            episodes = treatment["episodes"]
            assert len({e["scenario_seed"] for e in episodes}) == len(episodes)
            row = {"condition": condition, "treatment": name, "training_seed": seed,
                   "evaluation_scenarios": len(episodes), "scenario_seed_min": min(e["scenario_seed"] for e in episodes),
                   "scenario_seed_max": max(e["scenario_seed"] for e in episodes)}
            for metric in METRICS:
                values = [float(e[metric]) for e in episodes]
                assert all(math.isfinite(value) for value in values)
                row[metric + "_mean"] = mean(values)
                row[metric + "_sd_across_scenarios"] = stdev(values)
            rows.append(row)
        if condition == "full_nocomm":
            frozen = data["treatments"]["frozen25"]["episodes"]
            assert all(a["scenario_seed"] == b["scenario_seed"] and a["prefix25_sha256"] == b["prefix25_sha256"]
                       for a, b in zip(normal, frozen))
            adaptation.append({"training_seed": seed, "evaluation_scenarios": len(normal),
                "adaptive_post25_min_rate": mean(e["post25_min_user_rate"] for e in normal),
                "frozen_post25_min_rate": mean(e["post25_min_user_rate"] for e in frozen),
                "paired_post25_min_rate_difference": mean(a["post25_min_user_rate"] - b["post25_min_user_rate"] for a, b in zip(normal, frozen)),
                "paired_episode_return_difference": mean(a["net_return"] - b["net_return"] for a, b in zip(normal, frozen))})
    primary = sorted([r for r in rows if r["treatment"] == "normal"], key=lambda r: (ORDER.index(r["condition"]), r["training_seed"]))
    assert len(primary) == 85
    groups = {condition: [r for r in primary if r["condition"] == condition] for condition in ORDER}
    for condition, entries in groups.items():
        assert [r["training_seed"] for r in entries] == list(range(5))
        for metric in METRICS:
            average = mean(r[metric + "_mean"] for r in entries)
            aggregate = report["estimates"][condition + "/normal"]
            if metric in aggregate:
                assert math.isclose(average, aggregate[metric]["mean"], rel_tol=1e-12, abs_tol=1e-12)
    out = args.campaign / "report"
    write_csv(out / "primary_per_seed.csv", primary)
    write_csv(out / "all_treatments_per_seed.csv", rows)
    write_csv(out / "adaptation_per_seed.csv", sorted(adaptation, key=lambda r: r["training_seed"]))
    markdown = ["# Kết quả held-out từng seed huấn luyện", "",
        "Mỗi ô là trung bình trên cùng 50 scenario held-out (40000–40049) của một policy được train độc lập. Các cột seed 0–4 là seed huấn luyện, không phải seed đánh giá.", "",
        "Cột Mean ± SD mô tả biến động giữa 5 policy huấn luyện. Trong CSV, `*_sd_across_scenarios` mô tả biến động giữa scenario của một policy; hai loại SD này khác nhau. Các seed cùng số ở hai thuật toán không được coi là một cặp policy cho kiểm định.", ""]
    split = audit_seed_split(args.campaign, list(range(40000, 40050)))
    if not split["evaluation_disjoint_from_training"] or not split["development_disjoint_from_training"]:
        markdown[0] = "# Kết quả từng seed huấn luyện — bộ đánh giá gốc"
        markdown[2:2] = ["> " + SEED_WARNING, ""]
    for metric, title, precision in [("net_return", "Return theo từng seed", 2), ("min_user_rate", "Min-user rate (bps/Hz) theo từng seed", 3)]:
        markdown.extend([f"## {title}", "", "| Điều kiện | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± SD giữa seed |", "|---|---:|---:|---:|---:|---:|---:|"])
        for condition, entries in groups.items():
            values = [r[metric + "_mean"] for r in entries]
            cells = [f"{value:.{precision}f}" for value in values]
            markdown.append("| " + LABELS[condition] + " | " + " | ".join(cells) + f" | {mean(values):.{precision}f} ± {stdev(values):.{precision}f} |")
        markdown.append("")
    markdown.extend(["## Adaptive–frozen25: đối chứng trong cùng policy", "",
                     "Đây là so sánh ghép cặp hợp lệ: cùng policy, cùng scenario và cùng 25 bước đầu. Min-rate chỉ tính đoạn sau bước 25.", "",
                     "| Seed | Adaptive min-rate | Frozen min-rate | Chênh lệch min-rate | Chênh lệch return cả episode |",
                     "|---|---:|---:|---:|---:|"])
    for row in sorted(adaptation, key=lambda r: r["training_seed"]):
        markdown.append(f"| {row['training_seed']} | {row['adaptive_post25_min_rate']:.3f} | {row['frozen_post25_min_rate']:.3f} | {row['paired_post25_min_rate_difference']:+.3f} | {row['paired_episode_return_difference']:+.3f} |")
    markdown.extend(["", "## Dữ liệu chi tiết", "",
        "- [primary_per_seed.csv](primary_per_seed.csv): 85 policy, mọi metric chính, trung bình và SD giữa scenario.",
        "- [all_treatments_per_seed.csv](all_treatments_per_seed.csv): mọi treatment, gồm freeze, can thiệp communication, CSI, delay và dynamics; các grid exploratory dùng 20 scenario.",
        "- [adaptation_per_seed.csv](adaptation_per_seed.csv): hiệu ứng adaptive–frozen25 từng policy.",
        "- File gốc: `../evaluation/<condition>/seed_<seed>/results.json`; trace từng bước trong CSV cùng thư mục.",
        "", "Return ở các điều kiện price dùng mức phí riêng. Khi so sánh ở cùng mức phí, dùng `return_at_default_price_mean`; physical metrics không chịu phép trừ phí này."])
    (out / "KET_QUA_TUNG_SEED.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Validated 85 primary policies, {len(rows)} condition/treatment/seed summaries and {len(adaptation)} paired adaptation summaries.")
    for condition in ORDER[:5]:
        print(condition, "return", [round(r["net_return_mean"], 2) for r in groups[condition]])
    print("Written:", out / "KET_QUA_TUNG_SEED.md")


if __name__ == "__main__":
    main()

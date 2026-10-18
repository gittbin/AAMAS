"""Evidence-backed tables and policy-level inference; never fills missing data."""
import argparse
from itertools import combinations, product
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts_marl.evaluate_paper_v2 import PROTOCOL, write_json
from scripts_marl.train_paper_condition import CONDITIONS
from scripts_marl.paper_seed_audit import audit_seed_split, SEED_WARNING


def nested_interval(left, right=None, paired=False, repetitions=10000, seed=7631):
    left = np.asarray(left, dtype=float)
    if left.ndim != 2 or not np.isfinite(left).all():
        raise ValueError("Bootstrap requires finite policy-by-scenario matrices")
    rng = np.random.default_rng(seed)
    scenarios = rng.integers(left.shape[1], size=(repetitions, left.shape[1]))
    lpol = rng.integers(left.shape[0], size=(repetitions, left.shape[0]))
    draws = left[lpol[:, :, None], scenarios[:, None, :]].mean(axis=(1, 2))
    point = float(left.mean())
    if right is not None:
        right = np.asarray(right, dtype=float)
        if right.shape[1] != left.shape[1]:
            raise ValueError("Both conditions must use identical scenario IDs")
        if paired and right.shape != left.shape:
            raise ValueError("Paired interventions require the same policies")
        rpol = lpol if paired else rng.integers(right.shape[0], size=(repetitions, right.shape[0]))
        draws -= right[rpol[:, :, None], scenarios[:, None, :]].mean(axis=(1, 2))
        point -= float(right.mean())
    interval = np.percentile(draws, [2.5, 97.5])
    return {"mean": point, "ci95": [float(x) for x in interval], "n_training_policies": len(left), "n_scenarios": left.shape[1]}


def exact_test(left, right, paired=False):
    a, b = np.asarray(left).mean(axis=1), np.asarray(right).mean(axis=1)
    observed = abs(float(a.mean() - b.mean()))
    if paired:
        differences = a - b
        permuted = [abs(float(np.mean(differences * signs))) for signs in product((-1, 1), repeat=len(a))]
    else:
        all_values = np.concatenate([a, b])
        permuted = []
        for selected in combinations(range(len(all_values)), len(a)):
            mask = np.zeros(len(all_values), dtype=bool)
            mask[list(selected)] = True
            permuted.append(abs(float(all_values[mask].mean() - all_values[~mask].mean())))
    return float(np.mean(np.asarray(permuted) >= observed - 1e-12))


def holm(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = previous
    return adjusted.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="outputs_marl/paper_v2/campaign")
    parser.add_argument("--allow-partial", action="store_true", help="Preview report; never marks primary study complete")
    args = parser.parse_args()
    campaign = Path(args.campaign)
    datasets = {}
    for path in sorted((campaign / "evaluation").glob("*/seed_*/results.json")):
        data = json.loads(path.read_text())
        if not data.get("complete"):
            continue
        for treatment, entry in data["treatments"].items():
            datasets.setdefault((data["condition"], treatment), []).append((data["training_seed"], entry["episodes"]))
    missing = [condition for condition in CONDITIONS if len(datasets.get((condition, "normal"), [])) != 5]
    baseline_path = campaign / "baselines/results.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {"complete": False, "references": {}}
    if (missing or not baseline.get("complete")) and not args.allow_partial:
        raise RuntimeError(f"Final reporting requires five evaluated policies per condition and the full baseline set; missing {missing}")
    scenarios_expected = list(range(PROTOCOL["heldout_seed_start"], PROTOCOL["heldout_seed_start"] + 50))
    def matrix(condition, treatment, metric):
        entries = sorted(datasets[(condition, treatment)])
        episodes = [sorted(values, key=lambda x: x["scenario_seed"]) for _, values in entries]
        ids = [[episode["scenario_seed"] for episode in values] for values in episodes]
        if any(seeds != ids[0] for seeds in ids):
            raise ValueError("Mismatched scenario IDs within a condition")
        if not args.allow_partial and treatment in ("normal", "frozen25", "static_learned") and ids[0] != scenarios_expected:
            raise ValueError("Primary table requires all 50 held-out scenarios")
        return np.asarray([[episode[metric] for episode in values] for values in episodes])
    metrics = ["net_return", "return_at_default_price", "return_before_message_cost", "min_user_rate", "sum_rate", "illumination_db",
               "outage_fraction", "requested", "eligible", "delivered", "reconfiguration", "post25_min_user_rate", "post25_return"]
    report = {"complete_primary_evidence": not missing and baseline.get("complete", False), "missing_conditions": missing,
              "aggregation": "independent trained policies and common heldout scenario resampling; 10000 nested bootstrap draws",
              "estimates": {}, "primary_comparisons": []}
    report["seed_split_audit"] = audit_seed_split(campaign, scenarios_expected)
    report["strict_final_test_valid"] = (report["seed_split_audit"]["evaluation_disjoint_from_training"]
                                         and report["seed_split_audit"]["development_disjoint_from_training"])
    if not report["strict_final_test_valid"]:
        report["interpretation_warning"] = SEED_WARNING
    for condition, treatment in datasets:
        report["estimates"][f"{condition}/{treatment}"] = {metric: nested_interval(matrix(condition, treatment, metric)) for metric in metrics}
    for name, entry in baseline["references"].items():
        report["estimates"][f"baseline/{name}"] = {metric: nested_interval([[episode[metric] for episode in entry["episodes"]]]) for metric in metrics}
    hypotheses = [("full_nocomm", "normal", "full_nocomm", "frozen25", "post25_min_user_rate", True),
                  *[("full_nocomm", "normal", other, "normal", "min_user_rate", False)
                    for other in ("ma", "power_ris", "power_ma", "ris_ma")],
                  ("full_comm", "normal", "full_nocomm", "normal", "net_return", False)]
    for lc, lt, rc, rt, metric, paired in hypotheses:
        if (lc, lt) not in datasets or (rc, rt) not in datasets:
            continue
        left, right = matrix(lc, lt, metric), matrix(rc, rt, metric)
        comparison = {"left": f"{lc}/{lt}", "right": f"{rc}/{rt}", "metric": metric,
                      "paired_policy_intervention": paired, **nested_interval(left, right, paired),
                      "exact_permutation_p": exact_test(left, right, paired)}
        report["primary_comparisons"].append(comparison)
    if len(report["primary_comparisons"]) == 6:
        adjusted = holm([entry["exact_permutation_p"] for entry in report["primary_comparisons"]])
        for entry, value in zip(report["primary_comparisons"], adjusted):
            entry["holm_adjusted_p"] = value
    out = campaign / "report"
    out.mkdir(exist_ok=True)
    write_json(out / "results_summary.json", report)
    columns = ["net_return", "min_user_rate", "illumination_db", "outage_fraction", "requested", "delivered"]
    markdown = ["# Executed paper-v2 evidence", "", "Status: " + ("primary evidence complete" if report["complete_primary_evidence"] else "partial; paper conclusions remain pending"),
                "", "Cells: mean [95% interval]. Each learned condition uses independent training replicates; scenarios are not counted as training seeds.", "",
                "| Condition/treatment | Return | Min-rate | Illumination dB | Outage | Requests | Delivered |",
                "|---|---:|---:|---:|---:|---:|---:|"]
    if not report["strict_final_test_valid"]:
        markdown[2:2] = ["> " + SEED_WARNING, ""]
    tex = [r"\begin{table*}[t]\centering\scriptsize", r"\caption{Original evaluation results: mean [95\% nested bootstrap interval]. Five trained policies and 50 common scenarios per primary condition. Scenario 40000 overlaps training replicates 0 and 4; this collection is not a fully unseen final test. Static references bootstrap scenarios only.}",
           r"\begin{tabular}{lrrrrrr}\toprule", r"Condition/treatment &Return &Min-rate &Illumination &Outage &Requests &Delivered\\\midrule"]
    for name, estimates in report["estimates"].items():
        values = [f"{estimates[key]['mean']:.3f} [{estimates[key]['ci95'][0]:.3f}, {estimates[key]['ci95'][1]:.3f}]" for key in columns]
        markdown.append("| " + name + " | " + " | ".join(values) + " |")
        tex.append(name.replace("_", r"\_") + " & " + " & ".join(values) + r"\\")
    tex.extend([r"\bottomrule\end{tabular}\end{table*}"])
    markdown.extend(["", "## Prespecified comparisons", "", "Intervals and adjusted tests are separate summaries. Nonsignificance is not equivalence. Five paired policies yield only 32 sign-flip assignments.", ""])
    for entry in report["primary_comparisons"]:
        markdown.append(f"- {entry['left']} minus {entry['right']} ({entry['metric']}): {entry['mean']:.4f}, CI {entry['ci95']}, exact p={entry['exact_permutation_p']:.4f}, Holm p={entry.get('holm_adjusted_p', 'pending')}.")
    markdown.extend(["", "Remaining manuscript work: select essential tables/figures, interpret measured effects, complete abstract/conclusion, validate topology study if included, and check the official eight-page content limit. Decision-kernel timing is not deployed end-to-end latency."])
    (out / "results.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (out / "tables_full.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Report written: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

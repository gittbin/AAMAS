"""Task 1 (HV part) — complete, defensible HV statistics from the existing pickle.

Reprocesses ``results/experiment.pkl`` (25 seeds, full config) WITHOUT re-running
any optimiser.  The paper's headline robust-HV metric is the cumulative-best
robust hypervolume on the fixed per-seed eval set, tracked every generation; the
final value per seed is ``hv_gen[method][:, -1]``.

For every method it reports mean/std HV; for RAMP vs each baseline it reports the
paired difference, a paired 95% CI (Student-t AND percentile bootstrap), a
Wilcoxon signed-rank test, a Holm-corrected p-value across the four comparisons,
and the matched-pairs rank-biserial effect size.  It also builds convergence
bands (median + IQR + mean±CI) over generations for the plots.

Outputs:
  resubmission/out/task1_hv_stats.json    machine-readable
  resubmission/out/task1_hv_stats.txt     human-readable + filled placeholders
  resubmission/out/task1_convergence.npz  per-gen bands for the figure
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

PKL = "results/experiment.pkl"
OUT = "resubmission/out"

# data-key -> (short manuscript tag, display label)
LABELS = {
    "adaptive": ("RAMP", "RAMP (proposed, multitask adaptive-RMP)"),
    "no_transfer": ("NT", "No-transfer independent NSGA-II"),
    "mo_de": ("DE", "MO-DE (GDE3-style)"),
    "pooled": ("POOL", "Pooled single-task NSGA-II"),
    "mopso": ("PSO", "MOPSO (Coello-style)"),
    "fixed_rmp": ("FIX", "Fixed-RMP MO-MFEA"),
}
# comparisons for Holm correction (RAMP vs each baseline), manuscript order
COMPARISONS = ["no_transfer", "mo_de", "pooled", "mopso"]


def _load():
    import pickle
    with open(PKL, "rb") as f:
        return pickle.load(f)


def _boot_ci(diff, n_boot=20000, seed=12345):
    """Percentile bootstrap 95% CI of the paired mean difference."""
    rng = np.random.default_rng(seed)
    n = diff.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _rank_biserial_signedrank(diff):
    """Matched-pairs rank-biserial correlation (Kerby 2014) for Wilcoxon signed-rank.

    r = (sum positive ranks - sum negative ranks) / sum of all ranks.
    Zero-differences are dropped (Wilcoxon convention).  r in [-1, 1];
    positive => RAMP tends to be larger.
    """
    d = diff[diff != 0.0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    total = ranks.sum()
    return float((r_plus - r_minus) / total)


def _holm(pvals):
    """Holm-Bonferroni step-down corrected p-values (same order as input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    corrected = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)          # enforce monotonicity
        corrected[idx] = min(1.0, running)
    return corrected


def main():
    os.makedirs(OUT, exist_ok=True)
    d = _load()
    seeds = list(d.seeds)
    n_seeds = len(seeds)

    hv_final = {k: np.asarray(v)[:, -1] for k, v in d.hv_gen.items()}
    base = hv_final["adaptive"]

    # ---------- per-method mean/std ----------
    per_method = {}
    for k in ["adaptive"] + COMPARISONS + (["fixed_rmp"] if "fixed_rmp" in hv_final else []):
        col = hv_final[k]
        per_method[k] = {
            "tag": LABELS[k][0],
            "label": LABELS[k][1],
            "mean": float(col.mean()),
            "std": float(col.std(ddof=1)),
            "median": float(np.median(col)),
            "iqr": [float(np.percentile(col, 25)), float(np.percentile(col, 75))],
            "min": float(col.min()),
            "max": float(col.max()),
            "n": int(col.size),
        }

    # ---------- paired comparisons RAMP vs each baseline ----------
    raw_p, comps = [], {}
    for k in COMPARISONS:
        other = hv_final[k]
        n = min(base.size, other.size)
        diff = base[:n] - other[:n]                      # RAMP - baseline
        mean_d = float(diff.mean())
        sd = float(diff.std(ddof=1))
        se = sd / math.sqrt(n)
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        t_lo, t_hi = mean_d - tcrit * se, mean_d + tcrit * se
        b_lo, b_hi = _boot_ci(diff)
        # Wilcoxon signed-rank (two-sided). Fall back cleanly if all-zero.
        try:
            w_stat, p_w = stats.wilcoxon(diff, zero_method="wilcox",
                                         alternative="two-sided", mode="auto")
        except ValueError:
            w_stat, p_w = float("nan"), 1.0
        rrb = _rank_biserial_signedrank(diff)
        raw_p.append(float(p_w))
        comps[k] = {
            "tag": LABELS[k][0], "label": LABELS[k][1], "n_pairs": int(n),
            "delta_mean": mean_d, "delta_std": sd,
            "ci95_t": [float(t_lo), float(t_hi)],
            "ci95_boot": [b_lo, b_hi],
            "wilcoxon_stat": float(w_stat), "p_raw": float(p_w),
            "rank_biserial": rrb,
            "n_wins_RAMP": int((diff > 0).sum()), "n_ties": int((diff == 0).sum()),
            "n_losses_RAMP": int((diff < 0).sum()),
        }
    holm = _holm(raw_p)
    for k, pc in zip(COMPARISONS, holm):
        comps[k]["p_holm"] = float(pc)
        comps[k]["significant_holm_0.05"] = bool(pc < 0.05)

    # ---------- convergence bands over generations ----------
    G = np.asarray(d.hv_gen["adaptive"]).shape[1]
    bands = {}
    for k in ["adaptive"] + COMPARISONS:
        arr = np.asarray(d.hv_gen[k])                    # (n_seeds, G)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)
        se = sd / math.sqrt(arr.shape[0])
        tcrit = float(stats.t.ppf(0.975, df=arr.shape[0] - 1))
        bands[k] = {
            "mean": mean, "ci_lo": mean - tcrit * se, "ci_hi": mean + tcrit * se,
            "median": np.median(arr, axis=0),
            "q25": np.percentile(arr, 25, axis=0), "q75": np.percentile(arr, 75, axis=0),
            "q10": np.percentile(arr, 10, axis=0), "q90": np.percentile(arr, 90, axis=0),
        }
    np.savez(os.path.join(OUT, "task1_convergence.npz"),
             **{f"{k}__{stat}": v for k, b in bands.items() for stat, v in b.items()})

    # ---------- filled placeholders ----------
    rr = per_method["adaptive"]
    hv_band = (f"RAMP final robust HV: mean {rr['mean']:.1f} +/- {rr['std']:.1f} "
               f"(median {rr['median']:.1f}, IQR [{rr['iqr'][0]:.1f}, {rr['iqr'][1]:.1f}]) "
               f"over {n_seeds} seeds")
    all_sig = all(comps[k]["significant_holm_0.05"] and comps[k]["delta_mean"] > 0
                  for k in COMPARISONS)
    stat_conclusion = (
        f"On the unified robust front, RAMP's final HV ({rr['mean']:.1f} +/- {rr['std']:.1f}) "
        + ("significantly exceeds" if all_sig else "differs from")
        + " every baseline: "
        + "; ".join(
            f"vs {comps[k]['tag']} Delta={comps[k]['delta_mean']:+.2f} "
            f"(Wilcoxon p_Holm={comps[k]['p_holm']:.3g}, r_rb={comps[k]['rank_biserial']:+.2f})"
            for k in COMPARISONS)
        + f", paired over {n_seeds} seeds.")
    transfer_statement = (
        f"Transfer helps: RAMP vs No-transfer Delta_HV = {comps['no_transfer']['delta_mean']:+.2f} "
        f"(95% CI t [{comps['no_transfer']['ci95_t'][0]:+.2f}, {comps['no_transfer']['ci95_t'][1]:+.2f}], "
        f"bootstrap [{comps['no_transfer']['ci95_boot'][0]:+.2f}, {comps['no_transfer']['ci95_boot'][1]:+.2f}], "
        f"Wilcoxon p_Holm={comps['no_transfer']['p_holm']:.3g}).")

    placeholders = {
        "HV_RAMP": f"{per_method['adaptive']['mean']:.1f} +/- {per_method['adaptive']['std']:.1f}",
        "HV_NT": f"{per_method['no_transfer']['mean']:.1f} +/- {per_method['no_transfer']['std']:.1f}",
        "HV_DE": f"{per_method['mo_de']['mean']:.1f} +/- {per_method['mo_de']['std']:.1f}",
        "HV_POOL": f"{per_method['pooled']['mean']:.1f} +/- {per_method['pooled']['std']:.1f}",
        "HV_PSO": f"{per_method['mopso']['mean']:.1f} +/- {per_method['mopso']['std']:.1f}",
        "CI_NT": f"[{comps['no_transfer']['ci95_t'][0]:+.2f}, {comps['no_transfer']['ci95_t'][1]:+.2f}]",
        "CI_DE": f"[{comps['mo_de']['ci95_t'][0]:+.2f}, {comps['mo_de']['ci95_t'][1]:+.2f}]",
        "CI_POOL": f"[{comps['pooled']['ci95_t'][0]:+.2f}, {comps['pooled']['ci95_t'][1]:+.2f}]",
        "CI_PSO": f"[{comps['mopso']['ci95_t'][0]:+.2f}, {comps['mopso']['ci95_t'][1]:+.2f}]",
        "P_NT": f"{comps['no_transfer']['p_holm']:.3g}",
        "P_DE": f"{comps['mo_de']['p_holm']:.3g}",
        "P_POOL": f"{comps['pooled']['p_holm']:.3g}",
        "P_PSO": f"{comps['mopso']['p_holm']:.3g}",
        "DELTA_NT": f"{comps['no_transfer']['delta_mean']:+.2f}",
        "CI_NT_TEXT": f"[{comps['no_transfer']['ci95_t'][0]:+.2f}, {comps['no_transfer']['ci95_t'][1]:+.2f}]",
        "P_NT_TEXT": f"Wilcoxon signed-rank p={comps['no_transfer']['p_raw']:.3g} "
                     f"(Holm {comps['no_transfer']['p_holm']:.3g})",
        "EFFECT_NT": f"rank-biserial r={comps['no_transfer']['rank_biserial']:+.2f}",
        "STAT_CONCLUSION": stat_conclusion,
        "TRANSFER_STATEMENT": transfer_statement,
        "HV_BAND": hv_band,
    }

    result = {
        "source": PKL, "n_seeds": n_seeds, "seeds": seeds, "n_generations": int(G),
        "metric": "cumulative-best robust hypervolume on the fixed per-seed eval set "
                  "(RobustHVTracker), final generation",
        "per_method_hv": per_method,
        "ramp_vs_baseline": comps,
        "placeholders_task1_HV": placeholders,
        "note_IGD": "IGD placeholders (N_REF, IGD_*) require per-seed unified fronts, "
                    "which the current pickle stores only for seed 0. They are produced "
                    "by resubmission/rerun_fronts.py (per-seed fronts) -> task1b_igd.py.",
    }
    with open(os.path.join(OUT, "task1_hv_stats.json"), "w") as f:
        json.dump(result, f, indent=2)

    # human-readable
    lines = []
    lines.append(f"TASK 1 - HV statistics (reprocessed, no re-run)  [{n_seeds} seeds, G={G}]")
    lines.append("=" * 78)
    lines.append("Metric: cumulative-best robust HV on the fixed per-seed eval set "
                 "(RobustHVTracker), final gen.\n")
    lines.append(f"{'Method':<40}{'mean HV':>10}{'std':>8}{'median':>9}{'IQR':>18}")
    for k in ["adaptive"] + COMPARISONS:
        pm = per_method[k]
        lines.append(f"{pm['label']:<40}{pm['mean']:>10.2f}{pm['std']:>8.2f}"
                     f"{pm['median']:>9.2f}   [{pm['iqr'][0]:.1f}, {pm['iqr'][1]:.1f}]")
    lines.append("")
    lines.append("Paired comparisons  RAMP - baseline  (positive => RAMP larger):")
    lines.append("-" * 78)
    hdr = f"{'vs':<8}{'Delta':>9}{'t-CI95':>20}{'boot-CI95':>20}{'p_raw':>9}{'p_Holm':>9}{'r_rb':>7}{'W/T/L':>10}"
    lines.append(hdr)
    for k in COMPARISONS:
        c = comps[k]
        lines.append(
            f"{c['tag']:<8}{c['delta_mean']:>9.2f}"
            f"   [{c['ci95_t'][0]:+.2f},{c['ci95_t'][1]:+.2f}]"
            f"   [{c['ci95_boot'][0]:+.2f},{c['ci95_boot'][1]:+.2f}]"
            f"{c['p_raw']:>9.2g}{c['p_holm']:>9.2g}{c['rank_biserial']:>7.2f}"
            f"  {c['n_wins_RAMP']}/{c['n_ties']}/{c['n_losses_RAMP']}")
    lines.append("")
    lines.append("FILLED PLACEHOLDERS (Task 1 - HV):")
    lines.append("-" * 78)
    for key, val in placeholders.items():
        lines.append(f"{key} = {val}")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "task1_hv_stats.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()

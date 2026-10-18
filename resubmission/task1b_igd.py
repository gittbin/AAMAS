"""Task 1b - per-seed IGD + common reference front, from resubmission/out/rerun_fronts.pkl.

For EACH seed a common reference front is built as the non-dominated union of all
methods' unified robust fronts AT THAT SEED (so every method is scored against the
same paired reference under identical environments - consistent with the paper's
paired-per-seed HV methodology). Each method's per-seed front is scored by
normalised IGD (normalised by that seed's reference min/range, identical for all
methods at the seed). Reports mean +/- std IGD, N_REF (mean +/- range over seeds),
and a global common reference too (non-dominated union over all method x seed
fronts) as a robustness cross-check. Paired Wilcoxon (RAMP - baseline, Holm).
"""
from __future__ import annotations

import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from resubmission.harness import build_reference_front, igd_normalized

OUT = "resubmission/out"
CKPT = os.path.join(OUT, "rerun_fronts.pkl")
LAB = {"adaptive": "RAMP", "no_transfer": "NT", "mo_de": "DE", "pooled": "POOL", "mopso": "PSO"}
ORDER = ["adaptive", "no_transfer", "mo_de", "pooled", "mopso"]
COMPARISONS = ["no_transfer", "mo_de", "pooled", "mopso"]


def _holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    out = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * pvals[idx])
        out[idx] = min(1.0, run)
    return out


def main():
    with open(CKPT, "rb") as f:
        state = pickle.load(f)
    res = state["results"]
    seeds = state["meta"]["seeds"]

    # gather fronts[method][seed]
    fronts = {m: {} for m in ORDER}
    for key, r in res.items():
        m, sd = key.split("|")
        sd = int(sd)
        if m in fronts:
            fronts[m][sd] = np.asarray(r["front"], dtype=np.float64)

    # per-seed common reference front (union of all methods AT that seed)
    all_seeds = sorted({sd for m in ORDER for sd in fronts[m]})
    seed_ref_sizes = []
    igd = {m: [] for m in ORDER}
    igd_by_seed = {m: {} for m in ORDER}
    for sd in all_seeds:
        seed_fronts = [fronts[m][sd] for m in ORDER if sd in fronts[m]]
        if not seed_fronts:
            continue
        Rs = build_reference_front(seed_fronts)
        seed_ref_sizes.append(int(Rs.shape[0]))
        mn = Rs.min(axis=0); rng = Rs.max(axis=0) - mn; rng[rng < 1e-12] = 1.0
        for m in ORDER:
            if sd in fronts[m]:
                v = igd_normalized(fronts[m][sd], Rs, mn, rng)
                igd[m].append(v)
                igd_by_seed[m][sd] = v
    seed_ref_sizes = np.asarray(seed_ref_sizes, float)
    n_ref = float(seed_ref_sizes.mean()) if seed_ref_sizes.size else 0.0
    n_ref_range = ([int(seed_ref_sizes.min()), int(seed_ref_sizes.max())]
                   if seed_ref_sizes.size else [0, 0])
    # global reference (cross-check only)
    Rg = build_reference_front([fronts[m][sd] for m in ORDER for sd in fronts[m]])
    n_ref_global = int(Rg.shape[0])

    stats_igd = {}
    for m in ORDER:
        a = np.asarray(igd[m], float)
        stats_igd[m] = {"tag": LAB[m], "mean": float(a.mean()), "std": float(a.std(ddof=1)),
                        "median": float(np.median(a)), "n": int(a.size)}

    # paired Wilcoxon RAMP vs each baseline on IGD (lower is better -> baseline-RAMP)
    base_seeds = sorted(fronts["adaptive"])
    ramp = np.array([igd_by_seed["adaptive"][s] for s in base_seeds])
    comps, raw_p = {}, []
    for m in COMPARISONS:
        common = [s for s in base_seeds if s in igd_by_seed[m]]
        a = np.array([igd_by_seed["adaptive"][s] for s in common])
        b = np.array([igd_by_seed[m][s] for s in common])
        diff = b - a                                   # positive => RAMP better (lower IGD)
        try:
            w, pw = stats.wilcoxon(diff, alternative="two-sided")
        except ValueError:
            w, pw = float("nan"), 1.0
        raw_p.append(float(pw))
        comps[m] = {"tag": LAB[m], "delta_igd_mean": float(diff.mean()),
                    "ramp_better_count": int((diff > 0).sum()), "n": int(diff.size),
                    "p_raw": float(pw)}
    for m, pc in zip(COMPARISONS, _holm(raw_p)):
        comps[m]["p_holm"] = float(pc)

    placeholders = {
        "N_REF": f"{n_ref:.0f} (per-seed common reference, range {n_ref_range[0]}-{n_ref_range[1]}; "
                 f"global union = {n_ref_global})",
        "IGD_RAMP": f"{stats_igd['adaptive']['mean']:.3f} +/- {stats_igd['adaptive']['std']:.3f}",
        "IGD_NT": f"{stats_igd['no_transfer']['mean']:.3f} +/- {stats_igd['no_transfer']['std']:.3f}",
        "IGD_DE": f"{stats_igd['mo_de']['mean']:.3f} +/- {stats_igd['mo_de']['std']:.3f}",
        "IGD_POOL": f"{stats_igd['pooled']['mean']:.3f} +/- {stats_igd['pooled']['std']:.3f}",
        "IGD_PSO": f"{stats_igd['mopso']['mean']:.3f} +/- {stats_igd['mopso']['std']:.3f}",
    }
    result = {"n_ref_mean": n_ref, "n_ref_range": n_ref_range, "n_ref_global": n_ref_global,
              "seeds": base_seeds, "igd_stats": stats_igd,
              "ramp_vs_baseline_igd": comps, "placeholders_task1b": placeholders,
              "reference_note": "per-seed common reference = non-dominated union of all methods' "
                                "fronts AT that seed; IGD normalised by that seed's reference "
                                "min/range (identical for all methods at the seed)."}
    with open(os.path.join(OUT, "task1b_igd.json"), "w") as f:
        json.dump(result, f, indent=2)

    lines = [f"TASK 1b - IGD (per-seed common reference, N_REF~{n_ref:.0f} "
             f"[{n_ref_range[0]}-{n_ref_range[1]}], global={n_ref_global}, {len(base_seeds)} seeds)",
             "=" * 74, f"{'Method':<8}{'IGD mean':>12}{'std':>10}{'median':>10}"]
    for m in ORDER:
        s = stats_igd[m]
        lines.append(f"{s['tag']:<8}{s['mean']:>12.4f}{s['std']:>10.4f}{s['median']:>10.4f}")
    lines.append("\nPaired IGD (RAMP vs baseline; positive delta => RAMP lower/better):")
    for m in COMPARISONS:
        c = comps[m]
        lines.append(f"  vs {c['tag']:<6} delta={c['delta_igd_mean']:+.4f} "
                     f"RAMP-better {c['ramp_better_count']}/{c['n']}  "
                     f"p_raw={c['p_raw']:.3g} p_Holm={c['p_holm']:.3g}")
    lines.append("\nFILLED PLACEHOLDERS (Task 1b):")
    for k, v in placeholders.items():
        lines.append(f"{k} = {v}")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "task1b_igd.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()

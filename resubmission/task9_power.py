"""Task 9 - equal vs fairness-oriented (inverse-channel-gain) power allocation.

Equal policy (rho_k = P_max/K) results are REUSED from rerun_fronts.pkl. The
fairness policy runs RAMP, No-transfer and Pooled under power_policy='inv_gain'
(15 seeds) so we can check both the HV shift AND whether the method ranking is
stable across policies.

inv_gain policy:  rho_k proportional to 1/|h_k^H w_k|^2, renormalised to
sum_k rho_k = P_max (weak users get more power -> helps worst-user fairness).

Checkpointed to resubmission/out/task9_power_runs.pkl.
Usage: python resubmission/task9_power.py --seeds 15 --device cuda
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from momgrap.config import default_config
from momgrap.logging_utils import setup_logging
from resubmission.harness import build_reference_front, igd_normalized, run_instrumented

OUT = "resubmission/out"
RUNS = os.path.join(OUT, "task9_power_runs.pkl")
METHODS = ["adaptive", "no_transfer", "pooled"]
LAB = {"adaptive": "RAMP", "no_transfer": "No-transfer", "pooled": "Pooled"}


def load_runs():
    if os.path.exists(RUNS):
        with open(RUNS, "rb") as f:
            return pickle.load(f)
    return {}


def save_runs(d):
    tmp = RUNS + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(d, f)
    os.replace(tmp, RUNS)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=15)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    log = setup_logging(log_file=os.path.join(OUT, "task9_power.log"))
    cfg_fair = dataclasses.replace(default_config(), device_str=args.device,
                                   power_policy="inv_gain", log_every=0)
    seeds = list(range(args.seeds))

    runs = load_runs()
    for m in METHODS:
        for sd in seeds:
            key = f"{m}|{sd}"
            if key in runs:
                continue
            r = run_instrumented(m, cfg_fair, sd)
            runs[key] = r
            save_runs(runs)
            log.info(f"inv_gain {m} s{sd}: HV~{r['final_hv_curve']:.1f} |front|={r['n_front']} t={r['t_total']:.1f}s")
    analyse(runs, seeds, log)


def _collect(results_dict, keyfmt, seeds):
    hv, fronts = [], {}
    for sd in seeds:
        r = results_dict.get(keyfmt(sd))
        if r:
            hv.append(r["final_hv_curve"])
            fronts[sd] = np.asarray(r["front"], float)
    return np.asarray(hv, float), fronts


def analyse(runs, seeds, log):
    with open(os.path.join(OUT, "rerun_fronts.pkl"), "rb") as f:
        main_state = pickle.load(f)
    equal_res = main_state["results"]

    policies = {"equal": (equal_res, lambda m, sd: f"{m}|{sd}"),
                "inv_gain": (runs, lambda m, sd: f"{m}|{sd}")}
    data = {}
    for pol, (src, kf) in policies.items():
        data[pol] = {}
        for m in METHODS:
            hv, fronts = _collect(src, lambda sd: kf(m, sd), seeds)
            data[pol][m] = {"hv": hv, "fronts": fronts}

    # per-policy IGD: per-seed common reference (union of methods AT each seed),
    # consistent with Task 1b (keeps IGD in a comparable [0,~1] range).
    out_rows = {}
    for pol in policies:
        seeds_here = sorted({s for m in METHODS for s in data[pol][m]["fronts"]})
        igd_vals = {m: [] for m in METHODS}
        ref_sizes = []
        for s in seeds_here:
            sfronts = [data[pol][m]["fronts"][s] for m in METHODS if s in data[pol][m]["fronts"]]
            if not sfronts:
                continue
            Rs = build_reference_front(sfronts)
            ref_sizes.append(int(Rs.shape[0]))
            mn = Rs.min(axis=0); rng = Rs.max(axis=0) - mn; rng[rng < 1e-12] = 1.0
            for m in METHODS:
                if s in data[pol][m]["fronts"]:
                    igd_vals[m].append(igd_normalized(data[pol][m]["fronts"][s], Rs, mn, rng))
        rows = {}
        for m in METHODS:
            d = data[pol][m]
            igd = np.asarray(igd_vals[m], float)
            rows[m] = {"hv_mean": float(d["hv"].mean()), "hv_std": float(d["hv"].std(ddof=1) if d["hv"].size > 1 else 0),
                       "igd_mean": float(igd.mean()) if igd.size else float("nan"),
                       "igd_std": float(igd.std(ddof=1)) if igd.size > 1 else 0.0,
                       "n": int(d["hv"].size)}
        rank = sorted(METHODS, key=lambda mm: -rows[mm]["hv_mean"])
        out_rows[pol] = {"rows": rows, "ranking_by_hv": [LAB[m] for m in rank],
                         "reference_N": int(np.mean(ref_sizes)) if ref_sizes else 0}

    rank_equal = out_rows["equal"]["ranking_by_hv"]
    rank_fair = out_rows["inv_gain"]["ranking_by_hv"]
    stable = rank_equal == rank_fair
    ph = {
        "POWER_POLICY": ("Fairness-oriented inverse-channel-gain allocation: "
                         "rho_k proportional to 1/|h_k^H w_k|^2, renormalised so sum_k rho_k = P_max "
                         "(weak users receive more power). Baseline: equal rho_k = P_max/K."),
        "HV_EQUAL": f"{out_rows['equal']['rows']['adaptive']['hv_mean']:.1f} +/- "
                    f"{out_rows['equal']['rows']['adaptive']['hv_std']:.1f} (RAMP, equal power)",
        "HV_POWER": f"{out_rows['inv_gain']['rows']['adaptive']['hv_mean']:.1f} +/- "
                    f"{out_rows['inv_gain']['rows']['adaptive']['hv_std']:.1f} (RAMP, inverse-gain power)",
        "RANKING_POWER": (f"By robust HV: equal -> {' > '.join(rank_equal)}; "
                          f"inverse-gain -> {' > '.join(rank_fair)}. "
                          f"Method ranking is {'STABLE' if stable else 'NOT stable'} across policies."),
    }
    result = {"seeds": seeds, "by_policy": out_rows, "ranking_stable": stable, "placeholders_task9": ph}
    with open(os.path.join(OUT, "task9_power.json"), "w") as f:
        json.dump(result, f, indent=2)

    lines = ["TASK 9 - power-allocation ablation (equal vs inverse-channel-gain)", "=" * 70]
    for pol in ("equal", "inv_gain"):
        lines.append(f"\n[{pol}]  IGD ref N={out_rows[pol]['reference_N']}")
        lines.append(f"  {'method':<14}{'HV mean+/-std':>18}{'IGD mean+/-std':>18}")
        for m in METHODS:
            r = out_rows[pol]["rows"][m]
            lines.append(f"  {LAB[m]:<14}{r['hv_mean']:>10.1f}+/-{r['hv_std']:<5.1f}"
                         f"{r['igd_mean']:>11.3f}+/-{r['igd_std']:<5.3f}")
        lines.append(f"  ranking by HV: {' > '.join(out_rows[pol]['ranking_by_hv'])}")
    lines.append(f"\nranking stable across policies: {stable}")
    lines.append("\nFILLED PLACEHOLDERS (Task 9):")
    for k, v in ph.items():
        lines.append(f"{k} = {v}")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "task9_power.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()

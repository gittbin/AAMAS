"""Tasks 6/7/8 - sensitivity to MC batch B, scalability to RIS L, MA count M.

Runs RAMP (adaptive) at each swept configuration (15 seeds), recording per-seed
unified front (for IGD), HV, runtime and peak memory. The default points
(B=32, L=64, M=16) are REUSED from resubmission/out/rerun_fronts.pkl.

Scaling conventions (documented, fixed before running):
  * B sweep: only the MC batch changes; objective landscape + HV reference are
    identical, so HV is directly comparable across B.
  * L sweep (Task 7): FIXED inter-element spacing (0.5 lambda) -> the RIS physical
    aperture grows with L. The sensing reflected-path coefficient is normalised as
    gamma_ris / L (sensing.py), keeping a coherently-steered RIS contribution O(1)
    independent of L. (ETA_L_RULE)
  * M sweep (Task 8): near-square BS grid, genotype dim D = 2M + L, min-spacing
    d_min = 0.5 lambda unchanged; beamformer/precoder dimensions follow M.

Because L and M change the achievable objective values, HV/IGD are compared
within each sweep (per-sweep reference front), not across different knobs.

Checkpointed to resubmission/out/sensitivity_runs.pkl (resumable).
Usage: python resubmission/sensitivity.py --seeds 15 --device cuda [--task B|L|M|all]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from momgrap.config import default_config
from momgrap.logging_utils import setup_logging
from resubmission.harness import build_reference_front, igd_normalized, run_instrumented

OUT = "resubmission/out"
RUNS = os.path.join(OUT, "sensitivity_runs.pkl")


def pick_grid(n):
    best = (1, n)
    for r in range(1, int(math.isqrt(n)) + 1):
        if n % r == 0:
            best = (r, n // r)
    return best


def make_cfg(base, task, val):
    if task == "B":
        return dataclasses.replace(base, mc_samples=int(val))
    if task == "L":
        r, c = pick_grid(int(val))
        return dataclasses.replace(base, L=int(val), ris_grid_rows=r, ris_grid_cols=c)
    if task == "M":
        r, c = pick_grid(int(val))
        return dataclasses.replace(base, M=int(val), bs_grid_rows=r, bs_grid_cols=c)
    raise ValueError(task)


SWEEPS = {"B": (32, [8, 16, 64]), "L": (64, [32, 128]), "M": (16, [8, 24])}


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


def run_sweeps(tasks, seeds, device):
    log = setup_logging(log_file=os.path.join(OUT, "sensitivity.log"))
    base = dataclasses.replace(default_config(), device_str=device, log_every=0)
    runs = load_runs()
    for task in tasks:
        _, vals = SWEEPS[task]
        for val in vals:
            cfg = make_cfg(base, task, val)
            cfg.validate()
            for sd in seeds:
                key = f"{task}={val}|{sd}"
                if key in runs:
                    continue
                r = run_instrumented("adaptive", cfg, sd)
                r["knob"] = task
                r["val"] = val
                runs[key] = r
                save_runs(runs)
                log.info(f"{task}={val} s{sd}: HV~{r['final_hv_curve']:.1f} |front|={r['n_front']} "
                         f"t={r['t_total']:.1f}s mem={r['peak_mem_bytes']/1e6:.0f}MB "
                         f"D={cfg.D} L={cfg.L} M={cfg.M} S={cfg.mc_samples}")
    return runs


def _default_adaptive_from_main(seeds):
    """Reuse adaptive at the default point (B=32,L=64,M=16) from rerun_fronts.pkl."""
    path = os.path.join(OUT, "rerun_fronts.pkl")
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        st = pickle.load(f)
    out = {}
    for sd in seeds:
        r = st["results"].get(f"adaptive|{sd}")
        if r:
            out[sd] = r
    return out


def analyse(runs, seeds):
    default_pts = _default_adaptive_from_main(seeds)
    results = {}
    for task, (default_val, vals) in SWEEPS.items():
        allvals = sorted(set([default_val] + vals))
        # collect fronts + HV + timing + mem per value
        per_val = {}
        for v in allvals:
            fronts, hv, tt, mem = {}, [], [], []
            if v == default_val and default_pts:
                for sd, r in default_pts.items():
                    fronts[sd] = np.asarray(r["front"], float)
                    hv.append(r["final_hv_curve"]); tt.append(r["t_total"])
                    mem.append(r["peak_mem_bytes"] / 1e6)
            else:
                for sd in seeds:
                    r = runs.get(f"{task}={v}|{sd}")
                    if r:
                        fronts[sd] = np.asarray(r["front"], float)
                        hv.append(r["final_hv_curve"]); tt.append(r["t_total"])
                        mem.append(r["peak_mem_bytes"] / 1e6)
            per_val[v] = {"fronts": fronts, "hv": np.asarray(hv, float),
                          "t": np.asarray(tt, float), "mem": np.asarray(mem, float)}
        # per-seed common reference = union across swept values AT each seed
        # (consistent with Task 1b; keeps IGD in a comparable [0,~1] range).
        seeds_here = sorted({s for v in allvals for s in per_val[v]["fronts"]})
        igd_vals = {v: [] for v in allvals}
        ref_sizes = []
        for s in seeds_here:
            sfronts = [per_val[v]["fronts"][s] for v in allvals if s in per_val[v]["fronts"]]
            if not sfronts:
                continue
            Rs = build_reference_front(sfronts)
            ref_sizes.append(int(Rs.shape[0]))
            mn = Rs.min(axis=0); rng = Rs.max(axis=0) - mn; rng[rng < 1e-12] = 1.0
            for v in allvals:
                if s in per_val[v]["fronts"]:
                    igd_vals[v].append(igd_normalized(per_val[v]["fronts"][s], Rs, mn, rng))
        rows = {}
        for v in allvals:
            fr = per_val[v]
            igd = np.asarray(igd_vals[v], float)
            rows[v] = {
                "hv_mean": float(fr["hv"].mean()), "hv_std": float(fr["hv"].std(ddof=1) if fr["hv"].size > 1 else 0),
                "igd_mean": float(igd.mean()) if igd.size else float("nan"),
                "igd_std": float(igd.std(ddof=1)) if igd.size > 1 else 0.0,
                "runtime_s_mean": float(fr["t"].mean()), "peak_mem_MB_mean": float(fr["mem"].mean()),
                "n": int(fr["hv"].size), "is_default": bool(v == default_val),
            }
        results[task] = {"reference_N": int(np.mean(ref_sizes)) if ref_sizes else 0,
                         "rows": rows, "values": allvals}
    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--task", default="all", choices=["all", "B", "L", "M"])
    args = p.parse_args(argv)
    seeds = list(range(args.seeds))
    tasks = ["B", "L", "M"] if args.task == "all" else [args.task]
    runs = run_sweeps(tasks, seeds, args.device)
    results = analyse(runs, seeds)

    # placeholders
    tasknames = {"B": ("SENS_B", "Task 6 (MC batch B)"),
                 "L": ("SENS_L", "Task 7 (RIS L)"), "M": ("SENS_M", "Task 8 (MA M)")}
    ph = {}
    mem_all = []
    for task, res in results.items():
        key, name = tasknames[task]
        parts = []
        for v in res["values"]:
            r = res["rows"][v]
            mem_all.append(r["peak_mem_MB_mean"])
            parts.append(f"{task}={v}{'*' if r['is_default'] else ''}: HV={r['hv_mean']:.1f}+/-{r['hv_std']:.1f} "
                         f"IGD={r['igd_mean']:.3f} t={r['runtime_s_mean']:.0f}s mem={r['peak_mem_MB_mean']:.0f}MB")
        ph[key] = " | ".join(parts) + "  (*=default, reused)"
    if "L" in results:
        ph["ETA_L_RULE"] = ("Fixed inter-element spacing (0.5 lambda): RIS aperture grows with L; "
                            "the sensing reflected-path coefficient is normalised as gamma_ris/L, "
                            "so a coherently-steered RIS contribution stays O(1) independent of L.")
    if mem_all:
        ph["MEM_RANGE"] = f"peak GPU memory across sensitivity/scalability configs: {min(mem_all):.0f}-{max(mem_all):.0f} MB"

    out = {"results": results, "placeholders_sensitivity": ph}
    with open(os.path.join(OUT, "sensitivity.json"), "w") as f:
        json.dump(out, f, indent=2)

    lines = ["TASKS 6/7/8 - sensitivity / scalability (RAMP, 15 seeds)", "=" * 74]
    for task, res in results.items():
        lines.append(f"\n[{tasknames[task][1]}]  IGD ref N={res['reference_N']}")
        lines.append(f"  {'val':<8}{'HV mean+/-std':>18}{'IGD mean+/-std':>18}{'runtime(s)':>12}{'peakMem(MB)':>13}")
        for v in res["values"]:
            r = res["rows"][v]
            tag = "*" if r["is_default"] else " "
            lines.append(f"  {str(v)+tag:<8}{r['hv_mean']:>10.1f}+/-{r['hv_std']:<5.1f}"
                         f"{r['igd_mean']:>11.3f}+/-{r['igd_std']:<5.3f}{r['runtime_s_mean']:>12.0f}{r['peak_mem_MB_mean']:>13.0f}")
    lines.append("\nFILLED PLACEHOLDERS (Tasks 6/7/8):")
    for k, v in ph.items():
        lines.append(f"{k} = {v}")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "sensitivity.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()

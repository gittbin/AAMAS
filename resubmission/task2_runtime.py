"""Task 2 - runtime, throughput, memory summary from resubmission/out/rerun_fronts.pkl.

Per method (mean over seeds): total wall-clock, evolutionary search time, final
archive cross-regime re-evaluation time, peak memory, total fitness evaluations,
evaluations/second, and the archive-overhead fraction.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

OUT = "resubmission/out"
CKPT = os.path.join(OUT, "rerun_fronts.pkl")
LAB = {"adaptive": "RAMP", "fixed_rmp": "FIX", "no_transfer": "NT",
       "pooled": "POOL", "mo_de": "DE", "mopso": "PSO"}
ORDER = ["adaptive", "no_transfer", "mo_de", "pooled", "mopso", "fixed_rmp"]


def _agg(vals):
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1) if a.size > 1 else 0.0),
            "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def main():
    with open(CKPT, "rb") as f:
        state = pickle.load(f)
    res = state["results"]
    dev = state["meta"].get("device", "?")
    by_method = {}
    for key, r in res.items():
        m = key.split("|")[0]
        by_method.setdefault(m, []).append(r)

    per = {}
    for m in ORDER:
        rows = by_method.get(m)
        if not rows:
            continue
        mem_kind = rows[0]["mem_kind"]
        per[m] = {
            "tag": LAB[m], "n_seeds": len(rows),
            "t_total_s": _agg([x["t_total"] for x in rows]),
            "t_search_s": _agg([x["t_search"] for x in rows]),
            "t_archive_s": _agg([x["t_archive"] for x in rows]),
            "archive_frac": _agg([x["archive_frac"] for x in rows]),
            "peak_mem_MB": _agg([x["peak_mem_bytes"] / 1e6 for x in rows]),
            "n_fitness_evals": int(rows[0]["n_fitness_evals"]),
            "evals_per_sec": _agg([x["evals_per_sec"] for x in rows]),
            "mem_kind": mem_kind,
        }

    memlabel = "peak GPU mem (MB)" if per.get("adaptive", {}).get("mem_kind") == "cuda_alloc" \
        else "peak CPU RSS (MB)"
    ph = {}
    for m in ["adaptive", "no_transfer", "mo_de", "pooled", "mopso"]:
        if m in per:
            ph[f"T_{LAB[m]}"] = f"{per[m]['t_total_s']['mean']:.1f} s"
            ph[f"MEM_{LAB[m]}"] = f"{per[m]['peak_mem_MB']['mean']:.0f} MB ({per[m]['mem_kind']})"
            ph[f"THR_{LAB[m]}"] = f"{per[m]['evals_per_sec']['mean']:.0f} evals/s"
    ph["T_RAMP"] = ph.pop("T_RAMP", None)
    if "adaptive" in per:
        ph["ARCHIVE_FRAC"] = f"{per['adaptive']['archive_frac']['mean']*100:.1f}% of total wall-clock (RAMP)"
    ph["RUNTIME_CONCLUSION"] = (
        f"On {dev}, RAMP completes a full {state['meta']['seeds'][-1]+1 if state['meta']['seeds'] else '?'}"
        f"-seed-scale run in {per['adaptive']['t_total_s']['mean']:.1f} s/seed "
        f"({per['adaptive']['evals_per_sec']['mean']:.0f} evals/s), of which "
        f"{per['adaptive']['archive_frac']['mean']*100:.1f}% is the one-off archive "
        f"cross-regime re-evaluation; runtime/throughput are within "
        f"{_ratio(per):.2f}x across all five methods." if "adaptive" in per else "n/a")

    result = {"device": dev, "memory_metric": memlabel, "per_method": per,
              "placeholders_task2": {k: v for k, v in ph.items() if v is not None}}
    with open(os.path.join(OUT, "task2_runtime.json"), "w") as f:
        json.dump(result, f, indent=2)

    lines = [f"TASK 2 - runtime / throughput / memory  [device={dev}]", "=" * 78,
             f"{'Method':<8}{'total(s)':>10}{'search(s)':>11}{'archive(s)':>12}"
             f"{'arch%':>8}{memlabel:>20}{'evals/s':>12}"]
    for m in ORDER:
        if m not in per:
            continue
        pm = per[m]
        lines.append(f"{pm['tag']:<8}{pm['t_total_s']['mean']:>10.1f}"
                     f"{pm['t_search_s']['mean']:>11.1f}{pm['t_archive_s']['mean']:>12.1f}"
                     f"{pm['archive_frac']['mean']*100:>7.1f}%{pm['peak_mem_MB']['mean']:>20.0f}"
                     f"{pm['evals_per_sec']['mean']:>12.0f}")
    lines.append("\nFILLED PLACEHOLDERS (Task 2):")
    for k, v in result["placeholders_task2"].items():
        lines.append(f"{k} = {v}")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "task2_runtime.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


def _ratio(per):
    ts = [per[m]["t_total_s"]["mean"] for m in per]
    return max(ts) / min(ts) if ts else 1.0


if __name__ == "__main__":
    main()

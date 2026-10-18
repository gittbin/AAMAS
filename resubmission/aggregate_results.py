"""Aggregate every filled placeholder from the resubmission out/*.json into one file.

Writes resubmission/out/ALL_PLACEHOLDERS.md - the manuscript placeholder -> value
map, marking anything still pending.
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "resubmission/out"

# every placeholder the reviewer document lists, grouped by task
EXPECTED = {
    "Task 1 (stats)": ["N_REF", "IGD_RAMP", "IGD_NT", "IGD_DE", "IGD_POOL", "IGD_PSO",
                        "CI_NT", "CI_DE", "CI_POOL", "CI_PSO", "P_NT", "P_DE", "P_POOL", "P_PSO",
                        "DELTA_NT", "CI_NT_TEXT", "P_NT_TEXT", "EFFECT_NT",
                        "STAT_CONCLUSION", "TRANSFER_STATEMENT", "HV_BAND",
                        "HV_RAMP", "HV_NT", "HV_DE", "HV_POOL", "HV_PSO"],
    "Task 2 (runtime)": ["T_RAMP", "T_NT", "T_DE", "T_POOL", "T_PSO",
                         "MEM_RAMP", "MEM_NT", "MEM_DE", "MEM_POOL", "MEM_PSO",
                         "THR_RAMP", "THR_NT", "THR_DE", "THR_POOL", "THR_PSO",
                         "ARCHIVE_FRAC", "RUNTIME_CONCLUSION"],
    "Task 3 (cpu/gpu)": ["SPEEDUP_GPU"],
    "Task 4 (RMP)": ["RMP_SWEEP", "RMP_FINAL", "TRANSFER_INTERPRETATION", "TRANSFER_STATEMENT"],
    "Task 5 (similarity)": ["N_SIM", "TASK_SIM", "TRANSFER_INTERPRETATION"],
    "Task 6/7/8 (sens)": ["SENS_B", "SENS_L", "SENS_M", "ETA_L_RULE", "MEM_RANGE"],
    "Task 9 (power)": ["POWER_POLICY", "HV_EQUAL", "HV_POWER", "RANKING_POWER"],
    "Task 10 (online)": ["N_ONLINE", "G_ONLINE", "HV_RATIO"],
    "Task 11 (constants)": ["SVD_TOL", "EPS_NORM"],
    "Task 12 (env)": ["PYTHON_VER", "PYTORCH_VER", "CUDA_VER", "CPU_RAM"],
}

# placeholders not produced by a *.json (documented directly in RESULTS.md / code)
STATIC = {
    "SVD_TOL": "no explicit SVD singular-value tolerance; BD null-space taken structurally "
               "as the last M-(K-1) right-singular vectors of a full SVD (comm.py:32); "
               "precoder-norm floor clamp_min(1e-12) (comm.py:39)",
    "EPS_NORM": "1e-12 (precoder-norm floor comm.py:39; dB log-floor _DB_EPS objectives.py:26)",
    "PYTHON_VER": "3.14.3",
    "PYTORCH_VER": "2.11.0+cu128 (CUDA build; a 2.12.0+cpu build was replaced to enable the GPU)",
    "CUDA_VER": "12.8 (PyTorch runtime; nvidia-smi driver-compat reports 13.1)",
    "CPU_RAM": "64 GB (Intel Core i9-14900K, 32 logical processors)",
}


def collect():
    found = {}
    for path in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        _walk(data, found, os.path.basename(path))
    found.update({k: (v, "static/code") for k, v in STATIC.items() if k not in found})
    return found


def _walk(obj, found, src):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("placeholders") and isinstance(v, dict):
                for pk, pv in v.items():
                    if pv is not None and pk not in found:
                        found[pk] = (pv, src)
            else:
                _walk(v, found, src)
    elif isinstance(obj, list):
        for it in obj:
            _walk(it, found, src)


def main():
    found = collect()
    lines = ["# All manuscript placeholders (aggregated)", ""]
    n_done = n_total = 0
    for group, keys in EXPECTED.items():
        lines.append(f"## {group}")
        for k in keys:
            n_total += 1
            if k in found:
                n_done += 1
                val, src = found[k]
                sval = str(val)
                if len(sval) > 400:
                    sval = sval[:397] + "..."
                lines.append(f"- **{k}** = {sval}  _(src: {src})_")
            else:
                lines.append(f"- **{k}** = _PENDING_")
        lines.append("")
    header = f"> {n_done}/{n_total} placeholders filled.\n"
    txt = "# All manuscript placeholders (aggregated)\n\n" + header + "\n" + "\n".join(lines[2:])
    with open(os.path.join(OUT, "ALL_PLACEHOLDERS.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"{n_done}/{n_total} placeholders filled -> {OUT}/ALL_PLACEHOLDERS.md")
    # also print the pending ones
    pend = [k for keys in EXPECTED.values() for k in keys if k not in found]
    if pend:
        print("PENDING:", ", ".join(sorted(set(pend))))


if __name__ == "__main__":
    main()

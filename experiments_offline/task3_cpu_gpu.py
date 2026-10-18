"""Task 3 - matched CPU vs GPU speedup for RAMP.

Runs the SAME RAMP configuration (identical pop N, generations G, MC samples S,
and seed) on CPU and on GPU and reports the wall-clock speedup T_CPU / T_GPU.
The two devices draw different RNG streams, but the amount of computation is
identical, so the timing comparison is matched by construction (per the reviewer's
note - no unmatched settings are compared).

Usage: python experiments_offline/task3_cpu_gpu.py --seeds 3
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from momgrap.config import default_config
from momgrap.logging_utils import setup_logging
from experiments_offline.harness import run_instrumented

OUT = "experiments_offline/out"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--gens", type=int, default=None, help="reduce G if CPU is too slow (report explicitly)")
    args = p.parse_args(argv)
    log = setup_logging(log_file=os.path.join(OUT, "task3_cpu_gpu.log"))

    if not torch.cuda.is_available():
        log.warning("CUDA not available - Task 3 cannot produce a GPU number on this machine.")
        return

    over = {"log_every": 0}
    if args.gens is not None:
        over["max_gen"] = args.gens
    cfg_cpu = dataclasses.replace(default_config(), device_str="cpu", **over)
    cfg_gpu = dataclasses.replace(default_config(), device_str="cuda", **over)
    seeds = list(range(args.seeds))
    log.info(f"Task3: matched RAMP CPU vs GPU, seeds={seeds}, "
             f"N={cfg_cpu.pop_size} G={cfg_cpu.max_gen} S={cfg_cpu.mc_samples}")

    rows = []
    for sd in seeds:
        rg = run_instrumented("adaptive", cfg_gpu, sd)
        rc = run_instrumented("adaptive", cfg_cpu, sd)
        speed_total = rc["t_total"] / rg["t_total"]
        speed_search = rc["t_search"] / rg["t_search"]
        rows.append({"seed": sd,
                     "t_cpu_total": rc["t_total"], "t_gpu_total": rg["t_total"],
                     "t_cpu_search": rc["t_search"], "t_gpu_search": rg["t_search"],
                     "speedup_total": speed_total, "speedup_search": speed_search,
                     "gpu_peak_MB": rg["peak_mem_bytes"] / 1e6})
        log.info(f"seed {sd}: CPU {rc['t_total']:.1f}s  GPU {rg['t_total']:.1f}s  "
                 f"speedup(total)={speed_total:.2f}x  speedup(search)={speed_search:.2f}x")

    st = np.array([r["speedup_total"] for r in rows])
    ss = np.array([r["speedup_search"] for r in rows])
    matched = (args.gens is None)
    ph = {
        "SPEEDUP_GPU": f"{st.mean():.2f}x (total wall-clock, mean over {len(seeds)} matched seeds; "
                       f"search-only {ss.mean():.2f}x)",
        "RUNTIME_CONCLUSION": (
            f"On matched settings (N={cfg_cpu.pop_size}, G={cfg_cpu.max_gen}, S={cfg_cpu.mc_samples}, "
            f"same seed), the RTX 5070 Ti runs RAMP {st.mean():.2f}x faster than the i9-14900K CPU "
            f"(total wall-clock; {ss.mean():.2f}x on the evolutionary search alone). "
            + ("Full configuration matched on both devices." if matched
               else f"NOTE: reduced to G={cfg_cpu.max_gen} for a practical CPU run; reported as such.")),
    }
    result = {"matched_full_config": matched, "seeds": seeds, "rows": rows,
              "speedup_total_mean": float(st.mean()), "speedup_total_std": float(st.std(ddof=1) if st.size > 1 else 0),
              "speedup_search_mean": float(ss.mean()), "placeholders_task3": ph}
    with open(os.path.join(OUT, "task3_cpu_gpu.json"), "w") as f:
        json.dump(result, f, indent=2)

    lines = [f"TASK 3 - CPU vs GPU speedup for RAMP ({len(seeds)} matched seeds)", "=" * 60,
             f"config: N={cfg_cpu.pop_size} G={cfg_cpu.max_gen} S={cfg_cpu.mc_samples}  "
             f"(matched={matched})",
             f"{'seed':<6}{'CPU total(s)':>14}{'GPU total(s)':>14}{'speedup':>10}"]
    for r in rows:
        lines.append(f"{r['seed']:<6}{r['t_cpu_total']:>14.1f}{r['t_gpu_total']:>14.1f}{r['speedup_total']:>9.2f}x")
    lines.append(f"\nmean speedup (total)  = {st.mean():.2f}x +/- {st.std(ddof=1) if st.size>1 else 0:.2f}")
    lines.append(f"mean speedup (search) = {ss.mean():.2f}x")
    lines.append("\nFILLED PLACEHOLDERS (Task 3):")
    for k, v in ph.items():
        lines.append(f"{k} = {v}")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "task3_cpu_gpu.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()

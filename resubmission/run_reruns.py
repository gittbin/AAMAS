"""Orchestration driver - runs every dependent compute task in sequence on GPU.

Assumes resubmission/out/rerun_fronts.pkl is complete (Task 1b/2 foundation).
Each step is checkpointed and independently resumable; a failure in one step is
logged and the driver continues to the next.

Order: analyses (fast) -> new GPU runs -> figures -> aggregate.
Usage: python resubmission/run_reruns.py [--seeds 15]
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(name, modname, argv):
    mod = importlib.import_module(f"resubmission.{modname}")
    importlib.reload(mod)
    print(f"\n{'='*80}\n>>> STEP: {name}  ({modname}.main {argv})\n{'='*80}", flush=True)
    t0 = time.perf_counter()
    try:
        mod.main(argv) if argv is not None else mod.main()
        print(f"<<< {name} done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)
        return True
    except Exception:
        print(f"!!! {name} FAILED:\n{traceback.format_exc()}", flush=True)
        return False


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=15)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    s = str(args.seeds)
    dv = args.device

    steps = [
        ("Task 1b IGD", "task1b_igd", None),
        ("Task 2 runtime", "task2_runtime", None),
        ("Task 4 RMP sweep", "task4_rmp_sweep", ["--seeds", s, "--device", dv]),
        ("Tasks 6/7/8 sensitivity", "sensitivity", ["--seeds", s, "--device", dv, "--task", "all"]),
        ("Task 9 power", "task9_power", ["--seeds", s, "--device", dv]),
        ("Task 10 online", "task10_online", ["--device", dv, "--n-online", "30", "--seed", "0"]),
        ("Task 3 cpu/gpu", "task3_cpu_gpu", ["--seeds", "3"]),
        ("Task 1b IGD (refresh)", "task1b_igd", None),
        ("Task 2 runtime (refresh)", "task2_runtime", None),
        ("Figures main", "task13_figures", None),
        ("Figures sensitivity", "task13_sensfigs", None),
        ("Aggregate placeholders", "aggregate_results", None),
    ]
    t0 = time.perf_counter()
    results = {}
    for name, mod, av in steps:
        results[name] = _run(name, mod, av)
    print(f"\n{'#'*80}\nALL RERUN STEPS DONE in {(time.perf_counter()-t0)/60:.1f} min")
    for name, ok in results.items():
        print(f"  {'OK ' if ok else 'FAIL'}  {name}")


if __name__ == "__main__":
    main()

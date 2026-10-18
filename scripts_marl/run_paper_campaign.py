"""Run independent paper-v2 training processes and record every success/failure.

Workers are OS processes running single-thread Torch; no existing process is stopped.
Completed compatible runs are reused; incomplete runs remain available for inspection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts_marl.train_paper_condition import CONDITIONS, source_digest


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--steps", type=int, default=400384)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--out", default="outputs_marl/paper_v2/campaign")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(condition, seed) for condition in args.conditions for seed in args.seeds]
    digest = source_digest()
    state = {"pid": os.getpid(), "complete": False, "source_sha256": digest,
             "start_unix": time.time(), "steps_per_run": args.steps, "workers": args.workers,
             "planned_runs": len(jobs), "successful": [], "failed": []}
    atomic_json(out / "campaign_status.json", state)
    def run(job):
        condition, seed = job
        if source_digest() != digest:
            return {"condition": condition, "seed": seed, "returncode": -1, "error": "Core source changed after lock"}
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        log = out / f"{condition}_seed_{seed}.log"
        command = [sys.executable, "-u", str(ROOT / "scripts_marl/train_paper_condition.py"),
                   "--condition", condition, "--seed", str(seed), "--steps", str(args.steps),
                   "--out", str(out / "train")]
        with log.open("a", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        return {"condition": condition, "seed": seed, "returncode": result.returncode, "log": str(log)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            state["successful" if result["returncode"] == 0 else "failed"].append(result)
            state["last_update_unix"] = time.time()
            atomic_json(out / "campaign_status.json", state)
            print(json.dumps(result), flush=True)
    state.update(training_complete=True, training_end_unix=time.time())
    atomic_json(out / "campaign_status.json", state)
    if state["failed"]:
        state.update(complete=True, evaluation_complete=False, end_unix=time.time())
        atomic_json(out / "campaign_status.json", state)
        return 1
    with (out / "evaluation.log").open("a", encoding="utf-8") as handle:
        result = subprocess.run([sys.executable, "-u", str(ROOT / "scripts_marl/evaluate_paper_v2.py"),
                                 "--campaign", str(out), "--workers", str(args.workers)],
                                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    state.update(complete=True, evaluation_complete=result.returncode == 0,
                 evaluation_returncode=result.returncode, end_unix=time.time())
    atomic_json(out / "campaign_status.json", state)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

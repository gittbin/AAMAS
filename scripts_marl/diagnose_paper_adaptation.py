"""Development-only adaptive/frozen checks of eight locked v3 no-comm policies."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts_marl.run_paper_confirmation import load_selected, sha, verify_sources
from scripts_marl.run_paper_improvements import write_json
from scripts_marl.evaluate_paper_v2 import rollout
from scripts_marl.report_paper_v2 import nested_interval

CAMPAIGN = ROOT / "outputs_marl/paper_v3/confirmation_v2"
OUT = ROOT / "outputs_marl/paper_v4/adaptation_development"


def worker(out, seed):
    torch.set_num_threads(1)
    lock = json.loads((CAMPAIGN / "protocol_locked.json").read_text())
    verify_sources(lock)
    entry = json.loads((CAMPAIGN / "selected_checkpoints_locked.json").read_text())["selected"][f"no_comm/seed_{seed}"]
    cfg, policy = load_selected(CAMPAIGN / "train/no_comm" / f"seed_{seed}", "best", entry["selected_checkpoint_sha256"])
    folder = out / f"seed_{seed}"
    folder.mkdir()
    ids = lock["protocol"]["development_seeds"]
    adaptive = rollout(cfg, policy, {}, ids, folder / "adaptive.csv")
    frozen = rollout(cfg, policy, {"freeze_step": 25}, ids, folder / "frozen25.csv")
    if any(a["prefix25_sha256"] != b["prefix25_sha256"] for a, b in zip(adaptive, frozen)):
        raise ValueError("Adaptive/frozen prefix mismatch")
    write_json(folder / "results.json", {"complete": True, "training_seed": seed, "scope": "Development only",
               "checkpoint_sha256": entry["selected_checkpoint_sha256"], "scenario_ids": ids,
               "treatments": {"adaptive": adaptive, "frozen25": frozen}, "prefix25_matches": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.out, args.worker)
        return
    if args.out.exists():
        raise RuntimeError("Preserve existing development results")
    args.out.mkdir(parents=True)
    start = time.time()
    write_json(args.out / "protocol_locked.json", {"scope": "Development adaptation diagnostic", "freeze_step": 25,
               "seeds": list(range(30, 38)), "scenario_ids": list(range(1000000, 1000030)),
               "selected_manifest_sha256": sha(CAMPAIGN / "selected_checkpoints_locked.json"), "script_sha256": sha(Path(__file__))})
    state = {"complete": False, "stage": "diagnosing", "failed": [], "final_scenarios_opened": False, "start_unix": start}
    write_json(args.out / "campaign_status.json", state)
    def run(seed):
        with (args.out / f"seed_{seed}.log").open("w") as handle:
            return subprocess.run([sys.executable, str(Path(__file__)), "--out", str(args.out), "--worker", str(seed)], stdout=handle, stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(run, seed): seed for seed in range(30, 38)}
        for job in as_completed(jobs):
            code = job.result()
            if code:
                state["failed"].append({"seed": jobs[job], "returncode": code})
            write_json(args.out / "campaign_status.json", state)
    if state["failed"]:
        raise RuntimeError("Preserved worker failures")
    data = [json.loads((args.out / f"seed_{seed}/results.json").read_text()) for seed in range(30, 38)]
    effects = {}
    for metric in ("post25_min_user_rate", "post25_return", "post25_illumination_db", "post25_outage_fraction", "post25_reconfiguration"):
        a = [[e[metric] for e in d["treatments"]["adaptive"]] for d in data]
        b = [[e[metric] for e in d["treatments"]["frozen25"]] for d in data]
        effects[metric] = nested_interval(a, b, paired=True, seed=8642)
    write_json(args.out / "report/results.json", {"complete": True, "scope": "Development diagnostic, not final evidence", "episodes": 480,
               "prefix25_matches": True, "freeze_step": 25, "adaptive_minus_frozen": effects})
    lines = ["# Adaptive và frozen-policy trên development", "", "8 checkpoint no-comm v3 đã khóa; 30 development scenario; 480 episode. Prefix trước freeze=25 giống nhau. Low-level BD giữ dùng estimate mới, power freezing giữ profile rule. Đây là diagnostic, không phải final confirmatory evidence.", "",
             "| Metric sau bước 25 | Adaptive − frozen | CI95 descriptive |", "|---|---:|---:|"]
    lines += [f"| {metric} | {v['mean']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |" for metric, v in effects.items()]
    (args.out / "report/KET_QUA_ADAPTATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    state.update(complete=True, stage="complete", episodes=480, end_unix=time.time())
    write_json(args.out / "campaign_status.json", state)


if __name__ == "__main__":
    main()

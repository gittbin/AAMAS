"""Development-only inbox and fixed-send content probes of locked v3 policies."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from momgrap.marl.communication.graph import valid_action_mask
from scripts_marl.run_paper_confirmation import load_selected, sha, verify_sources
from scripts_marl.run_paper_improvements import write_json
from scripts_marl.evaluate_paper_v2 import rollout

CAMPAIGN = ROOT / "outputs_marl/paper_v3/confirmation_v2"
DEFAULT_OUT = ROOT / "outputs_marl/paper_v4/diagnostics"
AGENTS = ("bs_agent", "ris_agent", "ma_agent")


def inbox_start(cfg, agent, length):
    return length - cfg.communication.max_messages_per_step * (cfg.communication.message_dim + 4) - int(cfg.csi.expose_tau_to_actor) - 1


class ObservedPolicy:
    def __init__(self, policy, cfg, path):
        self.policy, self.cfg = policy, cfg
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer = None
        self.channel_rows = []

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def act(self, obs, env):
        estimate = env.current_estimate()
        direct, reflected, aggregate = estimate.direct, estimate.reflected, estimate.aggregate
        self.channel_rows.append({"reflected_direct_db": float(10 * torch.log10(reflected.abs().square().mean().clamp_min(1e-30) / direct.abs().square().mean().clamp_min(1e-30))),
                                  "aggregate_direct_relative_error": float((aggregate - direct).norm() / direct.norm().clamp_min(1e-30))})
        for index, agent in enumerate(AGENTS):
            raw = torch.as_tensor(obs[agent]).unsqueeze(0)
            start = inbox_start(self.cfg, agent, raw.shape[-1])
            slot = raw[0, start:start + 16]
            normalized = self.policy.obs_rms[agent].normalize(raw)
            normalized_slot = normalized[0, start:start + 16]
            alt = raw.clone()
            # Keep availability and sender identity; zero payload in raw space.
            alt[:, start + 4:start + 16] = 0
            alt = self.policy.obs_rms[agent].normalize(alt)
            pm, cm = env.physical_action_mask(agent).unsqueeze(0), valid_action_mask(index, env.adjacency).unsqueeze(0)
            hidden = self.policy._hidden[agent].clone()
            actual = self.policy.actors[agent](normalized, hidden, torch.ones(1, 1), (pm, cm))
            changed = self.policy.actors[agent](alt, hidden.clone(), torch.ones(1, 1), (pm, cm))
            row = {"scenario_seed": env._seed, "step": env.step_count, "receiver": agent,
                   "available": int(slot[0]), "sender": int(slot[1:4].argmax()) if slot[0] else -1,
                   "delivery_age_steps": self.cfg.communication.delivery_delay_steps if slot[0] else -1,
                   "physical_argmax_changed": int(actual[0].argmax() != changed[0].argmax()),
                   "physical_tv": float((actual[0].softmax(-1) - changed[0].softmax(-1)).abs().sum() / 2),
                   "communication_tv": float((actual[1].softmax(-1) - changed[1].softmax(-1)).abs().sum() / 2),
                   **{f"raw_{j}": float(v) for j, v in enumerate(slot)},
                   **{f"normalized_{j}": float(v) for j, v in enumerate(normalized_slot)}}
            if self.writer is None:
                self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
        return self.policy.act(obs, env)

    def close(self):
        self.handle.close()


def summarize_inbox(path):
    grouped = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["available"]):
                grouped.setdefault((row["receiver"], int(row["sender"])), []).append(row)
    summaries = []
    for (receiver, sender), rows in grouped.items():
        raw = np.array([[float(r[f"raw_{j}"]) for j in range(4, 16)] for r in rows])
        norm = np.array([[float(r[f"normalized_{j}"]) for j in range(4, 16)] for r in rows])
        summaries.append({"receiver": receiver, "sender": AGENTS[sender], "received_intervals": len(rows),
                          "raw_payload_mean": raw.mean(0).tolist(), "raw_payload_std": raw.std(0).tolist(),
                          "normalized_payload_std": norm.std(0).tolist(),
                          "physical_argmax_change_fraction": float(np.mean([int(r["physical_argmax_changed"]) for r in rows])),
                          "physical_tv_mean": float(np.mean([float(r["physical_tv"]) for r in rows]))})
    return summaries


def worker(out, seed):
    torch.set_num_threads(1)
    lock = json.loads((CAMPAIGN / "protocol_locked.json").read_text())
    verify_sources(lock)
    manifest = json.loads((CAMPAIGN / "selected_checkpoints_locked.json").read_text())
    entry = manifest["selected"][f"comm_reference/seed_{seed}"]
    folder = out / f"seed_{seed}"
    folder.mkdir(parents=True)
    write_json(folder / "status.json", {"complete": False, "stage": "diagnosing", "pid": __import__('os').getpid()})
    cfg, policy = load_selected(CAMPAIGN / "train/comm_reference" / f"seed_{seed}", "best", entry["selected_checkpoint_sha256"])
    ids = lock["protocol"]["development_seeds"]
    if set(ids) & set(range(8000000, 8000050)):
        raise ValueError("Development diagnostic must not open final")
    observed = ObservedPolicy(policy, cfg, folder / "inbox.csv")
    try:
        normal = rollout(cfg, observed, {}, ids, folder / "normal.csv")
    finally:
        observed.close()
    with (folder / "normal.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    replay = {(int(r["scenario_seed"]), int(r["step"])): {a: int(r[f"comm_action_{a}"]) for a in AGENTS} for r in rows}
    off = rollout(cfg, policy, {"comm_mode": "none"}, ids, folder / "off.csv")
    zero = rollout(cfg, policy, {"corruption": "zero"}, ids, folder / "zero.csv", replay=replay)
    with (folder / "zero.csv").open(newline="") as handle:
        changed = list(csv.DictReader(handle))
    keys = ("scenario_seed", "step", "requested", "eligible", "delivered", "budget_dropped", "graph_rejected") + tuple(f"comm_action_{a}" for a in AGENTS)
    if len(rows) != len(changed) or any(any(a[k] != b[k] for k in keys) for a, b in zip(rows, changed)):
        raise ValueError("Fixed-send payload intervention changed accounting")
    result = {"complete": True, "scope": "development only", "training_seed": seed, "checkpoint_sha256": entry["selected_checkpoint_sha256"],
              "scenario_ids": ids, "treatments": {"normal": normal, "off": off, "zero": zero},
              "inbox": summarize_inbox(folder / "inbox.csv"),
              "channel": {k: {"mean": float(np.mean([r[k] for r in observed.channel_rows])),
                               "min": min(r[k] for r in observed.channel_rows), "max": max(r[k] for r in observed.channel_rows)} for k in observed.channel_rows[0]}}
    write_json(folder / "results.json", result)
    write_json(folder / "status.json", {"complete": True, "stage": "complete", "episodes": 90})


def summarize(out, seeds):
    data = [json.loads((out / f"seed_{s}/results.json").read_text()) for s in seeds]
    records = []
    for item in data:
        mean = lambda t, m: float(np.mean([e[m] for e in item["treatments"][t]]))
        records.append({"seed": item["training_seed"], "normal_return": mean("normal", "net_return"),
                        "off_return": mean("off", "net_return"), "zero_return": mean("zero", "net_return"),
                        "normal_minus_off": mean("normal", "net_return") - mean("off", "net_return"),
                        "normal_minus_zero": mean("normal", "net_return") - mean("zero", "net_return")})
    write_json(out / "report/results.json", {"complete": True, "scope": "Development diagnostic, not final evidence", "episodes": len(seeds) * 90,
               "per_seed": records, "inbox": [{"seed": d["training_seed"], "slots": d["inbox"], "channel": d["channel"]} for d in data]})
    lines = ["# Chẩn đoán payload/inbox trên development", "", "8 checkpoint đã khóa; 30 development scenario chung; normal/off/zero-payload replay lịch gửi. Đây là development, không phải final.", "",
             "| Seed | Normal return | Off | Zero | Normal − off | Normal − zero |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['seed']} | {r['normal_return']:.3f} | {r['off_return']:.3f} | {r['zero_return']:.3f} | {r['normal_minus_off']:+.3f} | {r['normal_minus_zero']:+.3f} |" for r in records]
    lines += ["", f"Mean normal − off: {np.mean([r['normal_minus_off'] for r in records]):+.3f}; normal − zero: {np.mean([r['normal_minus_zero'] for r in records]):+.3f}.",
              "", "Inbox sensitivity giữ local observation và recurrent hidden state giống nhau, chỉ zero raw payload; sender/availability giữ nguyên. Raw và normalized payload cùng thang STD, counts, physical probability TV/argmax-change nằm trong results.json và seed_*/inbox.csv. Các trường padding có STD=0 là dự kiến. Slot RIS last-move bằng 0 cần sửa metadata trong nhánh mới."]
    (out / "report/KET_QUA_DIAGNOSTIC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.out, args.worker)
        return
    if args.out.exists():
        raise RuntimeError("Preserve existing diagnostics; choose a fresh output directory")
    args.out.mkdir(parents=True)
    start = time.time()
    seeds = list(range(30, 38))
    state = {"complete": False, "stage": "diagnosing", "seeds": seeds, "failed": [], "start_unix": start, "final_scenarios_opened": False}
    write_json(args.out / "campaign_status.json", state)
    write_json(args.out / "protocol_locked.json", {"role": "Development diagnostics", "source_sha256": sha(Path(__file__)),
               "selected_manifest_sha256": sha(CAMPAIGN / "selected_checkpoints_locked.json"), "development_ids": list(range(1000000, 1000030)), "seeds": seeds})
    def run(seed):
        with (args.out / f"seed_{seed}.log").open("w") as handle:
            return subprocess.run([sys.executable, str(Path(__file__)), "--out", str(args.out), "--worker", str(seed)], stdout=handle, stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(run, seed): seed for seed in seeds}
        for future in as_completed(futures):
            seed, code = futures[future], future.result()
            if code:
                state["failed"].append({"seed": seed, "returncode": code})
            print(f"diagnostic seed {seed}: returncode {code}", flush=True)
            write_json(args.out / "campaign_status.json", state)
    if state["failed"]:
        raise RuntimeError("Diagnostic failures; preserved logs contain details")
    summarize(args.out, seeds)
    state.update(complete=True, stage="complete", episodes=720, end_unix=time.time())
    write_json(args.out / "campaign_status.json", state)


if __name__ == "__main__":
    main()

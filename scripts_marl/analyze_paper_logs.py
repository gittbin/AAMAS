"""Read-only training diagnostics; writes separate analysis artifacts, never updates weights."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from momgrap.config import RegimeSpec
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.algorithms.rmappo.trainer import RMAPPOTrainer
from momgrap.marl.algorithms.rmappo.checkpoint import load_checkpoint
from scripts_marl.evaluate_paper_v2 import load_policy, rollout
from scripts_marl.report_paper_v2 import nested_interval
from scripts_marl.paper_seed_audit import training_scenario_ids

CAMPAIGN = ROOT / "outputs_marl/paper_v2/campaign"
OUT = CAMPAIGN / "report/log_analysis"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def gradient_probe(condition, seed, checkpoint):
    path = CAMPAIGN / "train" / condition / f"seed_{seed}" / checkpoint
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    overrides = dict(ck["config"])
    overrides["physical"] = dict(overrides["physical"])
    overrides["physical"]["regimes"] = [RegimeSpec(**x) for x in overrides["physical"]["regimes"]]
    overrides["rmappo"] = dict(overrides["rmappo"], rollout_length=64)
    cfg = load_evocomm_config(ROOT / "configs_marl/paper_v2.yaml", overrides=overrides)
    trainer = RMAPPOTrainer(cfg, lambda i: MOMGRAPParallelEnv(cfg), seed=seed)
    load_checkpoint(path, trainer)
    observations, states = [], []
    for i, env in enumerate(trainer.envs):
        obs, state = trainer._reset_env(env, 1000000 + i)
        observations.append(obs)
        states.append(state)
    trainer._obs = trainer._stack_obs(observations)
    trainer._state = torch.as_tensor(np.stack(states))
    trainer.collect_rollout()  # Fresh diagnostic rollout; not resuming original simulator state.
    batch = next(trainer.buffer.recurrent_minibatches(cfg.rmappo.num_minibatches,
                                                    cfg.rmappo.sequence_length,
                                                    torch.Generator().manual_seed(912)))
    valid, adv = batch["valid"], batch["advantages"]
    count = valid.sum().clamp_min(1)
    actor_loss = torch.zeros(())
    for a in trainer.agents:
        lp, physical_entropy, comm_entropy, _ = trainer.actors[a].evaluate_actions_with_entropies(
            batch["obs"][a], batch["actor_hidden"][a], batch["masks"],
            batch["phys_actions"][a], batch["comm_actions"][a],
            action_masks=(batch["phys_masks"][a], batch["comm_masks"][a]))
        ratio = (lp - batch["log_probs_old"][a]).exp()
        surrogate = torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv)
        comm_coef = cfg.rmappo.comm_entropy_coef
        if comm_coef is None:
            comm_coef = cfg.rmappo.entropy_coef
        actor_loss += ((-surrogate - cfg.rmappo.entropy_coef * physical_entropy
                        - comm_coef * comm_entropy) * valid).sum() / count
    values, _ = trainer.critic.forward(batch["state"], batch["critic_hidden"], batch["masks"])
    clipped = batch["values_old"] + (values - batch["values_old"]).clamp(-0.2, 0.2)
    value_loss = 0.5 * (torch.maximum((values - batch["returns"]) ** 2,
                                    (clipped - batch["returns"]) ** 2) * valid).sum() / count
    trainer.optimizer.zero_grad()
    (actor_loss + cfg.rmappo.value_coef * value_loss).backward()
    def norm(parameters):
        return math.sqrt(sum(float(p.grad.detach().square().sum())
                             for p in parameters if p.grad is not None))
    actor_norms = {a: norm(trainer.actors[a].parameters()) for a in trainer.agents}
    critic_norm = norm(trainer.critic.parameters())
    total = math.sqrt(sum(x * x for x in actor_norms.values()) + critic_norm ** 2)
    result = {"condition": condition, "seed": seed, "checkpoint": checkpoint,
              "scope": "Fresh 64-step x 8-environment development rollout; one PPO minibatch; no optimizer step.",
              "actor_grad_norms": actor_norms, "weighted_critic_grad_norm": critic_norm,
              "global_grad_norm": total, "global_clip_multiplier": min(1, cfg.rmappo.max_grad_norm / total),
              "critic_fraction_of_squared_gradient": critic_norm ** 2 / total ** 2,
              "value_loss": float(value_loss.detach()), "projection_normalized_std": {}}
    for a in trainer.agents:
        result["projection_normalized_std"][a] = trainer.buffer.obs[a][:, :, 8:16].reshape(-1, 8).std(0).tolist()
    for env in trainer.envs:
        env.close()
    assert before == hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"scope": "Exploratory diagnostics; original campaign traces/checkpoints preserved.",
               "training": {}, "seed_overlap": {}, "clean49_sensitivity": {}, "routing": {},
               "gradient_probes": [], "fresh_development_inference": []}
    manifests = {}
    keys = ["train/min_rate", "train/messages", "loss/value", "diagnostics/communication_entropy",
            "diagnostics/physical_entropy", "diagnostics/approx_kl", "diagnostics/clipfrac"]
    for folder in sorted((CAMPAIGN / "train").iterdir()):
        records = []
        manifests[folder.name] = []
        for run in sorted(folder.glob("seed_*")):
            status = json.loads((run / "status.json").read_text())
            rows, dev = read_csv(run / "train.csv"), read_csv(run / "eval.csv")
            windows = {name: {key: float(np.mean([float(x[key]) for x in part])) for key in keys if key in part[0]}
                       for name, part in [("early20", rows[:20]), ("middle20", rows[185:205]), ("late20", rows[-20:])]}
            best = max(dev, key=lambda r: float(r["eval_return_mean"]))
            records.append({"seed": status["training_seed"], "updates": len(rows), "windows": windows,
                            "final_dev_return": float(dev[-1]["eval_return_mean"]),
                            "best_dev_return": float(best["eval_return_mean"]), "best_dev_step": int(best["global_step"]),
                            "nonfinite_cells": sum(not math.isfinite(float(v)) for r in rows for v in r.values())})
            ids = training_scenario_ids(status)
            summary["seed_overlap"][str(status["training_seed"])] = {
                "training_scenario_count_including_resets": len(ids),
                "final_overlap": sorted(ids & set(range(40000, 40050))),
                "training_development_overlap": sorted(ids & {10000, 10001, 10002}),
                "maximum_training_scenario_id": max(ids)}
            manifests[folder.name].append(json.loads((CAMPAIGN / "evaluation" / folder.name / run.name / "results.json").read_text()))
        summary["training"][folder.name] = records
    def matrix(condition, treatment, metric, exclude=False):
        return np.asarray([[e[metric] for e in m["treatments"][treatment]["episodes"]
                            if not exclude or e["scenario_seed"] != 40000] for m in manifests[condition]])
    for condition in ("full_nocomm", "full_comm", "centralized", "feedforward", "masked_csi", "power_ma", "ma"):
        summary["clean49_sensitivity"][condition] = {metric: {
            "original50_mean": float(matrix(condition, "normal", metric).mean()),
            "exclude40000_mean": float(matrix(condition, "normal", metric, True).mean())}
            for metric in ("net_return", "min_user_rate")}
    summary["clean49_comparisons"] = {}
    for label, a, ta, b, tb, paired in [
        ("comm_minus_nocomm", "full_comm", "normal", "full_nocomm", "normal", False),
        ("comm_minus_commoff", "full_comm", "normal", "full_comm", "comm_off", True),
        ("adaptive_minus_frozen25", "full_nocomm", "normal", "full_nocomm", "frozen25", True)]:
        metric = "post25_min_user_rate" if label.startswith("adaptive") else "net_return"
        summary["clean49_comparisons"][label] = nested_interval(matrix(a, ta, metric, True), matrix(b, tb, metric, True), paired=paired)
    for condition in ("full_comm", "price_free", "price_tenfold", "budget3_free", "budget3_tenfold"):
        agents = ("bs_agent", "ris_agent", "ma_agent")
        for seed in range(5):
            rows = read_csv(CAMPAIGN / "evaluation" / condition / f"seed_{seed}" / "normal.csv")
            requested, accepted = Counter(), Counter()
            budget = 3 if condition.startswith("budget3") else 1
            for row in rows:
                eligible = [a for a in agents if int(row[f"comm_action_{a}"]) > 0]
                assert int(row["graph_rejected"]) == 0 and len(eligible[:budget]) == int(row["delivered"])
                requested.update(eligible)
                accepted.update(eligible[:budget])
            physical = {key: dict(Counter(int(r[key]) for r in rows)) for key in ("power_action", "ris_action", "ma_action")}
            summary["routing"][f"{condition}/seed_{seed}"] = {
                "request_rate_by_sender": {a: requested[a] / len(rows) for a in agents},
                "accepted_rate_by_sender_reconstructed_equal_priorities": {a: accepted[a] / len(rows) for a in agents},
                "accepted_share_by_sender": {a: accepted[a] / max(1, sum(accepted.values())) for a in agents},
                "physical_action_counts": physical}
    for condition, seed, checkpoint in [("full_comm", 2, "checkpoint_100352.pt"),
                                        ("full_comm", 2, "checkpoint.pt"),
                                        ("full_comm", 1, "checkpoint.pt")]:
        probe = gradient_probe(condition, seed, checkpoint)
        summary["gradient_probes"].append(probe)
        print("gradient", json.dumps(probe), flush=True)
    # New development scenarios only. These are not an untouched final test set.
    for condition in ("full_comm", "full_nocomm"):
        for seed in range(5):
            folder = CAMPAIGN / "train" / condition / f"seed_{seed}"
            cfg, policy = load_policy(folder)
            for deterministic, replicate in [(True, 0), (False, 0), (False, 1)]:
                policy.deterministic = deterministic
                torch.manual_seed(77100 + seed * 10 + replicate)
                episodes = rollout(cfg, policy, {}, list(range(1000000, 1000008)))
                record = {"condition": condition, "seed": seed, "deterministic": deterministic,
                          "sampling_replicate": replicate, "scenario_ids": list(range(1000000, 1000008)),
                          "episodes": episodes}
                summary["fresh_development_inference"].append(record)
                print("inference", condition, seed, deterministic, replicate,
                      round(float(np.mean([e["net_return"] for e in episodes])), 4), flush=True)
    (OUT / "analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for condition in ("full_comm", "full_nocomm", "budget3_tenfold"):
        all_rows = [read_csv(CAMPAIGN / "train" / condition / f"seed_{s}" / "train.csv") for s in range(5)]
        x = np.array([int(r["global_step"]) for r in all_rows[0]])
        for ax, key in zip(axes, ("train/min_rate", "diagnostics/communication_entropy", "diagnostics/approx_kl")):
            if key not in all_rows[0][0]:
                continue
            y = np.array([[float(r[key]) for r in rows] for rows in all_rows])
            means = np.convolve(y.mean(0), np.ones(15) / 15, mode="valid")
            ax.plot(x[14:], means, label=condition)
            ax.set_xlabel("Training transitions")
    axes[0].set_ylabel("Minimum-user rate (training)")
    axes[1].set_ylabel("Communication entropy (3 agents)")
    axes[1].axhline(3 * math.log(3), color="gray", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Approximate KL (sum of agent heads)")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "learning_diagnostics.pdf")
    fig.savefig(OUT / "learning_diagnostics.png", dpi=160)
    plt.close(fig)
    print("Wrote", OUT / "analysis.json", flush=True)


if __name__ == "__main__":
    main()

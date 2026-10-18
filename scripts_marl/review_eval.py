"""Consolidated, protocol-standardized evaluation of all review controllers.

Every controller is evaluated on the SAME eval seeds [0..4] and reports
return, min_user_rate (fairness), and messages_requested/step (comm-usage
diagnostic). Handles both RMAPPO (decentralized) and Centralized-PPO
checkpoints. Writes outputs_marl/paper/review_eval.json.
"""

from __future__ import annotations

import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch

from momgrap.marl.algorithms.rmappo.centralized import CentralizedPPOTrainer
from momgrap.marl.algorithms.rmappo.checkpoint import load_checkpoint
from momgrap.marl.algorithms.rmappo.trainer import RMAPPOTrainer
from momgrap.marl.communication.modes import comm_mask_for_mode
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv

EVAL_SEEDS = list(range(10))  # match the main-table protocol (paper_results.py --eval-seeds)
MVP = str(_REPO_ROOT / "configs_marl" / "mvp.yaml")


def _factory(cfg, base):
    return lambda i: MOMGRAPParallelEnv(cfg, seed=base * 1000 + i)


def _cfg(overrides=None):
    ov = {"physical": {"device_str": "cpu"}}
    if overrides:
        ov.update(overrides)
    return load_evocomm_config(MVP, overrides=ov)


def eval_rmappo(ckpt_path, active_agents=None, comm_mode="learned", reward_overrides=None):
    """Load an RMAPPO checkpoint and roll out on EVAL_SEEDS collecting rich metrics."""
    cfg = _cfg({"reward": reward_overrides} if reward_overrides else None)
    tr = RMAPPOTrainer(cfg, _factory(cfg, 99), device="cpu", seed=0,
                       active_agents=active_agents, comm_mode=comm_mode)
    load_checkpoint(ckpt_path, tr)
    env = MOMGRAPParallelEnv(cfg, seed=0, eval_mode=True, comm_mode=comm_mode)
    adj = env.adjacency
    comm_masks = {a: comm_mask_for_mode(i, adj, tr.comm_mode).unsqueeze(0).to(tr.device)
                  for i, a in enumerate(tr.agents)}
    rets, minrates, msgs = [], [], []
    for seed in EVAL_SEEDS:
        obs, _ = env.reset(seed=seed)
        h = {a: tr.actors[a].zero_hidden(1, tr.device) for a in tr.agents}
        masks = torch.ones(1, 1, device=tr.device)
        ep_ret = 0.0
        step_min, step_msg = [], []
        while env.agents:
            actions = {}
            for a in tr.agents:
                o = tr.obs_rms[a].normalize(torch.as_tensor(obs[a], device=tr.device).unsqueeze(0))
                act, _lp, h[a] = tr.actors[a].act(
                    o, h[a], masks,
                    action_masks=(tr._phys_mask(a, 1), comm_masks[a]),
                    deterministic=True)
                actions[a] = act[0].cpu().numpy()
            obs, rewards, _te, _tr, info = env.step(actions)
            ep_ret += float(rewards[tr.agents[0]])
            i0 = info[tr.agents[0]]
            step_min.append(float(i0["min_user_rate"]))
            step_msg.append(float(i0["message_requested"]))
        rets.append(ep_ret)
        minrates.append(float(np.mean(step_min)))
        msgs.append(float(np.mean(step_msg)))
    return _stats(rets, minrates, msgs)


def eval_centralized(ckpt_path):
    cfg = _cfg()
    tr = CentralizedPPOTrainer(cfg, _factory(cfg, 99), device="cpu", seed=0)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tr.actor.load_state_dict(ck["actor"])
    tr.critic.load_state_dict(ck["critic"])
    tr.state_rms.load_state_dict(ck["state_rms"])
    env = MOMGRAPParallelEnv(cfg, seed=0, eval_mode=True)
    rets, minrates, msgs = [], [], []
    for seed in EVAL_SEEDS:
        env.reset(seed=seed)
        h = tr.actor.zero_hidden(1, tr.device)
        masks = torch.ones(1, 1, device=tr.device)
        ep = 0.0
        step_min, step_msg = [], []
        while env.agents:
            s = tr.state_rms.normalize(torch.as_tensor(env.state(), device=tr.device).unsqueeze(0))
            a, _lp, h = tr.actor.act(s, h, masks, deterministic=True)
            env_act = {tr.agents[0]: np.array([int(a["power"][0]), 0]),
                       tr.agents[1]: np.array([int(a["ris"][0]), 0]),
                       tr.agents[2]: np.array([int(a["ma"][0]), 0])}
            _o, rew, _te, _tr, info = env.step(env_act)
            ep += float(rew[tr.agents[0]])
            i0 = info[tr.agents[0]]
            step_min.append(float(i0["min_user_rate"]))
            step_msg.append(float(i0["message_requested"]))
        rets.append(ep)
        minrates.append(float(np.mean(step_min)))
        msgs.append(float(np.mean(step_msg)))
    return _stats(rets, minrates, msgs)


def eval_noop():
    cfg = _cfg()
    env = MOMGRAPParallelEnv(cfg, seed=0, eval_mode=True)
    agents = list(env.possible_agents)
    rets, minrates = [], []
    for seed in EVAL_SEEDS:
        env.reset(seed=seed)
        ep = 0.0
        step_min = []
        while env.agents:
            act = {a: np.array([0, 0]) for a in agents}
            _o, rew, _te, _tr, info = env.step(act)
            ep += float(rew[agents[0]])
            step_min.append(float(info[agents[0]]["min_user_rate"]))
        rets.append(ep)
        minrates.append(float(np.mean(step_min)))
    return _stats(rets, minrates, [0.0] * len(rets))


def _stats(rets, minrates, msgs):
    return {
        "return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)),
        "min_rate_mean": float(np.mean(minrates)), "min_rate_std": float(np.std(minrates)),
        "msg_per_step_mean": float(np.mean(msgs)), "msg_per_step_std": float(np.std(msgs)),
        "returns": rets, "n": len(rets),
    }


def _agg(seed_results):
    """Aggregate several per-seed dicts (each already averaged over EVAL_SEEDS)."""
    keys = ["return_mean", "min_rate_mean", "msg_per_step_mean"]
    out = {}
    for k in keys:
        vals = [r[k] for r in seed_results]
        out[k] = float(np.mean(vals))
        out[k.replace("_mean", "_sd_across_seeds")] = float(np.std(vals))
    out["n_train_seeds"] = len(seed_results)
    out["per_seed"] = seed_results
    return out


def main():
    rev = _REPO_ROOT / "outputs_marl" / "review"
    ms = _REPO_ROOT / "outputs_marl" / "multiseed"
    results = {}

    print("[noop]", flush=True)
    results["noop"] = eval_noop()

    # Full MARL @200k (fair budget vs centralized) and @400k (final) -- 5 train seeds.
    print("[full MARL @200k]", flush=True)
    results["marl_200k"] = _agg([eval_rmappo(ms / f"seed{s}" / "checkpoint_200704.pt")
                                 for s in range(5)])
    print("[full MARL @400k]", flush=True)
    results["marl_400k"] = _agg([eval_rmappo(ms / f"seed{s}" / "checkpoint.pt")
                                 for s in range(5)])

    print("[centralized PPO @200k]", flush=True)
    results["centralized"] = _agg([eval_centralized(rev / f"centralized_seed{s}" / "checkpoint.pt")
                                   for s in range(3)])

    for active, tag in (["bs_agent"], "power_only"), (["ris_agent"], "ris_only"), (["ma_agent"], "ma_only"):
        print(f"[ablation {tag}]", flush=True)
        results[f"ablation_{tag}"] = _agg([
            eval_rmappo(rev / f"ablation_{tag}_seed{s}" / "checkpoint.pt", active_agents=active)
            for s in range(2)])

    for wmsg, tag in (0.0, "w0"), (0.1, "w10x"):
        print(f"[msgcost {tag}]", flush=True)
        results[f"msgcost_{tag}"] = _agg([
            eval_rmappo(rev / f"msgcost_{tag}_seed{s}" / "checkpoint.pt",
                        comm_mode="learned", reward_overrides={"w_message": wmsg})
            for s in range(2)])

    outp = _REPO_ROOT / "outputs_marl" / "paper" / "review_eval.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Compact console summary
    print("\n=== REVIEW EVAL (eval seeds 0-4) ===")
    print(f"{'controller':<20}{'return':>10}{'min_rate':>10}{'msg/step':>10}")
    for k, v in results.items():
        print(f"{k:<20}{v['return_mean']:>10.2f}{v['min_rate_mean']:>10.3f}{v['msg_per_step_mean']:>10.3f}")
    print(f"\nwritten -> {outp}")


if __name__ == "__main__":
    main()

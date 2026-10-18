"""Synchronous recurrent MAPPO trainer (CTDE): per-agent actors + shared critic.

Collects a fixed-length rollout across ``num_envs`` synchronous environments, computes
GAE (with explicit truncation bootstrapping), and runs clipped PPO updates over
recurrent sequence-chunk minibatches. Observations / centralized states are normalised
with running statistics. On any non-finite loss it saves an emergency checkpoint and
raises (never silently continues). Logs to CSV (+ TensorBoard if available).
"""

from __future__ import annotations

import csv
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn

from ...actions.codebooks import ma_action_count, power_action_count, ris_action_count
from ...communication.modes import comm_mask_for_mode
from ...normalization import RunningMeanStd
from ...observations.builder import centralized_state_size, observation_sizes
from .buffer import RecurrentRolloutBuffer
from .checkpoint import save_checkpoint
from .networks import build_actors, build_critic

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:  # pragma: no cover
    _HAS_TB = False


class RMAPPOTrainer:
    def __init__(self, evo_cfg, env_factory, device="cpu", log_dir=None, seed=None,
                 comm_mode="learned", active_agents=None):
        self.evo = evo_cfg
        self.rc = evo_cfg.rmappo
        self.comm_mode = comm_mode if evo_cfg.communication.enabled else "none"
        self.device = torch.device(device)
        self.seed = self.rc.seed if seed is None else int(seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.env_factory = env_factory
        self.num_envs = self.rc.num_envs
        self.envs = [env_factory(i) for i in range(self.num_envs)]
        self.agents = list(self.envs[0].possible_agents)

        self.obs_sizes = observation_sizes(evo_cfg)
        self.state_dim = centralized_state_size(evo_cfg)
        self.n_phys = _phys_counts(evo_cfg)
        # agent ablation: inactive agents are forced to physical action 0 (no-op / equal power)
        self.active_agents = set(self.agents if active_agents is None else active_agents)

        self.actors = {a: m.to(self.device) for a, m in build_actors(evo_cfg, self.obs_sizes).items()}
        self.critic = build_critic(evo_cfg, self.state_dim).to(self.device)
        params = [p for act in self.actors.values() for p in act.parameters()]
        params += list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.rc.learning_rate, eps=1e-5)
        self.total_updates = max(1, self.rc.total_timesteps // (self.rc.rollout_length * self.num_envs))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda u: max(0.0, 1.0 - u / self.total_updates)
        )

        self.obs_rms = {a: RunningMeanStd(self.obs_sizes[a], device=self.device) for a in self.agents}
        self.state_rms = RunningMeanStd(self.state_dim, device=self.device)
        self.projection = getattr(self.envs[0], "projection", None)

        self.buffer = RecurrentRolloutBuffer(
            self.rc.rollout_length, self.num_envs, self.obs_sizes, self.n_phys,
            self.state_dim, self.rc.hidden_size, device=self.device, agents=tuple(self.agents),
        )

        # communication masks (constant per config): (N, 3) per agent
        adj = self.envs[0].adjacency
        self._comm_masks = {
            a: comm_mask_for_mode(i, adj, self.comm_mode).unsqueeze(0)
                .expand(self.num_envs, 3).contiguous().to(self.device)
            for i, a in enumerate(self.agents)
        }

        self.global_step = 0
        self.update_index = 0
        self._eval_env = None

        self.log_dir = pathlib.Path(log_dir) if log_dir else None
        self.writer = None
        self.csv_path = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path = self.log_dir / "train.csv"
            self._csv_header_written = False
            if _HAS_TB:
                self.writer = SummaryWriter(str(self.log_dir))

        self._init_runtime()

    # ------------------------------------------------------------------ runtime
    def _reset_env(self, env, seed):
        obs, _infos = env.reset(seed=seed)
        return obs, env.state()

    def _stack_obs(self, obs_list):
        return {a: torch.as_tensor(np.stack([o[a] for o in obs_list]), device=self.device)
                for a in self.agents}

    def _init_runtime(self):
        obs_list, state_list = [], []
        for i, env in enumerate(self.envs):
            obs, state = self._reset_env(env, seed=self.seed * 1000 + i)
            obs_list.append(obs)
            state_list.append(state)
        self._obs = self._stack_obs(obs_list)
        self._state = torch.as_tensor(np.stack(state_list), device=self.device)
        self._actor_h = {a: self.actors[a].zero_hidden(self.num_envs, self.device) for a in self.agents}
        self._critic_h = self.critic.zero_hidden(self.num_envs, self.device)
        self._masks = torch.ones(self.num_envs, 1, device=self.device)

    def _norm_obs(self, obs):
        return {a: self.obs_rms[a].normalize(obs[a]) for a in self.agents}

    def _phys_mask(self, agent, batch_size, envs=None):
        envs = self.envs if envs is None else envs
        if len(envs) != batch_size:
            raise ValueError("physical masks require the corresponding decision-time environments")
        m = torch.stack([e.physical_action_mask(agent).to(self.device)
                         if hasattr(e, "physical_action_mask") else
                         torch.ones(self.n_phys[agent], dtype=torch.bool, device=self.device)
                         for e in envs])
        if agent not in self.active_agents:
            m[:] = False
            m[:, 0] = True
        return m

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def collect_rollout(self):
        self.buffer.step = 0
        raw_obs = {a: [] for a in self.agents}
        raw_state = []
        self._terminal_values = torch.zeros(self.rc.rollout_length, self.num_envs, device=self.device)
        rollout_metrics = {"return": 0.0, "min_rate": [], "sensing_db": [], "outage": [],
                           "messages": [], "reconfiguration": [], "violation": []}

        for _t in range(self.rc.rollout_length):
            norm_obs = self._norm_obs(self._obs)
            norm_state = self.state_rms.normalize(self._state)
            actions, log_probs, next_actor_h = {}, {}, {}
            physical_masks = {a: self._phys_mask(a, self.num_envs) for a in self.agents}
            for a in self.agents:
                act, lp, h = self.actors[a].act(
                    norm_obs[a], self._actor_h[a], self._masks,
                    action_masks=(physical_masks[a], self._comm_masks[a]),
                )
                actions[a], log_probs[a], next_actor_h[a] = act, lp, h
            value, next_critic_h = self.critic.forward(norm_state, self._critic_h, self._masks)

            # step envs
            rewards = torch.zeros(self.num_envs, device=self.device)
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            terminal_values = torch.zeros(self.num_envs, device=self.device)
            next_obs_list = [None] * self.num_envs
            next_state = self._state.clone()
            next_masks = torch.ones(self.num_envs, 1, device=self.device)

            for n, env in enumerate(self.envs):
                act_n = {a: actions[a][n].cpu().numpy() for a in self.agents}
                obs_n, rew_n, term_n, trunc_n, info_n = env.step(act_n)
                rewards[n] = float(rew_n[self.agents[0]])
                is_trunc = bool(trunc_n[self.agents[0]])
                is_term = bool(term_n[self.agents[0]])
                truncated[n] = is_trunc
                terminated[n] = is_term
                self._accumulate_info(rollout_metrics, info_n)
                if is_trunc or is_term:
                    if is_trunc:
                        term_state = torch.as_tensor(env.state(), device=self.device).unsqueeze(0)
                        tv, _ = self.critic.forward(
                            self.state_rms.normalize(term_state),
                            next_critic_h[n:n + 1], torch.ones(1, 1, device=self.device),
                        )
                        terminal_values[n] = tv.squeeze(0)
                    obs_r, state_r = self._reset_env(env, seed=self.seed * 1000 + n + self.global_step + 1)
                    next_obs_list[n] = obs_r
                    next_state[n] = torch.as_tensor(state_r, device=self.device)
                    next_masks[n] = 0.0
                else:
                    next_obs_list[n] = obs_n
                    next_state[n] = torch.as_tensor(env.state(), device=self.device)
                    next_masks[n] = 1.0

            rollout_metrics["return"] += float(rewards.mean())

            # store the transition (now with rewards + term/trunc)
            self.buffer.insert(
                obs=norm_obs, state=norm_state,
                phys_actions={a: actions[a][:, 0] for a in self.agents},
                comm_actions={a: actions[a][:, 1] for a in self.agents},
                log_probs=log_probs, values=value, rewards=rewards,
                terminated=terminated, truncated=truncated,
                masks=self._masks, actor_hidden=self._actor_h, critic_hidden=self._critic_h,
                phys_masks=physical_masks, comm_masks=self._comm_masks,
            )
            for a in self.agents:
                raw_obs[a].append(self._obs[a])
            raw_state.append(self._state)
            self._terminal_values[self.buffer.step - 1] = terminal_values

            # advance
            self._obs = self._stack_obs(next_obs_list)
            self._state = next_state
            self._actor_h = {a: next_actor_h[a] * next_masks for a in self.agents}
            self._critic_h = next_critic_h * next_masks
            self._masks = next_masks
            self.global_step += self.num_envs

        # bootstrap value
        with torch.no_grad():
            bootstrap, _ = self.critic.forward(
                self.state_rms.normalize(self._state), self._critic_h, self._masks
            )
        self.buffer.compute_returns(bootstrap, self.rc.gamma, self.rc.gae_lambda,
                                    terminal_values=self._terminal_values)

        # update running normalisation with the raw rollout data
        for a in self.agents:
            self.obs_rms[a].update(torch.cat(raw_obs[a], dim=0))
        self.state_rms.update(torch.cat(raw_state, dim=0))

        metrics = {"return": rollout_metrics["return"]}
        for k in ("min_rate", "sensing_db", "outage", "messages", "reconfiguration", "violation"):
            vals = rollout_metrics[k]
            metrics[k] = float(np.mean(vals)) if vals else 0.0
        return metrics

    @staticmethod
    def _accumulate_info(metrics, info):
        i = info.get("bs_agent", {}) if isinstance(info, dict) else {}
        if not i:
            return
        metrics["min_rate"].append(i.get("min_user_rate", 0.0))
        metrics["sensing_db"].append(i.get("min_sensing_illumination_db", 0.0))
        metrics["outage"].append(i.get("outage_count", 0))
        metrics["messages"].append(i.get("message_delivered", 0))
        metrics["reconfiguration"].append(i.get("reconfiguration_cost", 0.0))
        metrics["violation"].append(i.get("violation_before", 0.0))

    # ------------------------------------------------------------------ update
    def update(self):
        rc = self.rc
        clip = rc.clip_coef
        stats = {"policy": [], "value": [], "entropy": [], "physical_entropy": [], "communication_entropy": [], "approx_kl": [], "clipfrac": []}
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + self.update_index)

        for _epoch in range(rc.update_epochs):
            for batch in self.buffer.recurrent_minibatches(rc.num_minibatches, rc.sequence_length, gen):
                valid = batch["valid"]
                n_valid = valid.sum().clamp_min(1.0)
                adv = batch["advantages"]

                policy_loss = torch.zeros((), device=self.device)
                entropy = torch.zeros((), device=self.device)
                physical_entropy = torch.zeros((), device=self.device)
                communication_entropy = torch.zeros((), device=self.device)
                approx_kl = torch.zeros((), device=self.device)
                clipfrac = torch.zeros((), device=self.device)
                for a in self.agents:
                    new_lp, phys_ent, comm_ent, _ = self.actors[a].evaluate_actions_with_entropies(
                        batch["obs"][a], batch["actor_hidden"][a], batch["masks"],
                        batch["phys_actions"][a], batch["comm_actions"][a],
                        action_masks=(batch["phys_masks"][a],
                                      batch["comm_masks"][a]),
                    )
                    old_lp = batch["log_probs_old"][a]
                    logratio = new_lp - old_lp
                    ratio = logratio.exp()
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
                    pl = -torch.min(surr1, surr2)
                    policy_loss = policy_loss + (pl * valid).sum() / n_valid
                    physical_entropy = physical_entropy + (phys_ent * valid).sum() / n_valid
                    communication_entropy = communication_entropy + (comm_ent * valid).sum() / n_valid
                    entropy = physical_entropy + communication_entropy
                    approx_kl = approx_kl + (((ratio - 1) - logratio) * valid).sum() / n_valid
                    clipfrac = clipfrac + (((ratio - 1).abs() > clip).float() * valid).sum() / n_valid

                values, _ = self.critic.forward(batch["state"], batch["critic_hidden"], batch["masks"])
                returns = batch["returns"]
                v_old = batch["values_old"]
                v_clipped = v_old + (values - v_old).clamp(-clip, clip)
                v_loss = torch.max((values - returns) ** 2, (v_clipped - returns) ** 2)
                value_loss = 0.5 * (v_loss * valid).sum() / n_valid

                comm_coef = rc.entropy_coef if rc.comm_entropy_coef is None else rc.comm_entropy_coef
                loss = policy_loss - rc.entropy_coef * physical_entropy - comm_coef * communication_entropy + rc.value_coef * value_loss

                if not torch.isfinite(loss):
                    self._on_nan(loss, policy_loss, value_loss, entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for act in self.actors.values() for p in act.parameters()]
                    + list(self.critic.parameters()), rc.max_grad_norm)
                self.optimizer.step()

                stats["policy"].append(float(policy_loss.detach()))
                stats["value"].append(float(value_loss.detach()))
                stats["entropy"].append(float(entropy.detach()))
                stats["physical_entropy"].append(float(physical_entropy.detach()))
                stats["communication_entropy"].append(float(communication_entropy.detach()))
                stats["approx_kl"].append(float(approx_kl.detach()))
                stats["clipfrac"].append(float(clipfrac.detach()))

        self.scheduler.step()
        self.update_index += 1

        ev = self._explained_variance()
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["explained_variance"] = ev
        return out

    def _explained_variance(self) -> float:
        y = self.buffer.returns.reshape(-1)
        pred = self.buffer.values.reshape(-1)
        var_y = y.var()
        return float(1.0 - (y - pred).var() / var_y) if float(var_y) > 1e-8 else 0.0

    def _on_nan(self, loss, policy_loss, value_loss, entropy):
        meta = {"update_index": self.update_index, "global_step": self.global_step,
                "loss": float(loss.detach()), "policy": float(policy_loss.detach()),
                "value": float(value_loss.detach()), "entropy": float(entropy.detach())}
        if self.log_dir is not None:
            emergency = self.log_dir / "emergency_checkpoint.pt"
            save_checkpoint(emergency, self)
            meta["emergency_checkpoint"] = str(emergency)
        raise RuntimeError(f"non-finite loss encountered; aborting. metadata={meta}")

    # ------------------------------------------------------------------ train
    def train(self, total_timesteps=None, curriculum=None, eval_seeds=(0, 1, 2), progress_callback=None):
        target = total_timesteps or self.rc.total_timesteps
        next_eval = self.rc.eval_interval
        next_save = self.rc.save_interval
        while self.global_step < target:
            if curriculum is not None:
                cost = curriculum.cost_at(self.update_index)
                if cost is not None:
                    for env in self.envs:
                        env.set_message_cost(cost)
            train_metrics = self.collect_rollout()
            loss_metrics = self.update()
            self._log(train_metrics, loss_metrics)

            if self.rc.eval_interval > 0 and self.global_step >= next_eval:
                ev = self.evaluate(list(eval_seeds))
                self._log_eval(ev)
                next_eval += self.rc.eval_interval
            if (self.log_dir is not None and self.rc.save_interval > 0
                    and self.global_step >= next_save):
                save_checkpoint(self.log_dir / f"checkpoint_{self.global_step}.pt", self)
                next_save += self.rc.save_interval
            if progress_callback is not None:
                progress_callback(self.global_step, self.update_index)
        return {"global_step": self.global_step, "updates": self.update_index}

    def _log_eval(self, ev: dict) -> None:
        if self.writer is not None:
            self.writer.add_scalar("eval/return_mean", ev["eval_return_mean"], self.global_step)
        if self.csv_path is not None:
            evfile = self.csv_path.parent / "eval.csv"
            write_header = not evfile.exists()
            with open(evfile, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["global_step", "update_index",
                                                  "eval_return_mean", "eval_return_std"])
                if write_header:
                    w.writeheader()
                w.writerow({"global_step": self.global_step, "update_index": self.update_index,
                            "eval_return_mean": ev["eval_return_mean"],
                            "eval_return_std": ev["eval_return_std"]})

    def _log(self, train_metrics, loss_metrics):
        row = {"global_step": self.global_step, "update_index": self.update_index}
        for k, v in train_metrics.items():
            row[f"train/{k}"] = v
        for k, v in loss_metrics.items():
            prefix = "loss" if k in ("policy", "value", "entropy") else "diagnostics"
            row[f"{prefix}/{k}"] = v
        if self.writer is not None:
            for k, v in row.items():
                if k not in ("global_step", "update_index"):
                    self.writer.add_scalar(k, v, self.global_step)
        if self.csv_path is not None:
            write_header = not self._csv_header_written
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                    self._csv_header_written = True
                w.writerow(row)

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def evaluate(self, seeds, deterministic=True):
        if self._eval_env is None:
            self._eval_env = self.env_factory(10_000)
        env = self._eval_env
        returns = []
        adj = env.adjacency
        comm_masks = {a: comm_mask_for_mode(i, adj, self.comm_mode).unsqueeze(0).to(self.device)
                      for i, a in enumerate(self.agents)}
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            actor_h = {a: self.actors[a].zero_hidden(1, self.device) for a in self.agents}
            masks = torch.ones(1, 1, device=self.device)
            ep_return = 0.0
            while env.agents:
                actions = {}
                for a in self.agents:
                    o = torch.as_tensor(obs[a], device=self.device).unsqueeze(0)
                    o = self.obs_rms[a].normalize(o)
                    act, _lp, actor_h[a] = self.actors[a].act(
                        o, actor_h[a], masks,
                        action_masks=(self._phys_mask(a, 1, [env]), comm_masks[a]),
                        deterministic=deterministic)
                    actions[a] = act[0].cpu().numpy()
                obs, rewards, _term, _trunc, _info = env.step(actions)
                ep_return += float(rewards[self.agents[0]])
                masks = torch.ones(1, 1, device=self.device)
            returns.append(ep_return)
        return {"eval_return_mean": float(np.mean(returns)),
                "eval_return_std": float(np.std(returns)), "n": len(returns)}


def _phys_counts(evo_cfg) -> dict:
    from ...actions.codebooks import ma_action_count, power_action_count, ris_action_count
    return {
        "bs_agent": power_action_count(evo_cfg.actions),
        "ris_agent": ris_action_count(evo_cfg.actions),
        "ma_agent": ma_action_count(evo_cfg.actions),
    }

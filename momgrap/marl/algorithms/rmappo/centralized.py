"""Centralized single-agent recurrent PPO baseline.

One recurrent actor observes the union of local decision-time observations and emits the joint physical
action (power, RIS, MA) with no communication. This is the reference against which the
heterogeneous multi-agent decomposition is judged (RQ1: is MARL justified?). It reuses
the RNN core, GAE, and running normalization; the rollout/update loop is a single-agent
specialization of :class:`RMAPPOTrainer`.
"""

from __future__ import annotations

import csv
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ...actions.codebooks import ma_action_count, power_action_count, ris_action_count
from ...normalization import RunningMeanStd
from ...observations.builder import centralized_state_size, observation_sizes
from .buffer import compute_gae
from .networks import RNNLayer, _init_linear

_HEADS = ("power", "ris", "ma")


class CentralizedActor(nn.Module):
    def __init__(self, state_dim, n_power, n_ris, n_ma, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            _init_linear(nn.Linear(state_dim, hidden_size)), nn.Tanh(),
            _init_linear(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        self.rnn = RNNLayer(hidden_size, hidden_size)
        self.heads = nn.ModuleDict({
            "power": _init_linear(nn.Linear(hidden_size, n_power), gain=0.01),
            "ris": _init_linear(nn.Linear(hidden_size, n_ris), gain=0.01),
            "ma": _init_linear(nn.Linear(hidden_size, n_ma), gain=0.01)})

    def zero_hidden(self, batch, device=None):
        device = device or next(self.parameters()).device
        return torch.zeros(batch, self.hidden_size, device=device)

    def _dists(self, state, hidden, masks, action_masks=None):
        x = self.encoder(state)
        x, hidden = self.rnn(x, hidden, masks)
        logits = {k: self.heads[k](x) for k in _HEADS}
        if action_masks is not None:
            logits = {k: logits[k].masked_fill(~action_masks[k], -1e8) for k in _HEADS}
        return {k: Categorical(logits=logits[k]) for k in _HEADS}, hidden

    def act(self, state, hidden, masks, deterministic=False, action_masks=None):
        dists, hidden = self._dists(state, hidden, masks, action_masks)
        acts, logp = {}, 0.0
        for k in _HEADS:
            a = dists[k].probs.argmax(-1) if deterministic else dists[k].sample()
            acts[k] = a
            logp = logp + dists[k].log_prob(a)
        return acts, logp, hidden

    def evaluate(self, state, hidden, masks, actions, action_masks=None):
        dists, hidden = self._dists(state, hidden, masks, action_masks)
        logp = sum(dists[k].log_prob(actions[k]) for k in _HEADS)
        ent = sum(dists[k].entropy() for k in _HEADS)
        return logp, ent, hidden


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            _init_linear(nn.Linear(state_dim, hidden_size)), nn.Tanh(),
            _init_linear(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        self.rnn = RNNLayer(hidden_size, hidden_size)
        self.value_head = _init_linear(nn.Linear(hidden_size, 1), gain=1.0)

    def zero_hidden(self, batch, device=None):
        device = device or next(self.parameters()).device
        return torch.zeros(batch, self.hidden_size, device=device)

    def forward(self, state, hidden, masks):
        x = self.encoder(state)
        x, hidden = self.rnn(x, hidden, masks)
        return self.value_head(x).squeeze(-1), hidden


class CentralizedPPOTrainer:
    def __init__(self, evo_cfg, env_factory, device="cpu", log_dir=None, seed=None):
        self.evo = evo_cfg
        self.rc = evo_cfg.rmappo
        self.device = torch.device(device)
        self.seed = self.rc.seed if seed is None else int(seed)
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        self.num_envs = self.rc.num_envs
        self.env_factory = env_factory
        self.envs = [env_factory(i) for i in range(self.num_envs)]
        self.agents = list(self.envs[0].possible_agents)
        self.state_dim = centralized_state_size(evo_cfg)
        self.obs_dim = sum(observation_sizes(evo_cfg).values())
        self.actor = CentralizedActor(
            self.obs_dim, power_action_count(evo_cfg.actions),
            ris_action_count(evo_cfg.actions), ma_action_count(evo_cfg.actions),
            self.rc.centralized_actor_hidden_size or self.rc.hidden_size).to(self.device)
        self.critic = CentralizedCritic(self.state_dim, self.rc.hidden_size).to(self.device)
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.rc.learning_rate, eps=1e-5)
        total_updates = max(1, self.rc.total_timesteps // (self.rc.rollout_length * self.num_envs))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda u: max(0.0, 1 - u / total_updates))
        self.state_rms = RunningMeanStd(self.state_dim, device=self.device)
        self.actor_obs_rms = RunningMeanStd(self.obs_dim, device=self.device)
        self.global_step = 0
        self.update_index = 0
        self._eval_env = None
        self.log_dir = pathlib.Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self._init_runtime()

    def _reset(self, env, seed):
        env.reset(seed=seed)
        return env.state()

    def _init_runtime(self):
        self._state = torch.as_tensor(
            np.stack([self._reset(e, self.seed * 1000 + i) for i, e in enumerate(self.envs)]),
            device=self.device)
        self._h_a = self.actor.zero_hidden(self.num_envs, self.device)
        self._actor_obs = torch.as_tensor(np.stack([e.centralized_observation() for e in self.envs]), device=self.device)
        self._h_c = self.critic.zero_hidden(self.num_envs, self.device)
        self._masks = torch.ones(self.num_envs, 1, device=self.device)

    @torch.no_grad()
    def collect(self):
        T, N = self.rc.rollout_length, self.num_envs
        states = torch.zeros(T, N, self.state_dim, device=self.device)
        actor_obs = torch.zeros(T, N, self.obs_dim, device=self.device)
        action_masks = {k: torch.ones(T, N, head.out_features, dtype=torch.bool, device=self.device)
                        for k, head in self.actor.heads.items()}
        acts = {k: torch.zeros(T, N, dtype=torch.long, device=self.device) for k in _HEADS}
        logps = torch.zeros(T, N, device=self.device)
        values = torch.zeros(T, N, device=self.device)
        rewards = torch.zeros(T, N, device=self.device)
        term = torch.zeros(T, N, dtype=torch.bool, device=self.device)
        trunc = torch.zeros(T, N, dtype=torch.bool, device=self.device)
        masks = torch.ones(T, N, 1, device=self.device)
        h_a0 = torch.zeros(T, N, self.actor.hidden_size, device=self.device)
        h_c0 = torch.zeros(T, N, self.critic.hidden_size, device=self.device)
        term_vals = torch.zeros(T, N, device=self.device)
        raw_states = []
        raw_actor_obs = []
        ret = 0.0

        for t in range(T):
            ns = self.state_rms.normalize(self._state)
            no = self.actor_obs_rms.normalize(self._actor_obs)
            am = {k: torch.stack([e.physical_action_mask(agent).to(self.device) for e in self.envs])
                  for k, agent in zip(_HEADS, self.agents)}
            a, lp, nha = self.actor.act(no, self._h_a, self._masks, action_masks=am)
            v, nhc = self.critic.forward(ns, self._h_c, self._masks)
            states[t] = ns
            actor_obs[t] = no
            for k in _HEADS:
                action_masks[k][t] = am[k]
            for k in _HEADS:
                acts[k][t] = a[k]
            logps[t] = lp; values[t] = v; masks[t] = self._masks
            h_a0[t] = self._h_a; h_c0[t] = self._h_c
            raw_states.append(self._state)
            raw_actor_obs.append(self._actor_obs)

            next_state = self._state.clone()
            next_masks = torch.ones(N, 1, device=self.device)
            for n, env in enumerate(self.envs):
                env_act = {self.agents[0]: np.array([int(a["power"][n]), 0]),
                           self.agents[1]: np.array([int(a["ris"][n]), 0]),
                           self.agents[2]: np.array([int(a["ma"][n]), 0])}
                _o, rew, te, tr, _i = env.step(env_act)
                rewards[t, n] = float(rew[self.agents[0]])
                is_tr = bool(tr[self.agents[0]]); trunc[t, n] = is_tr; term[t, n] = bool(te[self.agents[0]])
                if is_tr or bool(te[self.agents[0]]):
                    ts = torch.as_tensor(env.state(), device=self.device).unsqueeze(0)
                    tv, _ = self.critic.forward(self.state_rms.normalize(ts), nhc[n:n + 1],
                                                torch.ones(1, 1, device=self.device))
                    term_vals[t, n] = tv.squeeze(0)
                    next_state[n] = torch.as_tensor(self._reset(env, self.seed * 1000 + n + self.global_step + 1),
                                                    device=self.device)
                    next_masks[n] = 0.0
                else:
                    next_state[n] = torch.as_tensor(env.state(), device=self.device)
            ret += float(rewards[t].mean())
            self._state = next_state
            self._actor_obs = torch.as_tensor(np.stack([e.centralized_observation() for e in self.envs]), device=self.device)
            self._h_a = nha * next_masks
            self._h_c = nhc * next_masks
            self._masks = next_masks
            self.global_step += N

        with torch.no_grad():
            boot, _ = self.critic.forward(self.state_rms.normalize(self._state), self._h_c, self._masks)
        adv, rets = compute_gae(rewards, values, boot, term, trunc, self.rc.gamma, self.rc.gae_lambda,
                                terminal_values=term_vals)
        self.state_rms.update(torch.cat(raw_states, dim=0))
        self.actor_obs_rms.update(torch.cat(raw_actor_obs, dim=0))
        self._roll = dict(states=states, acts=acts, logps=logps, values=values, masks=masks,
                          actor_obs=actor_obs, action_masks=action_masks,
                          h_a0=h_a0, h_c0=h_c0, adv=adv, rets=rets)
        return {"return": ret}

    def update(self):
        r = self._roll
        T, N = self.rc.rollout_length, self.num_envs
        L = self.rc.sequence_length
        clip = self.rc.clip_coef
        adv = (r["adv"] - r["adv"].mean()) / (r["adv"].std() + 1e-8)
        n_chunks = T // L
        gen = torch.Generator().manual_seed(self.seed + self.update_index)
        for _e in range(self.rc.update_epochs):
            chunks = [(n, c * L) for n in range(N) for c in range(n_chunks)]
            perm = torch.randperm(len(chunks), generator=gen).tolist()
            mb = max(1, len(chunks) // self.rc.num_minibatches)
            for s in range(0, len(chunks), mb):
                cs = [chunks[i] for i in perm[s:s + mb]]
                m = len(cs)
                def gather(x):
                    g = torch.stack([x[st:st + L, n] for (n, st) in cs], dim=1)
                    return g.reshape(L * m, *g.shape[2:])
                st = gather(r["states"])
                ms = gather(r["masks"])
                h_a = torch.stack([r["h_a0"][st0, n] for (n, st0) in cs], dim=0)
                h_c = torch.stack([r["h_c0"][st0, n] for (n, st0) in cs], dim=0)
                acts = {k: gather(r["acts"][k]) for k in _HEADS}
                old_lp = gather(r["logps"]); a_adv = gather(adv); rets = gather(r["rets"]); v_old = gather(r["values"])
                new_lp, ent, _ = self.actor.evaluate(gather(r["actor_obs"]), h_a, ms, acts,
                                                    {k: gather(r["action_masks"][k]) for k in _HEADS})
                ratio = (new_lp - old_lp).exp()
                pl = -torch.min(ratio * a_adv, torch.clamp(ratio, 1 - clip, 1 + clip) * a_adv).mean()
                v, _ = self.critic.forward(st, h_c, ms)
                vc = v_old + (v - v_old).clamp(-clip, clip)
                vloss = 0.5 * torch.max((v - rets) ** 2, (vc - rets) ** 2).mean()
                loss = pl - self.rc.entropy_coef * ent.mean() + self.rc.value_coef * vloss
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss (centralized)")
                self.optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                         self.rc.max_grad_norm)
                self.optimizer.step()
        self.scheduler.step()
        self.update_index += 1
        return {}

    def train(self, total_timesteps=None, eval_seeds=(0, 1, 2), progress_callback=None):
        target = total_timesteps or self.rc.total_timesteps
        while self.global_step < target:
            tm = self.collect(); self.update()
            if self.log_dir:
                with open(self.log_dir / "train.csv", "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if f.tell() == 0:
                        w.writerow(["global_step", "update_index", "rollout_return", "learning_rate"])
                    w.writerow([self.global_step, self.update_index, tm["return"], self.optimizer.param_groups[0]["lr"]])
            if self.log_dir and self.rc.eval_interval > 0 and self.global_step % max(1, self.rc.eval_interval) < self.num_envs * self.rc.rollout_length:
                ev = self.evaluate(list(eval_seeds))
                with open(self.log_dir / "eval.csv", "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if f.tell() == 0:
                        w.writerow(["global_step", "eval_return_mean", "eval_return_std"])
                    w.writerow([self.global_step, ev["eval_return_mean"], ev["eval_return_std"]])
            if progress_callback is not None:
                progress_callback(self.global_step, self.update_index)
        return {"global_step": self.global_step}

    @torch.no_grad()
    def evaluate(self, seeds, deterministic=True):
        if self._eval_env is None:
            self._eval_env = self.env_factory(10_000)
        env = self._eval_env
        rets = []
        for seed in seeds:
            env.reset(seed=seed)
            h = self.actor.zero_hidden(1, self.device)
            masks = torch.ones(1, 1, device=self.device)
            ep = 0.0
            while env.agents:
                s = self.actor_obs_rms.normalize(torch.as_tensor(env.centralized_observation(), device=self.device).unsqueeze(0))
                am = {k: env.physical_action_mask(agent).unsqueeze(0).to(self.device)
                      for k, agent in zip(_HEADS, self.agents)}
                a, _lp, h = self.actor.act(s, h, masks, deterministic=deterministic, action_masks=am)
                env_act = {self.agents[0]: np.array([int(a["power"][0]), 0]),
                           self.agents[1]: np.array([int(a["ris"][0]), 0]),
                           self.agents[2]: np.array([int(a["ma"][0]), 0])}
                _o, rew, _te, _tr, _i = env.step(env_act)
                ep += float(rew[self.agents[0]])
            rets.append(ep)
        return {"eval_return_mean": float(np.mean(rets)), "eval_return_std": float(np.std(rets))}

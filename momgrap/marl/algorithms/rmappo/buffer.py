"""Recurrent rollout buffer + GAE for RMAPPO.

Stores a time-major (T, N) rollout over N parallel envs with per-agent observations,
actions (physical + communication), action masks, log-probs and recurrent hidden
states, plus the shared centralized state, value, reward, termination / truncation
and RNN reset masks.

GAE handles time-limit truncation explicitly: value bootstrapping uses
``1 - terminated`` (so a truncated episode still bootstraps its terminal value), while
advantage propagation stops at *either* termination or truncation.

Minibatches are sequence chunks (never shuffled at the timestep level): each chunk
carries the hidden state at its first step; short tails are zero-padded and flagged by
a validity mask so the loss ignores padded entries.
"""

from __future__ import annotations

import math

import torch

AGENTS = ("bs_agent", "ris_agent", "ma_agent")


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    lam: float,
    terminal_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation with explicit truncation handling.

    Shapes: rewards/values/terminated/truncated (T,N); bootstrap_value (N,);
    terminal_values (T,N) or None (value of the terminal state at truncation steps).
    Returns (advantages (T,N), returns (T,N)).
    """
    T, N = rewards.shape
    terminated = terminated.float()
    truncated = truncated.float()
    advantages = torch.zeros(T, N, device=rewards.device, dtype=rewards.dtype)
    last_gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        next_value = bootstrap_value if t == T - 1 else values[t + 1]
        if terminal_values is not None:
            next_value = torch.where(truncated[t].bool(), terminal_values[t], next_value)
        nonterminal = 1.0 - terminated[t]
        notdone = 1.0 - torch.clamp(terminated[t] + truncated[t], max=1.0)
        delta = rewards[t] + gamma * nonterminal * next_value - values[t]
        last_gae = delta + gamma * lam * notdone * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


class RecurrentRolloutBuffer:
    def __init__(self, rollout_length, num_envs, obs_sizes, n_phys, state_dim,
                 hidden_size, device="cpu", agents=AGENTS):
        self.T = rollout_length
        self.N = num_envs
        self.agents = tuple(agents)
        self.hidden_size = hidden_size
        self.device = torch.device(device)
        T, N = rollout_length, num_envs

        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=self.device)
        self.obs = {a: z(T, N, obs_sizes[a]) for a in self.agents}
        self.actor_hidden = {a: z(T, N, hidden_size) for a in self.agents}
        self.phys_actions = {a: z(T, N, dt=torch.long) for a in self.agents}
        self.comm_actions = {a: z(T, N, dt=torch.long) for a in self.agents}
        self.log_probs = {a: z(T, N) for a in self.agents}
        self.phys_masks = {a: torch.ones(T, N, n_phys[a], dtype=torch.bool, device=self.device)
                           for a in self.agents}
        self.comm_masks = {a: torch.ones(T, N, 3, dtype=torch.bool, device=self.device)
                           for a in self.agents}

        self.state = z(T, N, state_dim)
        self.critic_hidden = z(T, N, hidden_size)
        self.values = z(T, N)
        self.rewards = z(T, N)
        self.terminated = torch.zeros(T, N, dtype=torch.bool, device=self.device)
        self.truncated = torch.zeros(T, N, dtype=torch.bool, device=self.device)
        self.masks = torch.ones(T, N, 1, device=self.device)   # RNN reset mask

        self.advantages = z(T, N)
        self.returns = z(T, N)
        self.step = 0

    def insert(self, obs, state, phys_actions, comm_actions, log_probs, values, rewards,
               terminated, truncated, masks, actor_hidden, critic_hidden,
               phys_masks=None, comm_masks=None):
        t = self.step
        if t >= self.T:
            raise RuntimeError("buffer is full")
        for a in self.agents:
            self.obs[a][t] = obs[a]
            self.actor_hidden[a][t] = actor_hidden[a]
            self.phys_actions[a][t] = phys_actions[a]
            self.comm_actions[a][t] = comm_actions[a]
            self.log_probs[a][t] = log_probs[a]
            if phys_masks is not None and phys_masks.get(a) is not None:
                self.phys_masks[a][t] = phys_masks[a]
            if comm_masks is not None and comm_masks.get(a) is not None:
                self.comm_masks[a][t] = comm_masks[a]
        self.state[t] = state
        self.critic_hidden[t] = critic_hidden
        self.values[t] = values
        self.rewards[t] = rewards
        self.terminated[t] = terminated
        self.truncated[t] = truncated
        self.masks[t] = masks.reshape(self.N, 1)
        self.step += 1

    def compute_returns(self, bootstrap_value, gamma, lam, terminal_values=None):
        self.advantages, self.returns = compute_gae(
            self.rewards, self.values, bootstrap_value, self.terminated, self.truncated,
            gamma, lam, terminal_values,
        )
        return self.advantages, self.returns

    def normalized_advantages(self) -> torch.Tensor:
        adv = self.advantages
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    # ---------------------------------------------------------------- minibatches
    def _gather(self, tensor_TN, chunks, L):
        mb = len(chunks)
        extra = tuple(tensor_TN.shape[2:])
        out = torch.zeros(L, mb, *extra, dtype=tensor_TN.dtype, device=self.device)
        for j, (n, start) in enumerate(chunks):
            end = min(start + L, self.T)
            out[: end - start, j] = tensor_TN[start:end, n]
        return out

    def recurrent_minibatches(self, num_minibatches, seq_len, generator=None):
        """Yield sequence-chunk minibatches (chunks shuffled, timesteps kept in order)."""
        L = seq_len
        n_chunks_per_env = math.ceil(self.T / L)
        chunks = [(n, c * L) for n in range(self.N) for c in range(n_chunks_per_env)]
        perm = torch.randperm(len(chunks), generator=generator).tolist()
        chunks = [chunks[i] for i in perm]

        adv_norm = self.normalized_advantages()
        mb_size = math.ceil(len(chunks) / num_minibatches)
        for start in range(0, len(chunks), mb_size):
            mb_chunks = chunks[start:start + mb_size]
            mb = len(mb_chunks)

            valid = torch.zeros(L, mb, device=self.device)
            hidden0_actor = {}
            for a in self.agents:
                hidden0_actor[a] = torch.stack(
                    [self.actor_hidden[a][s, n] for (n, s) in mb_chunks], dim=0
                )
            hidden0_critic = torch.stack(
                [self.critic_hidden[s, n] for (n, s) in mb_chunks], dim=0
            )
            for j, (n, s) in enumerate(mb_chunks):
                valid[: min(s + L, self.T) - s, j] = 1.0

            def flat(t_TN):
                g = self._gather(t_TN, mb_chunks, L)          # (L, mb, *)
                return g.reshape(L * mb, *g.shape[2:])

            batch = {
                "obs": {a: flat(self.obs[a]) for a in self.agents},
                "phys_actions": {a: flat(self.phys_actions[a]) for a in self.agents},
                "comm_actions": {a: flat(self.comm_actions[a]) for a in self.agents},
                "phys_masks": {a: flat(self.phys_masks[a]) for a in self.agents},
                "comm_masks": {a: flat(self.comm_masks[a]) for a in self.agents},
                "log_probs_old": {a: flat(self.log_probs[a]) for a in self.agents},
                "actor_hidden": hidden0_actor,
                "critic_hidden": hidden0_critic,
                "state": flat(self.state),
                "values_old": flat(self.values),
                "returns": flat(self.returns),
                "advantages": flat(adv_norm),
                "masks": flat(self.masks),                    # (L*mb, 1)
                "valid": valid.reshape(L * mb),               # (L*mb,)
                "seq_len": L,
                "batch_envs": mb,
            }
            yield batch

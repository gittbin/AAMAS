"""Recurrent MAPPO networks: per-agent recurrent actors + one centralized critic.

Agents are heterogeneous (different obs/action dims) so parameter sharing is not
forced — each agent gets its own :class:`RecurrentActor`. Each actor has two
categorical heads (physical action, communication action) and supports valid-action
masks, a hidden-reset mask, sampling / deterministic action, log-prob and entropy.
The critic maps the centralized state (which may include the true channel) to a
single shared value.

The recurrent core (:class:`RNNLayer`) handles both the single-step rollout path
(``x.size(0) == hidden.size(0)``) and the sequence-chunk training path, applying the
reset mask each step so episode boundaries clear the hidden state.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ...actions.codebooks import ma_action_count, power_action_count, ris_action_count

_MASK_FILL = -1e8
N_COMM_ACTIONS = 3


def _init_linear(layer: nn.Linear, gain: float = math.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _init_gru(gru: nn.GRU) -> nn.GRU:
    for name, p in gru.named_parameters():
        if "weight" in name:
            nn.init.orthogonal_(p)
        elif "bias" in name:
            nn.init.zeros_(p)
    return gru


class RNNLayer(nn.Module):
    """Masked GRU that works step-by-step (rollout) or over sequence chunks (training)."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = _init_gru(nn.GRU(input_size, hidden_size))

    def forward(self, x: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor):
        if x.size(0) == hxs.size(0):
            # single step: (N, feat)
            out, h = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            return out.squeeze(0), h.squeeze(0)
        # sequence: x is (T*N, feat), hxs is (N, hidden), masks is (T*N, 1)
        n = hxs.size(0)
        t = x.size(0) // n
        x = x.view(t, n, -1)
        masks = masks.view(t, n, 1)
        h = hxs.unsqueeze(0)                       # (1, N, hidden)
        outputs = []
        for step in range(t):
            out, h = self.gru(x[step:step + 1], h * masks[step:step + 1])
            outputs.append(out)
        x = torch.cat(outputs, dim=0).view(t * n, -1)
        return x, h.squeeze(0)


class RecurrentActor(nn.Module):
    def __init__(self, obs_dim: int, n_phys_actions: int, n_comm_actions: int = N_COMM_ACTIONS,
                 hidden_size: int = 128, recurrent: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_phys_actions = n_phys_actions
        self.n_comm_actions = n_comm_actions
        self.recurrent = recurrent
        self.encoder = nn.Sequential(
            _init_linear(nn.Linear(obs_dim, hidden_size)), nn.Tanh(),
            _init_linear(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
        )
        self.rnn = RNNLayer(hidden_size, hidden_size) if recurrent else None
        self.phys_head = _init_linear(nn.Linear(hidden_size, n_phys_actions), gain=0.01)
        self.comm_head = _init_linear(nn.Linear(hidden_size, n_comm_actions), gain=0.01)

    def zero_hidden(self, batch: int, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        return torch.zeros(batch, self.hidden_size, device=device)

    def _logits(self, obs, hidden, masks, action_masks):
        x = self.encoder(obs)
        if self.recurrent:
            x, hidden = self.rnn(x, hidden, masks)
        phys_logits = self.phys_head(x)
        comm_logits = self.comm_head(x)
        if action_masks is not None:
            phys_mask, comm_mask = action_masks
            if phys_mask is not None:
                phys_logits = phys_logits.masked_fill(~phys_mask, _MASK_FILL)
            if comm_mask is not None:
                comm_logits = comm_logits.masked_fill(~comm_mask, _MASK_FILL)
        return phys_logits, comm_logits, hidden

    def forward(self, obs, hidden, masks, action_masks=None):
        return self._logits(obs, hidden, masks, action_masks)

    def act(self, obs, hidden, masks, action_masks=None, deterministic: bool = False):
        phys_logits, comm_logits, hidden = self._logits(obs, hidden, masks, action_masks)
        phys_dist, comm_dist = Categorical(logits=phys_logits), Categorical(logits=comm_logits)
        if deterministic:
            phys_a = phys_logits.argmax(-1)
            comm_a = comm_logits.argmax(-1)
        else:
            phys_a = phys_dist.sample()
            comm_a = comm_dist.sample()
        log_prob = phys_dist.log_prob(phys_a) + comm_dist.log_prob(comm_a)
        actions = torch.stack([phys_a, comm_a], dim=-1)          # (..., 2)
        return actions, log_prob, hidden

    def evaluate_actions(self, obs, hidden, masks, phys_actions, comm_actions, action_masks=None):
        lp, phys_ent, comm_ent, hidden = self.evaluate_actions_with_entropies(
            obs, hidden, masks, phys_actions, comm_actions, action_masks)
        return lp, phys_ent + comm_ent, hidden

    def evaluate_actions_with_entropies(self, obs, hidden, masks, phys_actions, comm_actions, action_masks=None):
        phys_logits, comm_logits, hidden = self._logits(obs, hidden, masks, action_masks)
        phys_dist, comm_dist = Categorical(logits=phys_logits), Categorical(logits=comm_logits)
        log_prob = phys_dist.log_prob(phys_actions) + comm_dist.log_prob(comm_actions)
        return log_prob, phys_dist.entropy(), comm_dist.entropy(), hidden


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            _init_linear(nn.Linear(state_dim, hidden_size)), nn.Tanh(),
            _init_linear(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
        )
        self.rnn = RNNLayer(hidden_size, hidden_size)
        self.value_head = _init_linear(nn.Linear(hidden_size, 1), gain=1.0)

    def zero_hidden(self, batch: int, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        return torch.zeros(batch, self.hidden_size, device=device)

    def forward(self, state, hidden, masks):
        x = self.encoder(state)
        x, hidden = self.rnn(x, hidden, masks)
        value = self.value_head(x).squeeze(-1)
        return value, hidden


def build_actors(evo_cfg, obs_sizes: dict, hidden_size: int | None = None) -> dict:
    """One RecurrentActor per agent (heterogeneous obs / physical-action dims)."""
    hs = hidden_size or evo_cfg.rmappo.hidden_size
    n_phys = {
        "bs_agent": power_action_count(evo_cfg.actions),
        "ris_agent": ris_action_count(evo_cfg.actions),
        "ma_agent": ma_action_count(evo_cfg.actions),
    }
    return {
        a: RecurrentActor(obs_sizes[a], n_phys[a], N_COMM_ACTIONS, hs, recurrent=evo_cfg.rmappo.recurrent_actor)
        for a in ("bs_agent", "ris_agent", "ma_agent")
    }


def build_critic(evo_cfg, state_dim: int, hidden_size: int | None = None) -> CentralizedCritic:
    return CentralizedCritic(state_dim, hidden_size or evo_cfg.rmappo.hidden_size)

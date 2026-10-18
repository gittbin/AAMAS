"""Tests for the recurrent rollout buffer + GAE (Task 14)."""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from momgrap.marl.algorithms.rmappo.buffer import (
    AGENTS,
    RecurrentRolloutBuffer,
    compute_gae,
)

OBS_SIZES = {"bs_agent": 5, "ris_agent": 6, "ma_agent": 7}
N_PHYS = {"bs_agent": 7, "ris_agent": 11, "ma_agent": 9}
STATE_DIM = 12
HIDDEN = 8


# ---------------------------------------------------------------- GAE
def test_gae_hand_computed():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    bootstrap = torch.tensor([0.0])
    term = torch.zeros(3, 1, dtype=torch.bool)
    trunc = torch.zeros(3, 1, dtype=torch.bool)
    adv, ret = compute_gae(rewards, values, bootstrap, term, trunc, gamma=1.0, lam=1.0)
    assert torch.allclose(adv.squeeze(-1), torch.tensor([2.5, 1.5, 0.5]))
    assert torch.allclose(ret.squeeze(-1), torch.tensor([3.0, 2.0, 1.0]))


def test_gae_termination_stops_bootstrap_and_propagation():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    bootstrap = torch.tensor([0.0])
    term = torch.tensor([[False], [True], [False]])
    trunc = torch.zeros(3, 1, dtype=torch.bool)
    adv, _ = compute_gae(rewards, values, bootstrap, term, trunc, gamma=1.0, lam=1.0)
    # t=2: delta=1-0.5=0.5 -> A2=0.5
    # t=1: terminated -> no bootstrap, no propagation: delta=1-0.5=0.5 -> A1=0.5
    # t=0: next_value=V1=0.5 -> delta=1.0; notdone_0=1 -> A0=1.0+A1=1.5
    assert torch.allclose(adv.squeeze(-1), torch.tensor([1.5, 0.5, 0.5]))


def test_gae_truncation_uses_terminal_value():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    bootstrap = torch.tensor([0.0])
    term = torch.zeros(3, 1, dtype=torch.bool)
    trunc = torch.tensor([[False], [True], [False]])
    terminal_values = torch.tensor([[0.0], [2.0], [0.0]])   # V(terminal) at t=1
    adv, _ = compute_gae(rewards, values, bootstrap, term, trunc, gamma=1.0, lam=1.0,
                         terminal_values=terminal_values)
    # t=1: bootstrap with terminal_value 2.0 -> delta=1+2-0.5=2.5; truncated -> no propagation -> A1=2.5
    # t=0: next=V1=0.5 -> delta=1.0; notdone_0=1 -> A0=1.0+2.5=3.5
    assert torch.allclose(adv.squeeze(-1), torch.tensor([3.5, 2.5, 0.5]))


def test_gae_multi_env_shape():
    T, N = 4, 3
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    bootstrap = torch.randn(N)
    term = torch.zeros(T, N, dtype=torch.bool)
    trunc = torch.zeros(T, N, dtype=torch.bool)
    adv, ret = compute_gae(rewards, values, bootstrap, term, trunc, 0.99, 0.95)
    assert adv.shape == (T, N) and ret.shape == (T, N)
    assert torch.isfinite(adv).all()


# ---------------------------------------------------------------- buffer
def _fill_buffer(T=4, N=2):
    buf = RecurrentRolloutBuffer(T, N, OBS_SIZES, N_PHYS, STATE_DIM, HIDDEN)
    for t in range(T):
        obs = {a: torch.randn(N, OBS_SIZES[a]) for a in AGENTS}
        actor_hidden = {a: torch.randn(N, HIDDEN) for a in AGENTS}
        phys = {a: torch.randint(0, N_PHYS[a], (N,)) for a in AGENTS}
        comm = {a: torch.randint(0, 3, (N,)) for a in AGENTS}
        logp = {a: torch.randn(N) for a in AGENTS}
        buf.insert(
            obs=obs, state=torch.randn(N, STATE_DIM), phys_actions=phys, comm_actions=comm,
            log_probs=logp, values=torch.randn(N), rewards=torch.randn(N),
            terminated=torch.zeros(N, dtype=torch.bool), truncated=torch.zeros(N, dtype=torch.bool),
            masks=torch.ones(N, 1), actor_hidden=actor_hidden, critic_hidden=torch.randn(N, HIDDEN),
        )
    return buf


def test_buffer_fills_and_shapes():
    T, N = 4, 2
    buf = _fill_buffer(T, N)
    assert buf.step == T
    for a in AGENTS:
        assert buf.obs[a].shape == (T, N, OBS_SIZES[a])
        assert buf.phys_actions[a].shape == (T, N)
    assert buf.state.shape == (T, N, STATE_DIM)
    buf.compute_returns(torch.zeros(N), 0.99, 0.95)
    assert buf.advantages.shape == (T, N) and buf.returns.shape == (T, N)


def test_minibatch_shapes_and_hidden_preserved():
    T, N, L = 4, 2, 2
    buf = _fill_buffer(T, N)
    buf.compute_returns(torch.zeros(N), 0.99, 0.95)
    gen = torch.Generator().manual_seed(0)
    batches = list(buf.recurrent_minibatches(num_minibatches=2, seq_len=L, generator=gen))
    total_chunks = (T // L) * N
    assert sum(b["batch_envs"] for b in batches) == total_chunks
    for b in batches:
        mb = b["batch_envs"]
        assert b["obs"]["bs_agent"].shape == (L * mb, OBS_SIZES["bs_agent"])
        assert b["phys_actions"]["bs_agent"].shape == (L * mb,)
        assert b["actor_hidden"]["bs_agent"].shape == (mb, HIDDEN)
        assert b["critic_hidden"].shape == (mb, HIDDEN)
        assert b["state"].shape == (L * mb, STATE_DIM)
        assert b["masks"].shape == (L * mb, 1)
        assert b["valid"].shape == (L * mb,)


def test_minibatch_hidden_matches_chunk_start():
    T, N, L = 4, 2, 2
    buf = _fill_buffer(T, N)
    buf.compute_returns(torch.zeros(N), 0.99, 0.95)
    # single minibatch => contains all chunks; verify each chunk's hidden0 equals stored value
    gen = torch.Generator().manual_seed(1)
    (batch,) = list(buf.recurrent_minibatches(num_minibatches=1, seq_len=L, generator=gen))
    # rebuild the chunk order from the same permutation is opaque; instead verify hidden0 is
    # drawn from the stored actor_hidden at *some* chunk start for every batch entry
    stored = buf.actor_hidden["bs_agent"]        # (T,N,H)
    starts = [0, 2]                              # chunk starts for L=2, T=4
    for row in batch["actor_hidden"]["bs_agent"]:
        matches = any(torch.equal(row, stored[s, n]) for s in starts for n in range(N))
        assert matches


def test_minibatch_padding_when_not_divisible():
    T, N, L = 5, 1, 2                            # 5 not divisible by 2 -> last chunk padded
    buf = _fill_buffer(T, N)
    buf.compute_returns(torch.zeros(N), 0.99, 0.95)
    gen = torch.Generator().manual_seed(2)
    batches = list(buf.recurrent_minibatches(num_minibatches=1, seq_len=L, generator=gen))
    valid_total = sum(int(b["valid"].sum()) for b in batches)
    assert valid_total == T * N                  # exactly T*N valid, rest padded
    # there is at least one padded (invalid) entry
    assert sum(b["valid"].numel() for b in batches) > valid_total


def test_no_timestep_shuffle_within_chunk():
    # a chunk must contain consecutive timesteps in order for a single env
    T, N, L = 4, 1, 4                            # one chunk covering the whole episode
    buf = RecurrentRolloutBuffer(T, N, OBS_SIZES, N_PHYS, STATE_DIM, HIDDEN)
    for t in range(T):
        obs = {a: torch.full((N, OBS_SIZES[a]), float(t)) for a in AGENTS}
        buf.insert(
            obs=obs, state=torch.full((N, STATE_DIM), float(t)),
            phys_actions={a: torch.zeros(N, dtype=torch.long) for a in AGENTS},
            comm_actions={a: torch.zeros(N, dtype=torch.long) for a in AGENTS},
            log_probs={a: torch.zeros(N) for a in AGENTS},
            values=torch.zeros(N), rewards=torch.zeros(N),
            terminated=torch.zeros(N, dtype=torch.bool), truncated=torch.zeros(N, dtype=torch.bool),
            masks=torch.ones(N, 1),
            actor_hidden={a: torch.zeros(N, HIDDEN) for a in AGENTS},
            critic_hidden=torch.zeros(N, HIDDEN),
        )
    buf.compute_returns(torch.zeros(N), 0.99, 0.95)
    (batch,) = list(buf.recurrent_minibatches(num_minibatches=1, seq_len=L))
    # obs flattened time-major (L*mb, dim); values should be 0,1,2,3 in order
    col = batch["obs"]["bs_agent"][:, 0]
    assert torch.equal(col, torch.tensor([0.0, 1.0, 2.0, 3.0]))

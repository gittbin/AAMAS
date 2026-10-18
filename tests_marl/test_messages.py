"""Tests for communication graph + messages + budgeted routing (Task 08)."""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch

from momgrap.channels import bs_cell_centers, sample_environment
from momgrap.genotype import decode
from momgrap.marl.actions.codebooks import build_ris_group_map
from momgrap.marl.communication.budget import DeliveryLog, MessageRouter, SendRequest
from momgrap.marl.communication.graph import (
    AGENT_NAMES,
    action_is_valid,
    build_adjacency,
    destination_from_action,
    valid_action_mask,
    validate_adjacency,
)
from momgrap.marl.communication.messages import (
    MessageContext,
    build_agent_messages,
    message_priority,
)
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.physics.components import build_comm_channel_components
from momgrap.marl.physics.csi import estimate_channel_components

SMOKE = _REPO_ROOT / "configs_marl" / "smoke.yaml"


def _msg(dim: int, val: float = 1.0) -> torch.Tensor:
    return torch.full((dim,), val)


# ---------------------------------------------------------------- graph presets
def test_adjacency_presets_valid():
    for gt in ("none", "full", "ring", "star_bs"):
        A = build_adjacency(gt)
        validate_adjacency(A)
        assert A.shape == (3, 3) and A.dtype == torch.bool
        assert not bool(A.diagonal().any())
    assert build_adjacency("none").sum() == 0
    assert build_adjacency("full").sum() == 6
    ring = build_adjacency("ring")
    assert ring[0, 1] and ring[1, 2] and ring[2, 0] and ring.sum() == 3
    star = build_adjacency("star_bs")
    assert star[0, 1] and star[0, 2] and star[1, 0] and star[2, 0]
    assert not star[1, 2] and not star[2, 1] and star.sum() == 4


def test_adjacency_custom_and_invalid():
    A = build_adjacency("custom", custom=[[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert A.dtype == torch.bool and A[0, 1] and A[2, 0]
    with pytest.raises(ValueError):
        build_adjacency("custom", custom=[[1, 0, 0], [0, 0, 0], [0, 0, 0]])  # self-loop
    with pytest.raises(ValueError):
        build_adjacency("mesh")


# ---------------------------------------------------------------- actions
def test_no_self_message_mapping():
    for i in range(3):
        assert destination_from_action(i, 0) is None       # no-send
        assert destination_from_action(i, 1) != i
        assert destination_from_action(i, 2) != i
    # exact others order (ascending, excluding self)
    assert destination_from_action(0, 1) == 1 and destination_from_action(0, 2) == 2
    assert destination_from_action(1, 1) == 0 and destination_from_action(1, 2) == 2
    assert destination_from_action(2, 1) == 0 and destination_from_action(2, 2) == 1
    with pytest.raises(ValueError):
        destination_from_action(0, 3)


def test_valid_action_mask_no_send_always_valid():
    star = build_adjacency("star_bs")
    for i in range(3):
        mask = valid_action_mask(i, star)
        assert bool(mask[0]) is True                        # no-send always valid
    # ris(1)->ma(2) forbidden under star_bs: action 2 for agent 1 invalid
    assert not action_is_valid(1, 2, star)                  # ris -> ma (its 2nd other)
    assert action_is_valid(1, 1, star)                      # ris -> bs allowed


# ---------------------------------------------------------------- routing
def test_no_self_message_rejected_by_graph():
    router = MessageRouter(build_adjacency("full"), max_messages_per_step=4, delivery_delay_steps=1)
    log = router.submit(0, [SendRequest(sender=0, dest=0, message=_msg(8))])  # self
    assert log.requested == 1 and log.graph_rejected == 1 and log.delivered == 0


def test_graph_mask_rejects_forbidden_edge():
    router = MessageRouter(build_adjacency("star_bs"), max_messages_per_step=4, delivery_delay_steps=1)
    log = router.submit(0, [SendRequest(sender=1, dest=2, message=_msg(8))])  # ris->ma forbidden
    assert log.graph_rejected == 1 and log.delivered == 0


def test_budget_limits_delivery():
    router = MessageRouter(build_adjacency("full"), max_messages_per_step=1, delivery_delay_steps=1)
    reqs = [
        SendRequest(sender=0, dest=1, message=_msg(8), priority=0.9),
        SendRequest(sender=1, dest=2, message=_msg(8), priority=0.5),
        SendRequest(sender=2, dest=0, message=_msg(8), priority=0.1),
    ]
    log = router.submit(0, reqs)
    assert log.requested == 3 and log.graph_rejected == 0
    assert log.delivered == 1 and log.budget_dropped == 2


def test_deterministic_tie_break_by_sender():
    router = MessageRouter(build_adjacency("full"), max_messages_per_step=1, delivery_delay_steps=1)
    reqs = [
        SendRequest(sender=2, dest=0, message=_msg(8, 2.0), priority=0.5),
        SendRequest(sender=0, dest=1, message=_msg(8, 0.0), priority=0.5),  # equal priority
    ]
    router.submit(0, reqs)
    got = router.messages_for_step(1)
    # lower sender index (0 -> dest 1) wins the tie
    assert 1 in got and 0 not in got
    assert torch.equal(got[1][0], _msg(8, 0.0))


def test_no_send_only():
    router = MessageRouter(build_adjacency("full"), max_messages_per_step=1, delivery_delay_steps=1)
    log = router.submit(0, [SendRequest(sender=0, dest=None, message=_msg(8))])
    assert log.requested == 0 and log.delivered == 0


def test_delivery_delay_and_recipient_only():
    router = MessageRouter(build_adjacency("full"), max_messages_per_step=2, delivery_delay_steps=1)
    router.submit(0, [SendRequest(sender=0, dest=1, message=_msg(8, 3.0), priority=1.0)])
    assert router.messages_for_step(0) == {}                # not available same step
    got = router.messages_for_step(1)                       # available next step
    assert set(got.keys()) == {1}                           # only recipient ris(1)
    assert torch.equal(got[1][0], _msg(8, 3.0))


def test_router_rejects_zero_delay():
    with pytest.raises(ValueError):
        MessageRouter(build_adjacency("full"), max_messages_per_step=1, delivery_delay_steps=0)


# ---------------------------------------------------------------- messages
def _context():
    evo = load_evocomm_config(SMOKE)
    cfg, acfg = evo.physical, evo.actions
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    geno = torch.rand(1, cfg.D, generator=g, device="cpu", dtype=cfg.dtype_real)
    P, theta = decode(geno, cfg)
    env = sample_environment(cfg.regimes[0], 1, g, cfg)
    true = build_comm_channel_components(P, theta, env, cfg)
    est = estimate_channel_components(true, 0.1, g)
    centers = bs_cell_centers(cfg).to(dtype=cfg.dtype_real)
    return evo, cfg, acfg, MessageContext(
        estimated=est, theta=theta, positions=P, centers=centers,
        tau=0.1, power_action_index=2, power_codebook_size=acfg.power_codebook_size,
        outage_count=1, K=cfg.K, d_min=cfg.d_min,
        half_extent=cfg.cell_extent_factor * cfg.wavelength / 2.0,
        ma_update_countdown=2, ma_update_period=acfg.ma_update_period,
        ris_last_changed_groups=1, ris_group_count=acfg.ris_group_count, ris_last_sign=1.0,
    )


def test_message_vectors_finite_and_dim():
    evo, cfg, acfg, ctx = _context()
    dim = evo.communication.message_dim
    msgs = build_agent_messages(ctx, dim)
    assert set(msgs.keys()) == set(AGENT_NAMES)
    for name, vec in msgs.items():
        assert vec.shape == (dim,), (name, vec.shape)
        assert torch.isfinite(vec).all(), name


def test_message_deterministic():
    evo, cfg, acfg, ctx = _context()
    dim = evo.communication.message_dim
    a = build_agent_messages(ctx, dim)
    b = build_agent_messages(ctx, dim)
    for name in AGENT_NAMES:
        assert torch.equal(a[name], b[name])


def test_message_priority_orders_by_urgency():
    assert message_priority(0.2, 0.0) < message_priority(0.2, 1.0)
    assert message_priority(0.1, 0.0) < message_priority(0.3, 0.0)

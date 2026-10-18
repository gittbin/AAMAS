"""Information boundaries, unchanged execution, fair ties and checkpoint contracts."""
import csv
from dataclasses import replace
import numpy as np
import pytest
import torch

from experiments_marl.paper_v3 import ImprovementEnv, load_experiment_policy
from experiments_marl_v4.messages import (MessageEnv, MessageTrainer, RotatingRouter,
    make_config, save_policy, load_policy, CONDITIONS)
from momgrap.marl.communication.budget import SendRequest
from scripts_marl.diagnose_paper_messages import ObservedPolicy
from momgrap.marl.policies.baselines import TrainedActorPolicy


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny(condition="comm_reference"):
    cfg = make_config(condition, 98, 16)
    return replace(cfg, rmappo=replace(cfg.rmappo, num_envs=2, rollout_length=8,
                   sequence_length=4, hidden_size=16, update_epochs=1, num_minibatches=2))


def actions(ris=1):
    return {"bs_agent": np.array([1, 1]), "ris_agent": np.array([ris, 1]), "ma_agent": np.array([1, 1])}


def test_reference_env_matches_frozen_v3_and_all_branches_preserve_physics():
    cfg = tiny()
    original = ImprovementEnv(cfg)
    branches = [MessageEnv(cfg, condition) for condition in CONDITIONS if condition != "no_comm"]
    original_obs, _ = original.reset(seed=3000001)
    for env in branches:
        obs, _ = env.reset(seed=3000001)
        for agent in obs:
            np.testing.assert_array_equal(obs[agent], original_obs[agent])
    for step in range(7):
        chosen = actions(17 if step == 3 else 1)
        expected, reward, _, _, info = original.step(chosen)
        for env in branches:
            obs, actual_reward, _, _, actual_info = env.step(chosen)
            assert actual_reward == reward
            np.testing.assert_equal(actual_info, info)
            np.testing.assert_array_equal(env.state(), original.state())
            assert torch.equal(env._gen.get_state(), original._gen.get_state())
            if env.condition == "comm_reference":
                for agent in obs:
                    np.testing.assert_array_equal(obs[agent], expected[agent])


def test_payload_is_pre_action_delayed_finite_and_encodes_previous_ris_groups():
    cfg = tiny("comm_payload")
    a, b = MessageEnv(cfg, "comm_payload"), MessageEnv(cfg, "comm_payload")
    for env in (a, b):
        env.reset(seed=3000002)
    # Same decision information, different current physical actions: outgoing
    # semantic payload must agree and must not use post-action simulator values.
    x = {"bs_agent": np.array([0, 0]), "ris_agent": np.array([1, 1]), "ma_agent": np.array([0, 0])}
    y = {"bs_agent": np.array([8, 0]), "ris_agent": np.array([18, 1]), "ma_agent": np.array([8, 0])}
    a.step(x)
    b.step(y)
    first = a._last_obs_inputs["delivered"][0][0].payload
    second = b._last_obs_inputs["delivered"][0][0].payload
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first[4] == 0 and torch.isfinite(first).all() and len(first) == 12
    a.step(x)
    assert a._last_obs_inputs["delivered"][0][0].payload[4] == 1 / cfg.actions.ris_group_count
    b.step(y)
    assert b._last_obs_inputs["delivered"][0][0].payload[4] == -1
    assert a.current_estimate().step_index < a.step_count


def test_rotating_router_preserves_priority_delay_accounting_and_identity():
    adj = ~torch.eye(3, dtype=torch.bool)
    router = RotatingRouter(adj, 1, 1)
    requests = [SendRequest(i, (i + 1) % 3, torch.tensor([float(i)]), 0.1) for i in range(3)]
    for step in range(6):
        log = router.submit(step, requests)
        assert log.requested == log.eligible == 3 and log.delivered == 1 and log.budget_dropped == 2
        assert not router.deliveries_for_step(step)
        deliveries = router.deliveries_for_step(step + 1)
        message = next(iter(deliveries.values()))[0]
        assert message.sender == step % 3 and message.payload.item() == step % 3
    requests[2].priority = 10
    router.submit(6, requests)
    assert next(iter(router.deliveries_for_step(7).values()))[0].sender == 2
    router.reset()
    assert not router._pending


def test_checkpoint_roundtrip_rejects_v3_and_wrong_condition(tmp_path):
    cfg = tiny("comm_payload")
    trainer = MessageTrainer(cfg, lambda i: MessageEnv(cfg, "comm_payload"), "comm_payload", seed=98)
    trainer.collect_rollout()
    trainer.update()
    path = tmp_path / "payload.pt"
    save_policy(path, trainer)
    _, policy, ck = load_policy(path, "comm_payload")
    assert ck["observation_schema_version"] == 4 and ck["runtime_restorable"] is False
    for agent in policy.actors:
        for key, value in trainer.actors[agent].state_dict().items():
            assert torch.equal(value, policy.actors[agent].state_dict()[key])
    with pytest.raises(ValueError):
        load_experiment_policy(path)
    with pytest.raises(ValueError):
        load_policy(path, "comm_rotating_router")
    ck["experiment_settings"]["payload"] = "original"
    torch.save(ck, path)
    with pytest.raises(ValueError, match="contract"):
        load_policy(path)


def test_observer_is_passive_for_actions_hidden_state_and_physical_rng(tmp_path):
    cfg = tiny()
    trainer = MessageTrainer(cfg, lambda i: MessageEnv(cfg), "comm_reference", seed=98)
    original = TrainedActorPolicy(trainer.actors, trainer.obs_rms)
    copy = TrainedActorPolicy(trainer.actors, trainer.obs_rms)
    observed = ObservedPolicy(copy, cfg, tmp_path / "inbox.csv")
    env = MessageEnv(cfg, eval_mode=True)
    obs, _ = env.reset(seed=3000003)
    original.reset()
    observed.reset()
    for _ in range(4):
        rng = env._gen.get_state().clone()
        expected = original.act(obs, env)
        actual = observed.act(obs, env)
        assert torch.equal(rng, env._gen.get_state())
        for agent in expected:
            np.testing.assert_array_equal(expected[agent], actual[agent])
            assert torch.equal(original._hidden[agent], copy._hidden[agent])
        obs, _, _, _, _ = env.step(actual)
    observed.close()
    with (tmp_path / "inbox.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 12


def test_reference_trainer_update_exactly_matches_frozen_improvement_trainer():
    from experiments_marl.paper_v3 import ImprovementTrainer
    cfg = tiny()
    frozen = ImprovementTrainer(cfg, lambda i: ImprovementEnv(cfg), "comm_reference", seed=98)
    new = MessageTrainer(cfg, lambda i: MessageEnv(cfg), "comm_reference", seed=98)
    rng = torch.get_rng_state().clone()
    frozen.collect_rollout()
    torch.set_rng_state(rng)
    new.collect_rollout()
    frozen.update()
    new.update()
    for a in frozen.actors:
        for key, value in frozen.actors[a].state_dict().items():
            assert torch.equal(value, new.actors[a].state_dict()[key])
    for key, value in frozen.critic.state_dict().items():
        assert torch.equal(value, new.critic.state_dict()[key])

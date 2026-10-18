"""Cost-only and asymmetric-information controls using the actual delayed router."""
import numpy as np
import torch
from gymnasium.spaces import MultiDiscrete

from ..actions.codebooks import ma_action_count, power_action_count, ris_action_count
from ..communication.budget import MessageRouter, SendRequest
from ..communication.graph import AGENT_NAMES, build_adjacency, destination_from_action
from ..observations.builder import observation_sizes, centralized_state_size
from ..observations.features import inbox_features


class CommunicationDiagnosticEnv:
    def __init__(self, evo, task="positive", price=0.0):
        if task not in ("positive", "cost_only"):
            raise ValueError(task)
        self.evo, self.cfg = evo, evo.physical
        self.device = torch.device("cpu")
        self.possible_agents = list(AGENT_NAMES)
        self.task, self.price = task, price
        self.adjacency = build_adjacency("full" if task == "cost_only" else "custom",
                                         custom=[[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        self.router = MessageRouter(self.adjacency, evo.communication.max_messages_per_step, 1)
        self.counts = dict(zip(AGENT_NAMES, (power_action_count(evo.actions),
                                          ris_action_count(evo.actions), ma_action_count(evo.actions))))
        self.sizes = observation_sizes(evo)
        self.agents = []
        self.step_count = 0
        self.inbox = {}

    def action_space(self, agent):
        return MultiDiscrete([self.counts[agent], 3])

    def physical_action_mask(self, agent):
        mask = torch.zeros(self.counts[agent], dtype=torch.bool)
        mask[0] = True
        if self.task == "positive" and agent == "ris_agent" and self.step_count == 1:
            mask[1] = True
        return mask

    def state(self):
        state = np.zeros(centralized_state_size(self.evo), dtype=np.float32)
        state[0:2] = [self.bit, self.step_count]
        return state

    def _observations(self):
        observations = {a: np.zeros(self.sizes[a], dtype=np.float32) for a in AGENT_NAMES}
        for index, agent in enumerate(AGENT_NAMES):
            observations[agent][1] = self.step_count / self.length
            inbox = inbox_features(self.inbox.get(index, []), self.evo.communication.max_messages_per_step,
                                   self.evo.communication.message_dim, "cpu", torch.float32)
            observations[agent][-len(inbox) - 2:-2] = inbox.numpy()
        # Only the sender sees the bit, and only before the receiver chooses.
        if self.task == "positive" and self.step_count == 0:
            observations["bs_agent"][0] = self.bit
        return observations

    def reset(self, seed=None, options=None):
        self.bit = int(np.random.default_rng(seed).integers(2)) if self.task == "positive" else 0
        self.length = 2 if self.task == "positive" else 8
        self.step_count = 0
        self.agents = list(AGENT_NAMES)
        self.router.reset()
        self.inbox = {}
        return self._observations(), {a: {} for a in self.agents}

    def step(self, actions):
        requests = [SendRequest(i, destination_from_action(i, int(actions[a][1])),
                                torch.zeros(self.evo.communication.message_dim))
                    for i, a in enumerate(AGENT_NAMES)]
        log = self.router.submit(self.step_count, requests)
        accuracy = (float(int(actions["ris_agent"][0]) == self.bit)
                    if self.task == "positive" and self.step_count == 1 else None)
        reward = (0.0 if accuracy is None else accuracy) - self.price * log.delivered
        self.step_count += 1
        self.inbox = self.router.deliveries_for_step(self.step_count)
        terminated = self.step_count >= self.length
        info = {"message_requested": log.requested, "message_delivered": log.delivered,
                "message_eligible": log.eligible, "message_dropped": log.budget_dropped,
                "accuracy": accuracy, "private_bit": self.bit}
        observations = self._observations()
        rewards = {a: reward for a in self.agents}
        terms = {a: terminated for a in self.agents}
        truncs = {a: False for a in self.agents}
        infos = {a: info for a in self.agents}
        if terminated:
            self.agents = []
        return observations, rewards, terms, truncs, infos

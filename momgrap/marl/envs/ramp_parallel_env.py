"""PettingZoo ParallelEnv wrapping the RAMP one-interval physics + MARL machinery.

Three heterogeneous agents (bs / ris / ma) act each control interval under imperfect,
delayed CSI; the low-level BD precoder uses the current estimate while realized rates
are computed on the true channel; selective inter-agent communication is graph- and
budget-limited with a one-step delivery delay. Reward is a single cooperative scalar
shared by all agents.

Actors see only delayed *estimated* CSI (via :func:`build_actor_observations`, which
has no true-channel input); the true channel appears only in the centralized critic
:meth:`state`. See the step ordering in :meth:`step`.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch
from gymnasium.spaces import Box, MultiDiscrete
from pettingzoo import ParallelEnv

from ...channels import bs_cell_centers, build_comm_channel_components
from ...comm import bd_precoding, evm_sinr_rate
from ..actions.codebooks import (
    build_ris_group_map,
    ma_action_count,
    power_action_count,
    ris_action_count,
)
from ..actions.decoder import decode_ma_action, decode_ris_action
from ..actions.safety import project_to_safety
from ..communication.budget import MessageRouter, SendRequest, ReceivedMessage
from ..communication.graph import (
    AGENT_NAMES,
    build_adjacency,
    destination_from_action,
    validate_adjacency,
)
from ..communication.messages import MessageContext, build_agent_messages, message_priority
from ..communication.modes import COMM_MODES, forced_comm_action
from ..normalization import FeatureProjection
from ..observations.builder import (
    COMPLEX_PROJ_DIM,
    build_actor_observations,
    build_centralized_state,
    centralized_state_size,
    observation_sizes,
)
from ..physics.backend import evaluate_control_interval
from ..physics.csi import CSIHistoryBuffer, estimate_channel_components
from ..physics.dynamics import DynamicChannelProcess
from ..types import DynamicPhysicalState
from .reward import ObjectiveVector, compute_reward


class MOMGRAPParallelEnv(ParallelEnv):
    metadata = {"render_modes": [], "name": "momgrap_ramp_v0", "is_parallelizable": True}

    possible_agents = list(AGENT_NAMES)

    def __init__(
        self,
        evo_cfg,
        seed: int = 0,
        eval_mode: bool = False,
        init_mode: str = "uspa_zero_phase",
        custom_adjacency=None,
        projection_seed: int = 0,
        ramp_init=None,
        comm_mode: str = "learned",
    ):
        self.evo = evo_cfg
        self.cfg = evo_cfg.physical
        self.device = self.cfg.device
        self.possible_agents = list(AGENT_NAMES)
        self.agents: list[str] = []
        self.eval_mode = eval_mode
        self.init_mode = init_mode
        self._ramp_init = ramp_init
        self._seed = seed
        self._tau_override: float | None = None
        if comm_mode not in COMM_MODES:
            raise ValueError(f"unknown comm_mode {comm_mode!r}; expected {COMM_MODES}")
        self._comm_mode = comm_mode
        self._message_cost_override: float | None = None
        self._message_corruption: str = "none"   # none/zero/random/permute (message-ignored sanity)
        self._half = self.cfg.cell_extent_factor * self.cfg.wavelength / 2.0

        # static geometry / helpers
        rdtype = self.cfg.dtype_real
        self.centers = bs_cell_centers(self.cfg).to(device=self.device, dtype=rdtype)
        self.group_map = build_ris_group_map(self.cfg, evo_cfg.actions, device=self.device)
        self.phase_step = math.radians(evo_cfg.actions.ris_phase_step_deg)
        self.projection = FeatureProjection(
            2 * self.cfg.K * self.cfg.M, COMPLEX_PROJ_DIM, seed=projection_seed,
            device=self.device, dtype=rdtype,
        )

        # communication
        self.adjacency = build_adjacency(
            evo_cfg.communication.graph_type, custom=custom_adjacency, device=self.device
        )
        self.router = MessageRouter(
            self.adjacency,
            evo_cfg.communication.max_messages_per_step,
            evo_cfg.communication.delivery_delay_steps,
        )

        # dynamics + CSI history
        self.process = DynamicChannelProcess(self.cfg, evo_cfg.dynamics)
        self.csi = CSIHistoryBuffer()

        # fixed spaces
        self._obs_sizes = observation_sizes(evo_cfg, proj_dim=COMPLEX_PROJ_DIM)
        self._state_dim = centralized_state_size(evo_cfg, proj_dim=COMPLEX_PROJ_DIM)
        self._action_spaces = {
            "bs_agent": MultiDiscrete([power_action_count(evo_cfg.actions), 3]),
            "ris_agent": MultiDiscrete([ris_action_count(evo_cfg.actions), 3]),
            "ma_agent": MultiDiscrete([ma_action_count(evo_cfg.actions), 3]),
        }
        self._observation_spaces = {
            a: Box(-np.inf, np.inf, shape=(self._obs_sizes[a],), dtype=np.float32)
            for a in self.possible_agents
        }

        # runtime state
        self._pstate: DynamicPhysicalState | None = None
        self.step_count = 0
        self._gen: torch.Generator | None = None
        self._current_obs: dict | None = None
        self._prev_actions: dict = {}
        self._critic: dict | None = None
        self._last_delayed_est = None
        self._last_obs_inputs: dict | None = None
        self._frozen = False
        self._frozen_power_action = 0
        self._message_pricing = "delivered"
        self._csi_feature_mode = evo_cfg.csi.feature_mode
        self._csi_reference_features = None

    # ------------------------------------------------------------------ spaces
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def state_size(self) -> int:
        return self._state_dim

    def current_estimate(self):
        """The delayed estimated channel the actors currently see (never the truth)."""
        return self._last_delayed_est

    def centralized_observation(self) -> np.ndarray:
        """Union of decision-time actor observations, without privileged state."""
        return np.concatenate([self._current_obs[a].detach().cpu().numpy()
                               for a in self.possible_agents]).astype(np.float32)

    def decision_configuration(self):
        """Observable actuator geometry; exposes no simulator channel state."""
        return self._last_obs_inputs["positions"].clone(), self._last_obs_inputs["theta"].clone()

    def physical_action_mask(self, agent) -> torch.Tensor:
        mask = torch.ones(int(self.action_space(agent).nvec[0]), dtype=torch.bool, device=self.device)
        if self._frozen or (agent == "ma_agent" and self.step_count % self.evo.actions.ma_update_period):
            mask[:] = False
            mask[self._frozen_power_action if agent == "bs_agent" and self._frozen else 0] = True
        return mask

    def freeze_configuration(self) -> None:
        """Keep the current layout, phases and power-profile rule; BD stays fresh."""
        self._frozen = True
        self._frozen_power_action = self._prev_actions.get("bs_agent", 0)

    def set_message_pricing(self, mode: str) -> None:
        if mode not in ("delivered", "requested"):
            raise ValueError("message pricing must be delivered or requested")
        self._message_pricing = mode

    def set_csi_features(self, mode, reference=None):
        if mode not in ("normal", "masked", "fixed", "shuffled"):
            raise ValueError(mode)
        if mode in ("fixed", "shuffled") and reference is None:
            raise ValueError("fixed/shuffled CSI diagnostics need development-set features")
        self._csi_feature_mode, self._csi_reference_features = mode, reference

    def _apply_csi_features(self, observations):
        mode = self._csi_feature_mode
        if mode == "normal":
            return observations
        lengths = {"bs_agent": 4 + self.cfg.K + COMPLEX_PROJ_DIM,
                   "ris_agent": 5 + self.cfg.K + COMPLEX_PROJ_DIM,
                   "ma_agent": 4 + self.cfg.K + COMPLEX_PROJ_DIM}
        for name, length in lengths.items():
            if mode == "masked":
                observations[name][:length] = 0
            else:
                features = torch.as_tensor(self._csi_reference_features[name], device=self.device,
                                           dtype=self.cfg.dtype_real)
                if mode == "shuffled":
                    index = int(torch.randint(len(features), (1,), generator=self._intervention_gen, device=self.device))
                    features = features[index]
                observations[name][:length] = features
        return observations

    def set_tau(self, tau: float | None) -> None:
        """Override the per-step tau (used by the evaluation grid); None re-enables sampling."""
        self._tau_override = tau

    def set_communication_mode(self, mode: str) -> None:
        """Force the communication mode (none/full/random/learned); eval ablation."""
        if mode not in COMM_MODES:
            raise ValueError(f"unknown comm_mode {mode!r}; expected {COMM_MODES}")
        self._comm_mode = mode

    def set_message_cost(self, cost: float | None) -> None:
        """Override the delivered-message reward cost (curriculum); None restores config."""
        self._message_cost_override = cost

    def set_message_corruption(self, mode: str) -> None:
        """Corrupt delivered messages before they enter observations (message-ignored sanity)."""
        if mode not in ("none", "zero", "random", "permute"):
            raise ValueError(f"unknown message corruption {mode!r}")
        self._message_corruption = mode

    def _corrupt_messages(self, delivered: dict) -> dict:
        mode = self._message_corruption
        if mode in ("none", "permute") or not delivered:
            return delivered
        out = {}
        for dest, msgs in delivered.items():
            new = []
            for record in msgs:
                m = record.payload
                if mode == "zero":
                    payload = torch.zeros_like(m)
                elif mode == "random":
                    payload = torch.randn(m.shape, generator=self._intervention_gen, device=m.device, dtype=m.dtype)
                else:
                    payload = m[torch.randperm(m.shape[-1], generator=self._intervention_gen, device=m.device)]
                new.append(ReceivedMessage(record.sender, payload))
            out[dest] = new
        return out

    def set_adjacency(self, adjacency: torch.Tensor) -> None:
        """Replace the communication graph (used by the EA graph search); rebuilds router."""
        validate_adjacency(adjacency)
        self.adjacency = adjacency.to(self.device)
        self.router = MessageRouter(
            self.adjacency,
            self.evo.communication.max_messages_per_step,
            self.evo.communication.delivery_delay_steps,
        )

    # ------------------------------------------------------------------ helpers
    def _sample_tau(self) -> float:
        if self._tau_override is not None:
            return float(self._tau_override)
        csi = self.evo.csi
        if self.eval_mode:
            return float(csi.tau_eval)
        lo, hi = csi.tau_train_min, csi.tau_train_max
        if hi <= lo:
            return float(lo)
        u = torch.rand(1, generator=self._gen, device=self.device).item()
        return float(lo + u * (hi - lo))

    def _ma_countdown(self) -> int:
        period = self.evo.actions.ma_update_period
        return (period - 1) - (self.step_count % period)

    def _rand_comm(self) -> int:
        return int(torch.randint(0, 3, (1,), generator=self._comm_gen, device=self.device).item())

    @staticmethod
    def _ris_sign(ris_idx: int, G: int) -> float:
        if ris_idx == 0:
            return 0.0
        if ris_idx <= 2 * G:
            return 1.0 if (ris_idx - 1) % 2 == 0 else -1.0
        return 1.0 if ris_idx == 2 * G + 1 else -1.0

    def _initial_config(self):
        rdtype = self.cfg.dtype_real
        if self.init_mode == "uspa_zero_phase":
            positions = self.centers.unsqueeze(0).clone()
            theta = torch.zeros(1, self.cfg.L, device=self.device, dtype=rdtype)
        elif self.init_mode == "random":
            g = torch.Generator(device=self.device)
            g.manual_seed(self._seed + 5)
            from ...genotype import decode
            geno = torch.rand(1, self.cfg.D, generator=g, device=self.device, dtype=rdtype)
            positions, theta = decode(geno, self.cfg)
        elif self.init_mode.startswith("ramp_"):
            if self._ramp_init is None:
                raise ValueError(f"init_mode={self.init_mode!r} requires a ramp_init object")
            from ..physics.initialization import check_compatibility, initial_config_from
            check_compatibility(self._ramp_init, self.cfg)
            g = torch.Generator(device=self.device)
            g.manual_seed(self._seed + 9)
            positions, theta = initial_config_from(self._ramp_init, self.init_mode, g)
            positions = positions.to(device=self.device, dtype=rdtype)
            theta = theta.to(device=self.device, dtype=rdtype)
        else:
            raise ValueError(f"unknown init_mode {self.init_mode!r}")
        return positions, theta

    def _to_numpy(self, obs: dict) -> dict:
        return {a: obs[a].detach().cpu().numpy().astype(np.float32) for a in obs}

    def _validate_actions(self, actions: dict) -> None:
        for a in self.agents:
            if a not in actions:
                raise ValueError(f"missing action for active agent {a!r}")
            arr = np.asarray(actions[a]).reshape(-1)
            if arr.shape[0] != 2:
                raise ValueError(f"{a} action must have 2 components, got {arr.shape[0]}")
            n_phys = int(self._action_spaces[a].nvec[0])
            if not (0 <= int(arr[0]) < n_phys):
                raise ValueError(f"{a} physical action {int(arr[0])} out of range [0,{n_phys})")
            if not (0 <= int(arr[1]) < 3):
                raise ValueError(f"{a} comm action {int(arr[1])} out of range [0,3)")

    def _refresh_critic(self, true_components, positions, theta, prev_actions, tau, regime_index, step):
        self._critic = dict(
            true=true_components, positions=positions, theta=theta,
            prev_actions=prev_actions, tau=tau, regime=regime_index, step=step,
        )

    def state(self) -> np.ndarray:
        """Centralized critic state (uses the true channel; never fed to actors)."""
        c = self._critic
        st = build_centralized_state(
            c["true"], c["positions"], c["theta"], self.centers, self.group_map,
            c["prev_actions"], c["tau"], c["step"], c["regime"], self.evo, self.projection,
        )
        return st.detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = seed
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(self._seed + 10_007)
        self._comm_gen = torch.Generator(device=self.device).manual_seed(self._seed + 20_011)
        self._intervention_gen = torch.Generator(device=self.device).manual_seed(self._seed + 30_013)
        self._frozen = False
        self._frozen_power_action = 0

        if options and "positions" in options and "theta" in options:
            positions = torch.as_tensor(options["positions"], device=self.device, dtype=self.cfg.dtype_real)
            theta = torch.as_tensor(options["theta"], device=self.device, dtype=self.cfg.dtype_real)
        else:
            positions, theta = self._initial_config()

        self._pstate = self.process.reset(self._seed, positions, theta)
        self.router.reset()
        self.step_count = 0
        self.agents = list(self.possible_agents)
        self._prev_actions = {}

        true0 = build_comm_channel_components(
            self._pstate.positions, self._pstate.theta, self._pstate.environment, self.cfg
        )
        tau0 = self._sample_tau()
        est0 = estimate_channel_components(true0, tau0, self._gen, self.evo.csi.noise_scaling,
                                          positions=positions, theta=theta, step_index=0)
        self.csi.reset(est0)

        delayed = self.csi.get_delayed(self.evo.csi.actor_delay_steps)
        self._last_delayed_est = delayed
        countdown = self._ma_countdown()
        self._last_obs_inputs = dict(
            delayed=delayed, theta=self._pstate.theta, positions=self._pstate.positions,
            prev_actions={}, delivered={}, tau=tau0, step=0, countdown=countdown,
        )
        self._current_obs = build_actor_observations(
            delayed, self._pstate.theta, self._pstate.positions, self.centers, self.group_map,
            {}, {}, tau0, 0, self.evo, self.projection, countdown,
        )
        self._refresh_critic(true0, self._pstate.positions, self._pstate.theta, {}, tau0,
                             self._pstate.regime_index, 0)
        self._current_obs = self._apply_csi_features(self._current_obs)

        infos = {a: {} for a in self.agents}
        return self._to_numpy(self._current_obs), infos

    # ------------------------------------------------------------------ step
    def step(self, actions: dict):
        if not self.agents:
            return {}, {}, {}, {}, {}
        self._validate_actions(actions)

        cfg, evo = self.cfg, self.evo
        G = evo.actions.ris_group_count
        prev = self._pstate

        # decode physical actions
        power_idx, bs_comm = int(np.asarray(actions["bs_agent"]).reshape(-1)[0]), int(np.asarray(actions["bs_agent"]).reshape(-1)[1])
        ris_idx, ris_comm = int(np.asarray(actions["ris_agent"]).reshape(-1)[0]), int(np.asarray(actions["ris_agent"]).reshape(-1)[1])
        ma_idx, ma_comm = int(np.asarray(actions["ma_agent"]).reshape(-1)[0]), int(np.asarray(actions["ma_agent"]).reshape(-1)[1])

        if self._frozen:
            power_idx, ris_idx, ma_idx = self._frozen_power_action, 0, 0

        theta_cand, n_changed = decode_ris_action(prev.theta, ris_idx, self.group_map, self.phase_step)

        ma_allowed = (self.step_count % evo.actions.ma_update_period == 0)
        if ma_allowed:
            pos_cand = decode_ma_action(prev.positions, ma_idx, cfg, evo.actions)
            applied_ma_idx = ma_idx
        else:
            pos_cand = prev.positions
            applied_ma_idx = 0

        proj = project_to_safety(pos_cand, theta_cand, prev.power_alloc, cfg,
                                 previous_positions=prev.positions)
        positions_new, theta_new = proj.positions, proj.theta
        viol_before = (proj.boundary_violation_before
                       + max(0.0, cfg.d_min - proj.min_spacing_before)
                       + proj.power_violation_before)
        viol_after = (proj.boundary_violation_after
                      + max(0.0, cfg.d_min - proj.min_spacing_after)
                      + proj.power_violation_after)

        tau = self._sample_tau()
        res = evaluate_control_interval(
            prev.environment, positions_new, theta_new, power_idx, tau, cfg, evo, self._gen
        )
        if not (math.isfinite(res.min_user_rate) and math.isfinite(res.sum_rate)
                and math.isfinite(res.min_sensing_illumination_db)):
            raise RuntimeError(f"non-finite physical result at step {self.step_count}: {res}")

        # Outgoing payloads use decision-time delayed estimates and local
        # pre-action hardware state. No realized outage or fresh CSI enters them.
        decision_est = self.current_estimate()
        belief_W = bd_precoding(decision_est.aggregate, cfg)
        belief_rates = evm_sinr_rate(decision_est.aggregate, belief_W, cfg)
        belief_outage = int((belief_rates < evo.reward.rate_target_bpshz).sum())
        countdown = self._ma_countdown()
        ctx = MessageContext(
            estimated=decision_est, theta=prev.theta, positions=prev.positions,
            centers=self.centers, tau=decision_est.tau, power_action_index=self._prev_actions.get("bs_agent", 0),
            power_codebook_size=power_action_count(evo.actions), outage_count=belief_outage,
            K=cfg.K, d_min=cfg.d_min, half_extent=self._half,
            ma_update_countdown=countdown, ma_update_period=evo.actions.ma_update_period,
            ris_last_changed_groups=0, ris_group_count=G,
            ris_last_sign=self._ris_sign(self._prev_actions.get("ris_agent", 0), G),
        )
        msgs = build_agent_messages(ctx, evo.communication.message_dim)
        if self._message_corruption == "permute":
            # Swap payloads among senders, retaining sender IDs and delivery.
            order = torch.randperm(len(AGENT_NAMES), generator=self._intervention_gen, device=self.device).tolist()
            msgs = {name: msgs[AGENT_NAMES[order[i]]] for i, name in enumerate(AGENT_NAMES)}
        priority = message_priority(decision_est.tau, belief_outage / max(1, cfg.K))
        # apply the (possibly forced) communication mode to the chosen comm actions
        raw_comm = (bs_comm, ris_comm, ma_comm)
        comm_actions = tuple(
            forced_comm_action(i, raw_comm[i], self._comm_mode if evo.communication.enabled else "none", self._rand_comm())
            for i in range(len(AGENT_NAMES))
        )
        requests = []
        for i, name in enumerate(AGENT_NAMES):
            dest = destination_from_action(i, comm_actions[i])
            if dest is not None:
                requests.append(SendRequest(sender=i, dest=dest, message=msgs[name], priority=priority))
        log = self.router.submit(self.step_count, requests)

        ma_disp = (float((positions_new - prev.positions).norm(dim=-1).mean()) / max(self._half, 1e-12)
                   if ma_allowed else 0.0)
        reconfig = ma_disp + n_changed / max(1, G)

        obj = ObjectiveVector(
            min_user_rate=res.min_user_rate, sum_rate=res.sum_rate,
            min_sensing_illumination_db=res.min_sensing_illumination_db,
            outage_count=res.outage_count, delivered_messages=(log.delivered if self._message_pricing == "delivered" else log.requested),
            reconfiguration_cost=reconfig, constraint_violation_before=viol_before,
        )
        reward_cfg = (evo.reward if self._message_cost_override is None
                      else dataclasses.replace(evo.reward, w_message=self._message_cost_override))
        reward, _components = compute_reward(obj, reward_cfg)

        # update physical config + advance dynamics
        prev_actions = {"bs_agent": power_idx, "ris_agent": ris_idx, "ma_agent": applied_ma_idx}
        mid = DynamicPhysicalState(
            step_index=prev.step_index, regime_index=prev.regime_index,
            environment=prev.environment, user_velocity=prev.user_velocity,
            positions=positions_new, theta=theta_new, power_alloc=res.p_user,
            previous_physical_actions=prev_actions,
        )
        next_state = self.process.advance(mid)
        self._pstate = next_state
        self._prev_actions = prev_actions

        self.step_count += 1

        true_next = build_comm_channel_components(
            next_state.positions, next_state.theta, next_state.environment, cfg
        )
        est_next = estimate_channel_components(
            true_next, tau, self._gen, evo.csi.noise_scaling,
            positions=next_state.positions, theta=next_state.theta, step_index=self.step_count,
        )
        self.csi.append(est_next)

        delivered = self._corrupt_messages(self.router.deliveries_for_step(self.step_count))
        delayed = self.csi.get_delayed(evo.csi.actor_delay_steps)
        self._last_delayed_est = delayed
        countdown_next = self._ma_countdown()
        self._last_obs_inputs = dict(
            delayed=delayed, theta=next_state.theta, positions=next_state.positions,
            prev_actions=prev_actions, delivered=delivered, tau=delayed.tau,
            step=self.step_count, countdown=countdown_next,
        )
        self._current_obs = build_actor_observations(
            delayed, next_state.theta, next_state.positions, self.centers, self.group_map,
            prev_actions, delivered, delayed.tau, self.step_count, self.evo, self.projection, countdown_next,
        )

        self._refresh_critic(true_next, next_state.positions, next_state.theta, prev_actions,
                             tau, next_state.regime_index, self.step_count)
        self._current_obs = self._apply_csi_features(self._current_obs)

        truncated = self.step_count >= evo.dynamics.episode_length
        rewards = {a: reward for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        info = self._build_info(res, tau, log, reconfig, viol_before, viol_after,
                                power_idx, ris_idx, applied_ma_idx, obj)
        infos = {a: info for a in self.agents}

        obs_np = self._to_numpy(self._current_obs)
        if truncated:
            self.agents = []
        return obs_np, rewards, terminations, truncations, infos

    def _build_info(self, res, tau, log, reconfig, viol_before, viol_after,
                    power_idx, ris_idx, ma_idx, obj) -> dict:
        return {
            "regime": self.cfg.regimes[self._pstate.regime_index].name,
            "tau": float(tau),
            "min_user_rate": res.min_user_rate,
            "sum_rate": res.sum_rate,
            "min_sensing_illumination_db": res.min_sensing_illumination_db,
            "user_rates": res.user_rates.detach().cpu().numpy(),
            "sensing_per_target": res.sensing_per_target.detach().cpu().numpy(),
            "outage_count": res.outage_count,
            "message_requested": log.requested,
            "message_eligible": log.eligible,
            "message_delivered": log.delivered,
            "message_dropped": log.budget_dropped,
            "graph_rejected": log.graph_rejected,
            "reconfiguration_cost": reconfig,
            "violation_before": viol_before,
            "violation_after": viol_after,
            "power_action": power_idx,
            "ris_action": ris_idx,
            "ma_action": ma_idx,
            "objective_vector": obj.as_dict(),
        }

    def render(self):
        return None

    def close(self):
        return None

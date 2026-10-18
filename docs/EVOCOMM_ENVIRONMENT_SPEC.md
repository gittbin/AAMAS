# EvoComm-MOMGRAP — Environment Specification

`momgrap.marl.envs.ramp_parallel_env.MOMGRAPParallelEnv` is a PettingZoo
`ParallelEnv` wrapping the RAMP one-interval physics for three heterogeneous agents.

## Agents

```
possible_agents = ["bs_agent", "ris_agent", "ma_agent"]   # order fixed everywhere
```

Cooperative, shared scalar reward. Single controlled physical configuration (N=1),
single MC realisation per step (S=1) evolving with temporal correlation.

## Action spaces

```
bs_agent : MultiDiscrete([power_codebook_size, 3])
ris_agent: MultiDiscrete([ris_action_count, 3])   # ris_action_count = 2*G + 3
ma_agent : MultiDiscrete([ma_action_count, 3])     # ma_action_count = 9
```

The second component is the communication destination action: `0 = no-send`,
`1 = first other agent`, `2 = second other agent` (others in ascending index order).

## Step ordering (`step(actions)`)

1. validate active agents / actions;
2. actor observations were produced from the beginning-of-interval state;
3. decode physical actions (RIS grouped phase, MA layout pattern);
4. enforce `ma_update_period` (MA moves only on period boundaries);
5. safety projection (repair MA positions, wrap RIS phases; fallback if infeasible);
6. sample the current CSI error `tau`;
7. evaluate the control interval — true channel → **estimated CSI** → BD precoder from
   the estimate → power action decoded from the estimate → **realized rate on the true
   channel** → sensing illumination from the same precoder;
8. build outgoing messages from the current local state + selected actions;
9. graph mask + budget filtering of message requests;
10. compute the shared scalar reward (raw objective vector → normalised scalar);
11. update the physical configuration; advance user mobility + Gauss–Markov fading;
12. append the current estimated CSI to the delay history;
13. queue delivered messages for the next observation (delivery delay = 1);
14. build next observations; refresh the centralized critic state (true channel);
15. `truncation = True` at `episode_length`; `termination = False` (no natural end).

## Timing model (imperfect / delayed CSI)

- High-level agents observe a **delayed estimate** (`actor_delay_steps`), never truth.
- The low-level BD precoder uses the **current estimate** (`beamformer_uses_current_estimate`).
- Realized rate and reward use the **true channel**.
- A message selected at step *t* is available in the recipient's observation at *t+1*.

## Centralized critic

`env.state()` returns the centralized state (may include true-channel summaries, full
geometry, all previous actions, tau, regime one-hot, step). It is used only by the
critic and is never part of any actor observation.

## Communication modes

`comm_mode ∈ {learned, none, full, random}` forces the comm action at the env level
(evaluation ablation). `set_message_cost` overrides the delivered-message reward cost
(curriculum). `set_adjacency` swaps the graph (EA search).

## Infos (per step)

`regime, tau, min_user_rate, sum_rate, min_sensing_illumination_db, user_rates,
sensing_per_target, outage_count, message_requested/delivered/dropped, graph_rejected,
reconfiguration_cost, violation_before/after, power_action, ris_action, ma_action,
objective_vector`.

Numerical failure raises a detailed error (never silently zeroes the reward).

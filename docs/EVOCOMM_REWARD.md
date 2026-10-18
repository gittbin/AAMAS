# EvoComm-MOMGRAP — Reward

`momgrap.marl.envs.reward`. A single cooperative scalar shared by all agents, an
explicit linear combination of config-normalised terms.

## Raw objective vector (per step)

```
min_user_rate                realized worst-user rate (bps/Hz, true channel)
sum_rate                     realized sum rate
min_sensing_illumination_db  worst-target beampattern illumination (dB)
outage_count                 users below rate_target_bpshz
delivered_messages           messages delivered this step (after graph + budget)
reconfiguration_cost         normalized MA displacement + fraction of RIS groups changed
constraint_violation_before  boundary + spacing + power violation BEFORE repair
```

## Scalarisation

```
r = w_min_rate * (min_user_rate / rate_scale)
  + w_sum_rate * (sum_rate / rate_scale)
  + w_sensing  * ((min_sensing_illumination_db - sensing_offset_db) / sensing_scale_db)
  - w_outage         * outage_count
  - w_message        * delivered_messages
  - w_reconfiguration* reconfiguration_cost
  - w_violation      * constraint_violation_before
```

All normalisation constants (`rate_scale`, `sensing_offset_db`, `sensing_scale_db`,
weights, `rate_target_bpshz`) come from `RewardConfig`. Normalisation is **explicit**
and never derived from statistics of future evaluation episodes.

- The message penalty uses **delivered** messages (the sender pays), discouraging
  excess communication without any arbitrary communication *bonus*.
- The violation penalty uses the **pre-repair** magnitude so agents learn to propose
  feasible actions rather than relying on the safety shield.
- `CommunicationCurriculum` may anneal `w_message` from `start_cost` to `final_cost`
  over `warmup_updates` (passed to `trainer.train(curriculum=...)`).

All agents receive the same scalar: `rewards = {agent: r for agent in active_agents}`.

# EvoComm-MOMGRAP — Observations

Built by `momgrap.marl.observations.builder`. Sizes are **fixed** given the config
(`observation_sizes(evo_cfg)` / `centralized_state_size(evo_cfg)`).

## Actor observations (delayed estimate only)

`build_actor_observations(...)` has **no true-channel parameter** — actors structurally
cannot see the truth. Each agent's vector concatenates:

### bs_agent (delayed estimated **aggregate** channel)
- complex summary: log-mag mean/std/min/max (4) + per-user log-norms (K) + fixed random
  projection of re/im (`COMPLEX_PROJ_DIM=8`);
- previous power action (one-hot);
- delivered messages (budget × message_dim, zero-padded);
- tau (if `expose_tau_to_actor`); normalized episode time.

### ris_agent (delayed estimated **reflected** component)
- complex summary of reflected; reflected/direct log power ratio;
- grouped RIS phases as per-group mean cos/sin (2G);
- previous RIS action (one-hot); delivered messages; tau; time.

### ma_agent (delayed estimated **direct** component)
- complex summary of direct (captures per-antenna spatial variation);
- normalized MA offsets vs cell centers (2M); min-spacing margin;
- MA update countdown; previous MA action (one-hot); delivered messages; tau; time.

## No leakage

Actor observations never contain: true channel, true NLoS gains, future channel,
another agent's current action, the reward before acting, or the true objective vector.

## Centralized critic state

`build_centralized_state(...)` may use the **true** channel: summaries of true
aggregate/direct/reflected + grouped phases + MA offsets + spacing margin + all agents'
previous actions + tau + regime one-hot + episode time. Never fed to an actor.

## Complex features + normalization

`complex_to_features(x, mode="summary", projection)` produces the compact summary.
The random projection matrix is seeded once (`FeatureProjection`) and stored in the
checkpoint (never redrawn per episode). Observations/states are further normalised by
running `RunningMeanStd` statistics (also checkpointed).

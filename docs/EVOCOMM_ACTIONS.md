# EvoComm-MOMGRAP — Actions

Discrete codebooks (`momgrap.marl.actions`). No per-RIS-element control, no direct
complex-beamformer synthesis, no per-slot MA movement.

## bs_agent — power codebook (`decode_power_action`)

Index selects a power profile, computed from the **estimated** effective channel gains
`|h_k^H w_k|^2`:

```
0 equal
1 inverse-gain fairness (alpha=0.5)
2 inverse-gain fairness (alpha=1.0)
3 inverse-gain fairness (alpha=2.0)
4 equal / fairness blend
5.. focus user 0, focus user 1, ... (per user)
```

Output `p_user (1,1,K)`: nonnegative, sums to `p_max_watt`, respects a minimum floor
(`min_power_fraction`, default 5% of the equal share via water-filling). Never uses
`Config.power_policy`.

## ris_agent — grouped incremental phase (`decode_ris_action`)

L elements split into G contiguous groups. Actions:

```
0        no-op (exact)
1..2G    +/-Delta on a single group
2G+1     global +Delta
2G+2     global -Delta
```

Returns `(theta_next, n_changed_groups)`; modified phases wrapped to `[0, 2*pi)`.

## ma_agent — layout movement (`decode_ma_action`)

9 layout-level offset patterns (proposals, not absolute genotypes):

```
0 no-op   1 expand_y   2 contract_y   3 expand_z   4 contract_z
5 positive_shear   6 negative_shear   7 checkerboard_y   8 checkerboard_z
```

Step size = `ma_step_fraction_of_cell * cell_extent`; x is never moved. The candidate
`P (1,M,3)` then passes through the safety shield.

## Safety shield (`project_to_safety`)

- **MA**: `repair_positions` then validate; if still infeasible, fall back to the last
  feasible layout (or USPA cell centres). `cfg.repair_iters` is never mutated.
- **RIS**: wrap phases, optional quantization bits, finite check.
- **Power**: clamp negative, renormalise to `p_max`, equal power if the total is ~0.
- Records pre-repair violation magnitudes (the reward penalty uses the *before* values).

## Communication action

Second `MultiDiscrete` component per agent: `0=no-send`, `1/2=other agents`. Validity
follows the graph (`valid_action_mask`); no-send is always valid.

"""Evolutionary NSGA-II search over communication graphs for a fixed policy.

Run (baseline policy, no checkpoint needed):
  python scripts_marl/evolve_comm_graph.py --config configs_marl/smoke.yaml \
      --policy heuristic_full_comm --pop-size 12 --max-gen 8

With a trained policy:
  python scripts_marl/evolve_comm_graph.py --checkpoint <ckpt.pt> --config <cfg>

Outputs -> outputs_marl/graph_search/<run_id>/ (never touches results/ or figures/).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch

from momgrap.marl.algorithms.rmappo.networks import build_actors
from momgrap.marl.config import evocomm_config_to_dict, load_evocomm_config, save_resolved_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.evaluation.metrics import collect_metadata
from momgrap.marl.evolution.graph_encoding import GraphConstraints
from momgrap.marl.evolution.graph_fitness import cache_prefix, make_cached_evaluator, FitnessCache
from momgrap.marl.evolution.graph_search import evaluate_preset_graphs, run_graph_search
from momgrap.marl.normalization import RunningMeanStd
from momgrap.marl.observations.builder import observation_sizes
from momgrap.marl.policies.baselines import TrainedActorPolicy, make_policy


def _trained_policy_factory(evo, checkpoint, device):
    obs_sizes = observation_sizes(evo)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    actors = build_actors(evo, obs_sizes)
    for a in actors:
        actors[a].load_state_dict(ckpt["actors"][a])
        actors[a].to(device)
    obs_rms = {a: RunningMeanStd(obs_sizes[a], device=device) for a in actors}
    for a in obs_rms:
        obs_rms[a].load_state_dict(ckpt["obs_rms"][a])
    return lambda: TrainedActorPolicy(actors, obs_rms, device=device, deterministic=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_REPO_ROOT / "configs_marl" / "smoke.yaml"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--policy", default="heuristic_full_comm")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.1])
    ap.add_argument("--pop-size", type=int, default=12)
    ap.add_argument("--max-gen", type=int, default=8)
    ap.add_argument("--max-edges", type=int, default=None)
    ap.add_argument("--require-connectivity", action="store_true")
    ap.add_argument("--no-outage", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default=str(_REPO_ROOT / "outputs_marl" / "graph_search"))
    args = ap.parse_args()

    evo = load_evocomm_config(args.config)
    env = MOMGRAPParallelEnv(evo, seed=0, eval_mode=True, comm_mode="learned")

    if args.checkpoint:
        make_pol = _trained_policy_factory(evo, args.checkpoint, args.device)
        policy_id = f"ckpt:{args.checkpoint}"
    else:
        make_pol = lambda: make_policy(args.policy, seed=0)  # noqa: E731
        policy_id = f"baseline:{args.policy}"

    prefix = cache_prefix(policy_id, evocomm_config_to_dict(evo), args.seeds, args.taus)
    cache = FitnessCache(prefix)
    evaluate_fn, cache = make_cached_evaluator(
        env, make_pol, args.seeds, args.taus, cache=cache, include_outage=not args.no_outage)

    constraints = GraphConstraints(
        max_edges=args.max_edges, require_weak_connectivity=args.require_connectivity)

    result = run_graph_search(evaluate_fn, constraints, pop_size=args.pop_size,
                              max_gen=args.max_gen, seed=0)
    baselines = evaluate_preset_graphs(evaluate_fn)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"{pathlib.Path(args.config).stem}_{stamp}"
    out = pathlib.Path(args.outdir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    obj_names = ["min_rate", "sensing_db", "neg_messages"] + ([] if args.no_outage else ["neg_outage"])
    (out / "pareto_graphs.json").write_text(json.dumps({
        "objective_names": obj_names,
        "pareto": [{"bits": b, "objectives": o}
                   for b, o in zip(result.pareto_bits, result.pareto_objectives)],
        "baselines": baselines,
    }, indent=2), encoding="utf-8")

    with open(out / "pareto_objectives.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bits", *obj_names])
        for b, o in zip(result.pareto_bits, result.pareto_objectives):
            w.writerow(["".join(map(str, b)), *o])

    with open(out / "population_history.csv", "w", newline="", encoding="utf-8") as f:
        if result.history:
            w = csv.DictWriter(f, fieldnames=list(result.history[0].keys()))
            w.writeheader()
            w.writerows(result.history)

    cache.save(out / "cache.pkl")
    save_resolved_config(evo, out / "config_resolved.yaml")
    meta = collect_metadata(args.seeds)
    meta.update({"policy_id": policy_id, "taus": args.taus, "pop_size": args.pop_size,
                 "max_gen": args.max_gen, "cache_prefix": prefix})
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"pareto front: {len(result.pareto_bits)} graphs  -> {out}")
    for b, o in zip(result.pareto_bits, result.pareto_objectives):
        print(f"  {''.join(map(str,b))}  {[round(x,3) for x in o]}")
    print("baselines:")
    for name, d in baselines.items():
        print(f"  {name:10s} {''.join(map(str,d['bits']))}  {[round(x,3) for x in d['objectives']]}")


if __name__ == "__main__":
    main()

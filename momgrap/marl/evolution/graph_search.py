"""NSGA-II search over 6-bit communication graphs (reuses momgrap.nsga).

Operators: binary tournament (rank asc, crowding desc), uniform crossover, bit-flip
mutation, then :func:`repair_graph`. Survival is NSGA-II environmental selection over
the m-objective vectors. ``hypervolume_2d`` is NOT used (objectives are 3D/4D).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ...nsga import crowding_distance, environmental_selection, fast_nondominated_sort
from .graph_encoding import GraphConstraints, N_EDGES, decode_graph, repair_graph


@dataclass
class GraphSearchResult:
    pareto_bits: list[list[int]]
    pareto_objectives: list[list[float]]
    population_bits: list[list[int]]
    population_objectives: list[list[float]]
    history: list[dict] = field(default_factory=list)


def _random_bits(gen, constraints):
    bits = (torch.rand(N_EDGES, generator=gen) < 0.5).long()
    return repair_graph(bits, constraints)


def _uniform_crossover(a, b, gen):
    mask = torch.rand(N_EDGES, generator=gen) < 0.5
    c1 = torch.where(mask, a, b)
    c2 = torch.where(mask, b, a)
    return c1.long(), c2.long()


def _bitflip(bits, rate, gen):
    flip = torch.rand(N_EDGES, generator=gen) < rate
    return (bits.long() ^ flip.long())


def _tournament(rank, cd, gen):
    n = rank.shape[0]
    i = int(torch.randint(0, n, (1,), generator=gen).item())
    j = int(torch.randint(0, n, (1,), generator=gen).item())
    if rank[i] < rank[j]:
        return i
    if rank[j] < rank[i]:
        return j
    return i if cd[i] >= cd[j] else j


def _evaluate_population(pop, evaluate_fn):
    return torch.tensor([evaluate_fn(b) for b in pop], dtype=torch.float32)


def run_graph_search(evaluate_fn, constraints: GraphConstraints, pop_size=20, max_gen=15,
                     mutation_rate=1.0 / N_EDGES, seed=0) -> GraphSearchResult:
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    pop = [_random_bits(gen, constraints) for _ in range(pop_size)]
    F = _evaluate_population(pop, evaluate_fn)
    history = []

    for g in range(max_gen):
        rank = fast_nondominated_sort(F)
        cd = crowding_distance(F, rank)
        offspring = []
        while len(offspring) < pop_size:
            p1, p2 = _tournament(rank, cd, gen), _tournament(rank, cd, gen)
            c1, c2 = _uniform_crossover(pop[p1], pop[p2], gen)
            offspring.append(repair_graph(_bitflip(c1, mutation_rate, gen), constraints))
            offspring.append(repair_graph(_bitflip(c2, mutation_rate, gen), constraints))
        offspring = offspring[:pop_size]
        Foff = _evaluate_population(offspring, evaluate_fn)

        combined = pop + offspring
        Fcomb = torch.cat([F, Foff], dim=0)
        idx = environmental_selection(Fcomb, pop_size)
        pop = [combined[int(i)] for i in idx]
        F = Fcomb[idx]
        history.append({
            "generation": g,
            "front_size": int((fast_nondominated_sort(F) == 0).sum()),
            **{f"mean_F{k+1}": float(F[:, k].mean()) for k in range(F.shape[1])},
            **{f"max_F{k+1}": float(F[:, k].max()) for k in range(F.shape[1])},
        })

    rank = fast_nondominated_sort(F)
    keep = (rank == 0).nonzero().flatten().tolist()
    return GraphSearchResult(
        pareto_bits=[pop[i].tolist() for i in keep],
        pareto_objectives=[F[i].tolist() for i in keep],
        population_bits=[b.tolist() for b in pop],
        population_objectives=F.tolist(),
        history=history,
    )


def evaluate_preset_graphs(evaluate_fn, extra_random=2, seed=0) -> dict:
    """Objectives for the baseline graphs: none / full / ring / star_bs + random."""
    from ..communication.graph import build_adjacency
    from .graph_encoding import encode_graph

    out = {}
    for name in ("none", "full", "ring", "star_bs"):
        bits = encode_graph(build_adjacency(name))
        out[name] = {"bits": bits.tolist(), "objectives": evaluate_fn(bits)}
    gen = torch.Generator()
    gen.manual_seed(int(seed) + 4242)
    for r in range(extra_random):
        bits = (torch.rand(N_EDGES, generator=gen) < 0.5).long()
        out[f"random_{r}"] = {"bits": bits.tolist(), "objectives": evaluate_fn(bits)}
    return out

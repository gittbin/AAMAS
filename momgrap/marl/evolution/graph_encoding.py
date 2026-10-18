"""Binary encoding of the 3-agent directed communication graph (6 possible edges).

Edge order (bit index -> directed edge over bs=0, ris=1, ma=2):
  0: bs -> ris   1: bs -> ma   2: ris -> bs   3: ris -> ma   4: ma -> bs   5: ma -> ris
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

EDGES = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
N_EDGES = len(EDGES)
_BS_EDGES = tuple(k for k, (i, j) in enumerate(EDGES) if i == 0 or j == 0)  # edges touching bs


def encode_graph(adjacency: torch.Tensor) -> torch.Tensor:
    """Directed (3,3) bool adjacency -> (6,) long bitstring."""
    return torch.tensor([int(bool(adjacency[i, j])) for (i, j) in EDGES], dtype=torch.long)


def decode_graph(bits, device=None) -> torch.Tensor:
    """(6,) bits -> directed (3,3) bool adjacency (diagonal zero)."""
    bits = torch.as_tensor(bits).reshape(-1)
    A = torch.zeros(3, 3, dtype=torch.bool)
    for k, (i, j) in enumerate(EDGES):
        if int(bits[k]) == 1:
            A[i, j] = True
    return A.to(device) if device is not None else A


@dataclass
class GraphConstraints:
    max_edges: int | None = None
    allow_empty: bool = True
    require_weak_connectivity: bool = False
    require_bs_incident: bool = False


def _components(bits) -> set:
    parent = [0, 1, 2]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, (i, j) in enumerate(EDGES):
        if int(bits[k]) == 1:
            parent[find(i)] = find(j)
    return {find(x) for x in range(3)}


def is_weakly_connected(bits) -> bool:
    return len(_components(bits)) == 1


def _bs_incident(bits) -> bool:
    return any(int(bits[k]) == 1 for k in _BS_EDGES)


def repair_graph(bits, constraints: GraphConstraints) -> torch.Tensor:
    """Project a bitstring onto the feasible set (best-effort, deterministic).

    Order: cap edges -> ensure bs-incident -> ensure weak connectivity -> non-empty.
    Note: mutually infeasible constraints (e.g. max_edges=1 with weak connectivity over
    3 nodes) cannot all hold; later steps win.
    """
    bits = torch.as_tensor(bits).reshape(-1).clone().to(torch.long)

    if constraints.max_edges is not None:
        while int(bits.sum()) > constraints.max_edges:
            set_idx = (bits == 1).nonzero().flatten()
            bits[int(set_idx[-1])] = 0                       # drop highest-index edge

    if constraints.require_bs_incident and not _bs_incident(bits):
        bits[0] = 1                                          # bs -> ris

    if constraints.require_weak_connectivity:
        while not is_weakly_connected(bits):
            added = False
            for k, (i, j) in enumerate(EDGES):
                if int(bits[k]) == 0:
                    comps = _components(bits)
                    # would this edge merge two components?
                    trial = bits.clone()
                    trial[k] = 1
                    if len(_components(trial)) < len(comps):
                        bits[k] = 1
                        added = True
                        break
            if not added:
                break

    if not constraints.allow_empty and int(bits.sum()) == 0:
        bits[0] = 1

    return bits

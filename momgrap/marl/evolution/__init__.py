"""Evolutionary communication-graph search (NSGA-II over 6-bit graphs)."""

from __future__ import annotations

from .graph_encoding import (  # noqa: F401
    EDGES,
    N_EDGES,
    GraphConstraints,
    decode_graph,
    encode_graph,
    is_weakly_connected,
    repair_graph,
)
from .graph_fitness import (  # noqa: F401
    FitnessCache,
    cache_prefix,
    graph_objectives,
    make_cached_evaluator,
)
from .graph_search import (  # noqa: F401
    GraphSearchResult,
    evaluate_preset_graphs,
    run_graph_search,
)

__all__ = [
    "EDGES",
    "N_EDGES",
    "GraphConstraints",
    "encode_graph",
    "decode_graph",
    "repair_graph",
    "is_weakly_connected",
    "graph_objectives",
    "FitnessCache",
    "cache_prefix",
    "make_cached_evaluator",
    "run_graph_search",
    "evaluate_preset_graphs",
    "GraphSearchResult",
]

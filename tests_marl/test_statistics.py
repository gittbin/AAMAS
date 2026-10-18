"""Tests for the statistical protocol + Pareto utilities (Task 18)."""

from __future__ import annotations

import math
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from momgrap.marl.evaluation.statistics import (
    cohens_d_paired,
    epsilon_indicator,
    holm_correction,
    mean_ci,
    paired_t_test,
    pareto_coverage,
    pareto_front,
)


def test_mean_ci():
    r = mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(r["mean"] - 3.0) < 1e-9
    assert r["n"] == 5 and r["ci"] > 0
    single = mean_ci([2.0])
    assert single["mean"] == 2.0 and single["ci"] == 0.0


def test_paired_t_and_effect_size():
    a = [2.5, 3.0, 4.5, 5.0, 6.5]
    b = [1.0, 2.0, 3.0, 4.0, 5.0]      # positive but varying differences
    tt = paired_t_test(a, b)
    assert tt["p"] < 0.01               # consistently a > b
    assert cohens_d_paired(a, b) > 0
    # identical -> no effect
    assert abs(cohens_d_paired(a, a)) < 1e-9


def test_holm_correction():
    res = holm_correction([0.01, 0.04, 0.03], alpha=0.05)
    assert len(res["adjusted"]) == 3
    # monotone non-decreasing in sorted order; smallest p (0.01) -> 0.03
    assert abs(res["adjusted"][0] - 0.03) < 1e-9
    assert all(0.0 <= x <= 1.0 for x in res["adjusted"])
    # a clearly significant single test
    assert holm_correction([0.001])["reject"][0] is True


def test_pareto_front():
    # maximize both: (2,2) dominates (1,1); (3,0) and (0,3) non-dominated
    pts = [[1, 1], [2, 2], [3, 0], [0, 3]]
    nd = pareto_front(pts, maximize=[True, True])
    assert set(nd) == {1, 2, 3}         # index 0 dominated
    # mixed: maximize x, minimize y
    pts2 = [[1, 1], [1, 2], [2, 2]]
    nd2 = pareto_front(pts2, maximize=[True, False])
    assert 0 in nd2 and 2 in nd2 and 1 not in nd2


def test_pareto_coverage_and_epsilon():
    A = [[2.0, 2.0]]
    B = [[1.0, 1.0]]
    assert pareto_coverage(A, B, maximize=[True, True]) == 1.0   # A dominates B
    assert pareto_coverage(B, A, maximize=[True, True]) == 0.0
    eps = epsilon_indicator(A, B, maximize=[True, True])         # A already covers B
    assert eps <= 0.0
    eps2 = epsilon_indicator(B, A, maximize=[True, True])        # B needs +1 to reach A
    assert abs(eps2 - 1.0) < 1e-9

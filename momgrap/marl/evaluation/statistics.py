"""Statistical protocol + Pareto reporting for the scientific evaluation (Task 18).

Provides mean / std / 95% CI (t-based), paired t-tests, paired Cohen's d, Holm
multiple-comparison correction, and multi-objective Pareto utilities (non-dominated
set, coverage / C-metric, additive epsilon indicator, 2D projections).

3D hypervolume is intentionally NOT computed here: ``momgrap.metrics.hypervolume_2d``
is 2-objective only and must not be mis-adapted; coverage + epsilon are reported
instead (a verified HV library can be added later).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_ci(values, confidence: float = 0.95) -> dict:
    x = np.asarray(values, dtype=float)
    n = x.size
    mean = float(x.mean()) if n else 0.0
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        h = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1)) * std / np.sqrt(n)
    else:
        h = 0.0
    return {"mean": mean, "std": std, "ci": h, "n": int(n)}


def paired_t_test(a, b) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return {"t": 0.0, "p": 1.0, "n": int(a.size)}
    t, p = stats.ttest_rel(a, b)
    return {"t": float(t), "p": float(p), "n": int(a.size)}


def cohens_d_paired(a, b) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size < 2:
        return 0.0
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 1e-12 else 0.0


def holm_correction(pvalues, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down: return monotone adjusted p-values + reject flags."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return {"adjusted": [], "reject": []}
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)      # enforce monotonicity
        adjusted[idx] = running
    return {"adjusted": adjusted.tolist(), "reject": (adjusted <= alpha).tolist()}


# ---------------------------------------------------------------- Pareto
def _to_maximize(points, maximize) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    for k, mx in enumerate(maximize):
        if not mx:
            pts[:, k] = -pts[:, k]
    return pts


def pareto_front(points, maximize) -> list[int]:
    """Indices of the non-dominated points. ``maximize``: bool per objective."""
    pts = _to_maximize(points, maximize)
    n = pts.shape[0]
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                nd[i] = False
                break
    return np.nonzero(nd)[0].tolist()


def pareto_coverage(a_points, b_points, maximize) -> float:
    """C-metric C(A,B): fraction of B weakly dominated by some point of A."""
    A = _to_maximize(a_points, maximize)
    B = _to_maximize(b_points, maximize)
    if B.shape[0] == 0:
        return 0.0
    covered = 0
    for b in B:
        if any(np.all(a >= b) and np.any(a > b) for a in A):
            covered += 1
    return covered / B.shape[0]


def epsilon_indicator(a_points, reference, maximize) -> float:
    """Additive epsilon indicator I_eps+(A, R) (smaller is better)."""
    A = _to_maximize(a_points, maximize)
    R = _to_maximize(reference, maximize)
    if A.shape[0] == 0 or R.shape[0] == 0:
        return float("inf")
    worst = -float("inf")
    for r in R:
        best = min(float(np.max(r - a)) for a in A)
        worst = max(worst, best)
    return worst


def pairwise_2d_projections(points, names) -> list[dict]:
    pts = np.asarray(points, dtype=float)
    m = pts.shape[1]
    out = []
    for i in range(m):
        for j in range(i + 1, m):
            out.append({"x": names[i], "y": names[j],
                        "points": pts[:, [i, j]].tolist()})
    return out

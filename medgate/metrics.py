"""Statistical metrics used by the verification components.

Stdlib-only, deterministic, and written so that a *second, independent*
implementation can recompute the same quantities (see
``verification.independent_check``) — that independence is itself one of the
six Verification Coverage components.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple


def accuracy(pairs: Sequence[Tuple[int, int]]) -> float:
    """Fraction of (predicted, true) pairs that agree."""
    if not pairs:
        return 0.0
    correct = sum(1 for p, t in pairs if p == t)
    return correct / len(pairs)


def wilson_ci(
    successes: int, n: int, z: float = 1.96
) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (lower, upper). Used so that scope verification is about the
    *confidence* of the measured performance, not just the point estimate.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def ece(confidences: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    """Expected Calibration Error.

    The average gap between mean confidence and empirical accuracy within
    equal-width probability bins. 0 = perfectly calibrated, 1 = maximally
    miscalibrated.
    """
    total = len(labels)
    if total == 0:
        return 0.0
    bins: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, labels):
        b = min(int(c * n_bins), n_bins - 1)
        bins[b].append((c, y))
    e = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(y for _, y in b) / len(b)
        e += (len(b) / total) * abs(avg_conf - avg_acc)
    return e


def subgroup_accuracy(
    pairs_by_group: Dict[str, List[Tuple[int, int]]]
) -> Dict[str, Tuple[float, int]]:
    """Per-group (accuracy, n). Groups with no samples are omitted."""
    out: Dict[str, Tuple[float, int]] = {}
    for g, pairs in pairs_by_group.items():
        if pairs:
            out[g] = (accuracy(pairs), len(pairs))
    return out


def population_stability_index(
    expected: Sequence[float], actual: Sequence[float], eps: float = 1e-4
) -> float:
    """Population Stability Index between two probability distributions.

    0 = identical, ~0.1 = mild shift, >0.25 = large shift (common heuristic).
    Used by the post-market drift monitor to detect distribution drift.
    """
    total = 0.0
    for e, a in zip(expected, actual):
        e = max(e, eps)
        a = max(a, eps)
        total += (a - e) * math.log(a / e)
    return total


def composition(groups: Sequence[str]) -> Dict[str, float]:
    """Relative frequency of each group label (a probability distribution)."""
    if not groups:
        return {}
    counts: Dict[str, int] = {}
    for g in groups:
        counts[g] = counts.get(g, 0) + 1
    n = len(groups)
    return {g: c / n for g, c in counts.items()}
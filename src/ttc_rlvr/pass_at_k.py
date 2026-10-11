"""Unbiased pass@k estimator from Chen et al. (2021)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Return the probability that a size-k subset contains a correct sample.

    Args:
        n: Number of generated samples.
        c: Number of correct generated samples.
        k: Number of samples available to the hypothetical solver.
    """
    if isinstance(n, bool) or isinstance(c, bool) or isinstance(k, bool):
        raise TypeError("n, c and k must be integers, not booleans")
    if not all(isinstance(value, int) for value in (n, c, k)):
        raise TypeError("n, c and k must be integers")
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= c <= n:
        raise ValueError("c must satisfy 0 <= c <= n")
    if not 1 <= k <= n:
        raise ValueError("k must satisfy 1 <= k <= n")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mean_pass_at_k(correctness: Iterable[Sequence[bool]], ks: Sequence[int]) -> dict[int, float]:
    """Average the unbiased estimator over problems.

    Each row in ``correctness`` contains all sampled outcomes for one problem.
    """
    rows = [list(row) for row in correctness]
    if not rows:
        raise ValueError("correctness must contain at least one problem")
    sample_counts = {len(row) for row in rows}
    if len(sample_counts) != 1 or 0 in sample_counts:
        raise ValueError("all problems must have the same positive number of samples")
    n = sample_counts.pop()
    invalid = [k for k in ks if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= n]
    if invalid:
        raise ValueError(f"all k values must satisfy 1 <= k <= {n}; got {invalid}")
    return {
        k: sum(estimate_pass_at_k(n, sum(bool(x) for x in row), k) for row in rows) / len(rows)
        for k in ks
    }


"""Acquisition functions scoring unlabeled candidates for the next batch.

ucb:    mean + kappa * std            -- upper confidence bound
ei:     expected improvement over the best measured fitness so far
greedy: mean only                    -- pure exploitation baseline
"""

import numpy as np
from scipy.stats import norm


def ucb(mean: np.ndarray, std: np.ndarray, kappa: float) -> np.ndarray:
    return mean + kappa * std


def ei(mean: np.ndarray, std: np.ndarray, y_best: float, xi: float) -> np.ndarray:
    """Expected improvement over incumbent, exploiting mean-shift xi."""
    improvement = mean - y_best - xi
    z = improvement / std
    return improvement * norm.cdf(z) + std * norm.pdf(z)


def acquire(
    mean: np.ndarray,
    std: np.ndarray,
    kind: str,
    kappa: float = 2.0,
    xi: float = 0.01,
    y_best: float = 0.0,
) -> np.ndarray:
    if kind == "ucb":
        return ucb(mean, std, kappa)
    if kind == "ei":
        return ei(mean, std, y_best, xi)
    if kind == "greedy":
        return mean
    raise ValueError(f"unknown acquisition kind {kind!r}")

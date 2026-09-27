"""L2-penalised logistic regression by Newton steps, in float64 chunks.

Written out rather than imported so this supplement depends on numpy
alone. The penalty sits on the wires, never on the intercept: shrinking
the intercept would pull the fitted base rate off the observed one.
"""
from __future__ import annotations

import numpy as np

from .constants import BLOCK_ROWS, NEWTON_STEPS, NEWTON_TOL


def standardise_stats(X, fit_rows, block: int = BLOCK_ROWS):
    """Centre and scale of every wire, taken on the fit rows only."""
    fit_rows = np.asarray(fit_rows, dtype=bool)
    n_fit = int(fit_rows.sum())
    n_cols = int(X.shape[1])
    mean = np.zeros(n_cols)
    for a in range(0, X.shape[0], block):
        b = min(a + block, X.shape[0])
        m = fit_rows[a:b]
        if m.any():
            mean += np.asarray(X[a:b][m], dtype=np.float64).sum(0)
    mean /= max(n_fit, 1)
    var = np.zeros(n_cols)
    for a in range(0, X.shape[0], block):
        b = min(a + block, X.shape[0])
        m = fit_rows[a:b]
        if m.any():
            var += ((np.asarray(X[a:b][m], dtype=np.float64) - mean) ** 2
                    ).sum(0)
    var /= max(n_fit - 1, 1)
    sd = np.sqrt(var)
    # A wire constant on the fit half carries no information; pass it
    # through as zeros so the coefficients stay aligned with the names.
    sd[sd <= 0] = 1.0
    return mean, sd


def fit_logistic(X, y, fit_rows, mean, sd, ridge: float,
                 steps: int = NEWTON_STEPS, tol: float = NEWTON_TOL,
                 block: int = BLOCK_ROWS):
    """Newton fit. Returns (coefficients with intercept last, moves)."""
    fit_rows = np.asarray(fit_rows, dtype=bool)
    k = int(X.shape[1])
    w = np.zeros(k + 1)
    idx = np.flatnonzero(fit_rows)
    yv = np.asarray(y)[idx].astype(np.float64)
    moves: list[float] = []
    for _ in range(steps):
        g = np.zeros(k + 1)
        hess = np.zeros((k + 1, k + 1))
        for a in range(0, len(idx), block):
            sl = idx[a:a + block]
            z = (np.asarray(X[sl], dtype=np.float64) - mean) / sd
            z = np.hstack([z, np.ones((len(sl), 1))])
            eta = np.clip(z @ w, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-eta))
            g += z.T @ (yv[a:a + len(sl)] - p)
            hess += (z * (p * (1.0 - p))[:, None]).T @ z
        pen = ridge * float(np.trace(hess[:k, :k])) / max(k, 1) + 1e-9
        g[:k] -= pen * w[:k]
        hess.flat[::k + 2] += pen
        hess[k, k] -= pen
        step = np.linalg.solve(hess, g)
        w += step
        moves.append(float(np.abs(step).max()))
        if moves[-1] < tol:
            break
    return w, moves


def apply_logistic(X, pick_rows, mean, sd, w,
                   block: int = BLOCK_ROWS) -> np.ndarray:
    """Log-odds for every pick row."""
    pick_rows = np.asarray(pick_rows, dtype=bool)
    idx = np.flatnonzero(pick_rows)
    out = np.empty(len(idx), dtype=np.float64)
    for a in range(0, len(idx), block):
        sl = idx[a:a + block]
        z = (np.asarray(X[sl], dtype=np.float64) - mean) / sd
        out[a:a + len(sl)] = z @ w[:-1] + w[-1]
    return out

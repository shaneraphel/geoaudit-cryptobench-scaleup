"""Metrics. Pure functions over score vectors; no file access.

``roc_auc`` is the trapezoidal rule over the ROC curve with tied scores
handled by averaging ranks, which is what the frozen paper's metric does.
"""
from __future__ import annotations

import numpy as np


def roc_auc(scores, labels) -> float | None:
    """ROC-AUC, or None when one class is absent."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    t = np.asarray(labels).ravel()
    pos = t == 1
    n1 = int(pos.sum())
    n0 = int(len(t) - n1)
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranked = np.empty(len(s))
    i = 0
    while i < len(s):
        k = i
        while k + 1 < len(s) and s[order[k + 1]] == s[order[i]]:
            k += 1
        ranked[order[i:k + 1]] = 0.5 * (i + k) + 1.0
        i = k + 1
    return float((ranked[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def per_unit_auc_vector(score, y, n_res_per) -> list[float]:
    """Per-chain ROC-AUC; NaN where a chain has a single class."""
    out: list[float] = []
    off = 0
    for n in n_res_per:
        n = int(n)
        s = np.asarray(score[off:off + n], dtype=np.float64)
        t = np.asarray(y[off:off + n])
        off += n
        if t.sum() == 0 or t.sum() == n:
            out.append(float("nan"))
            continue
        a = roc_auc(s, t)
        out.append(float("nan") if a is None else float(a))
    return out


def per_unit_auc(score, y, n_res_per) -> float:
    """Mean of the per-chain ROC-AUC vector, NaN chains skipped."""
    vec = per_unit_auc_vector(score, y, n_res_per)
    good = [v for v in vec if not np.isnan(v)]
    return float(np.mean(good)) if good else float("nan")


def paired_bootstrap_ci(field_vec, logit_vec, draws: int, seed: int) -> dict:
    """Paired mean of field minus logistic over chains both arms scored."""
    x = np.array([np.nan if v is None else v for v in field_vec],
                 dtype=float)
    z = np.array([np.nan if v is None else v for v in logit_vec],
                 dtype=float)
    ok = ~(np.isnan(x) | np.isnan(z))
    d = x[ok] - z[ok]
    if len(d) == 0:
        return {"n_paired": 0, "delta": None, "ci95": None,
                "excludes_zero": False}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(draws, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_paired": int(len(d)),
            "delta": round(float(d.mean()), 6),
            "ci95": [round(float(lo), 6), round(float(hi), 6)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def cluster_bootstrap_ci(chain_units: list[str], chain_delta: list[float],
                         cluster_of: dict[str, str], draws: int,
                         seed: int) -> dict:
    """Bootstrap the mean chain delta, resampling whole MMseqs clusters."""
    groups: dict[str, list[float]] = {}
    for u, d in zip(chain_units, chain_delta):
        groups.setdefault(str(cluster_of[u]), []).append(float(d))
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for b in range(draws):
        choose = rng.integers(0, len(keys), size=len(keys))
        parts = [groups[keys[i]] for i in choose]
        boot[b] = np.mean(np.concatenate(parts))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    mean = float(np.mean(chain_delta))
    if lo > 0:
        sign = "field"
    elif hi < 0:
        sign = "logistic"
    else:
        sign = "unresolved"
    return {"mean": round(mean, 6),
            "ci95": [round(float(lo), 6), round(float(hi), 6)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "sign": sign,
            "n_chains": len(chain_units),
            "n_clusters": len(keys)}

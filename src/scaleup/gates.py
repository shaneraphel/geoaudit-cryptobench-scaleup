"""Spatial smoothing: the neighbourhood mean, matched to the raw spread.

Each residue's score gains its neighbours' mean within a radius, rescaled
so the gate term has the raw score's standard deviation. The gate search
tries the three predeclared (radius, weight) pairs on the scored half and
keeps the best; every arm gets the same search.
"""
from __future__ import annotations

import numpy as np

from .constants import GATE_CHUNK, GATES
from .metrics import per_unit_auc


def neighbourhood_mean(score, ctr, n_res_per, radius: float) -> np.ndarray:
    out = np.empty(len(score))
    r2 = radius * radius
    off = 0
    score = np.asarray(score, dtype=np.float64)
    ctr = np.asarray(ctr, dtype=np.float64)
    for n in n_res_per:
        n = int(n)
        c = ctr[off:off + n]
        v = score[off:off + n]
        acc = np.empty(n)
        for i in range(0, n, GATE_CHUNK):
            d2 = ((c[i:i + GATE_CHUNK, None, :] - c[None, :, :]) ** 2
                  ).sum(-1)
            a = (d2 <= r2).astype(np.float64)
            acc[i:i + GATE_CHUNK] = ((a @ v)
                                     / np.maximum(a.sum(1), 1.0))
        out[off:off + n] = acc
        off += n
    return out


def spread_matched_gate(score, ctr, n_res_per, radius: float,
                        weight: float) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    g = neighbourhood_mean(score, ctr, n_res_per, radius)
    sd_s, sd_g = float(np.std(score)), float(np.std(g))
    if sd_g <= 0:
        return score
    return score + weight * g * (sd_s / sd_g)


def gate_search(score, ctr, n_res_per, y,
                gates=GATES) -> tuple[float, float, str, np.ndarray]:
    """(raw AUC, best gated AUC, gate name, gated scores)."""
    raw = per_unit_auc(score, y, n_res_per)
    best, gname, chosen = -1.0, "", None
    for radius, weight in gates:
        g = spread_matched_gate(score, ctr, n_res_per, radius, weight)
        auc = per_unit_auc(g, y, n_res_per)
        if auc > best:
            best, gname, chosen = auc, f"r{int(radius)} w{weight}", g
    assert chosen is not None
    return raw, best, gname, chosen

"""Cluster-disjoint train/pick halvings.

Clusters (MMseqs2 ids from the manifest) are shuffled with a seeded RNG
and cut in half; chains inherit their cluster's side. Rows inherit their
chain's side. No residue, chain, or cluster ever sits on both sides.
"""
from __future__ import annotations

import numpy as np


def cluster_halving(units: list[str], cluster_of: dict[str, str],
                    seed: int) -> np.ndarray:
    """Boolean per-chain mask: True means the fit side."""
    missing = [u for u in units if u not in cluster_of]
    if missing:
        raise ValueError(f"{len(missing)} units lack a cluster id, "
                         f"first {missing[:3]}")
    clusters = sorted({cluster_of[u] for u in units})
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    fit = set(clusters[:len(clusters) // 2])
    return np.array([cluster_of[u] in fit for u in units])


def row_masks(is_fit_chain: np.ndarray, n_res_per) -> tuple[np.ndarray,
                                                            np.ndarray]:
    """Expand a per-chain mask to per-row fit/pick masks."""
    row_chain = np.repeat(np.arange(len(is_fit_chain)),
                          np.asarray(n_res_per, dtype=np.int64))
    fit = np.asarray(is_fit_chain, dtype=bool)[row_chain]
    return fit, ~fit


def pick_vectors(n_res_per, is_fit_chain, y, ctr, units):
    """Per-pick-half vectors: counts, labels, centres, unit names."""
    n_res_per = np.asarray(n_res_per)
    n_pick = np.array([n for n, f in zip(n_res_per, is_fit_chain) if not f])
    _fit, pick_rows = row_masks(np.asarray(is_fit_chain, dtype=bool),
                                n_res_per)
    ypick = np.asarray(y)[pick_rows]
    ctr_pick = np.asarray(ctr)[pick_rows]
    pick_units = [u for u, f in zip(units, is_fit_chain) if not f]
    return n_pick, ypick, ctr_pick, pick_units

"""The counting field: pair tables, cell rates, integer fan-out.

A table is a pair of wires. The two quaternary digits concatenate into an
address 0..15; the table stores how many fit-half residues landed in each
cell and how many were cryptic. Scoring reads one cell per table and adds
the cell rates with integer weights from a single regularised solve.

All passes stream row blocks. The only object that grows with the number
of tables is the KxK Gram solved once per half.
"""
from __future__ import annotations

import numpy as np

from .constants import BLOCK_ROWS


def partition_tables(n_wires: int, width: int, rounds: int,
                     seed: int) -> list[list[int]]:
    """``rounds`` random partitions of every wire into groups of ``width``."""
    rng = np.random.default_rng(seed)
    tables: list[list[int]] = []
    for _ in range(rounds):
        perm = rng.permutation(n_wires)
        tables += [perm[i:i + width].tolist()
                   for i in range(0, n_wires, width)]
    return [t for t in tables if len(t) >= 2]


def cell_offsets(tables, levels: int) -> np.ndarray:
    """Start of each table's cells inside one concatenated array."""
    sizes = [levels ** len(t) for t in tables]
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


def _block_rates(D_blk: np.ndarray, frac: np.ndarray, offsets: np.ndarray,
                 col0: np.ndarray, col1: np.ndarray) -> np.ndarray:
    """Cell rate of every row in the block under every pair table."""
    local = (D_blk[:, col0].astype(np.int64)
             + (D_blk[:, col1].astype(np.int64) << 2))
    return frac[local + offsets[:-1]]


def compile_cells(D, y, fit_rows, tables, offsets, levels: int,
                  block: int = BLOCK_ROWS):
    """Cell rates and cell totals on the fit rows. Width-2 tables only."""
    if any(len(t) != 2 for t in tables):
        raise ValueError("this supplement scores the published pair field")
    idx = np.flatnonzero(np.asarray(fit_rows, dtype=bool))
    total = int(offsets[-1])
    tot = np.zeros(total, dtype=np.int64)
    pos = np.zeros(total, dtype=np.float64)
    yv = np.asarray(y)[idx].astype(np.float64)
    col0 = np.array([t[0] for t in tables], dtype=np.int32)
    col1 = np.array([t[1] for t in tables], dtype=np.int32)
    k = len(tables)
    for a in range(0, len(idx), block):
        sl = idx[a:a + block]
        blk = np.asarray(D[sl])
        local = (blk[:, col0].astype(np.int64)
                 + (blk[:, col1].astype(np.int64) << 2))
        flat = (local + offsets[:-1]).ravel()
        tot += np.bincount(flat, minlength=total)
        w = np.repeat(yv[a:a + len(sl)], k)
        pos += np.bincount(flat, weights=w, minlength=total)
    rate = float(yv.mean()) if len(yv) else 0.0
    frac = np.where(tot > 0, pos / np.maximum(tot, 1), rate)
    return frac, tot


def integer_fanout(D, y, fit_rows, tables, offsets, frac, ridge: float,
                   cap: int, block: int = BLOCK_ROWS) -> np.ndarray:
    """One regularised solve over table outputs, rounded to integers."""
    idx = np.flatnonzero(np.asarray(fit_rows, dtype=bool))
    yv = np.asarray(y)[idx]
    pos = yv == 1
    n1 = int(pos.sum())
    n0 = int(len(yv) - n1)
    if n1 == 0 or n0 == 0:
        raise ValueError("a fit half with one class cannot be scored")
    k = len(tables)
    col0 = np.array([t[0] for t in tables], dtype=np.int32)
    col1 = np.array([t[1] for t in tables], dtype=np.int32)
    s1 = np.zeros(k)
    s0 = np.zeros(k)
    for a in range(0, len(idx), block):
        sl = idx[a:a + block]
        v = _block_rates(np.asarray(D[sl]), frac, offsets, col0, col1)
        p = pos[a:a + len(sl)]
        s1 += v[p].sum(0)
        s0 += v[~p].sum(0)
    mu1, mu0 = s1 / n1, s0 / n0
    gram = np.zeros((k, k))
    for a in range(0, len(idx), block):
        sl = idx[a:a + block]
        v = _block_rates(np.asarray(D[sl]), frac, offsets, col0, col1)
        p = pos[a:a + len(sl)]
        c = np.where(p[:, None], v - mu1, v - mu0)
        gram += c.T @ c
    gram /= max(len(yv) - 2, 1)
    gram.flat[::k + 1] += ridge * float(np.trace(gram)) / k + 1e-12
    w = np.linalg.solve(gram, mu1 - mu0)
    peak = float(np.abs(w).max())
    if peak <= 0:
        return np.zeros(k, dtype=np.int64)
    return np.round(w / peak * cap).astype(np.int64)


def score_pick(D, pick_rows, tables, offsets, frac, mult,
               block: int = BLOCK_ROWS) -> np.ndarray:
    """Field score for every pick row."""
    idx = np.flatnonzero(np.asarray(pick_rows, dtype=bool))
    out = np.empty(len(idx), dtype=np.float64)
    m = np.asarray(mult, dtype=np.float64)
    col0 = np.array([t[0] for t in tables], dtype=np.int32)
    col1 = np.array([t[1] for t in tables], dtype=np.int32)
    for a in range(0, len(idx), block):
        sl = idx[a:a + block]
        v = _block_rates(np.asarray(D[sl]), frac, offsets, col0, col1)
        out[a:a + len(sl)] = v @ m
    return out

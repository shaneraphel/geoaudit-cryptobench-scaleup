"""Within-chain quartile banding of the wires.

Each residue is ranked against the other residues of its own chain, ties
sharing the mid-rank, and the rank is cut into ``levels`` bands. One row
block (one chain) is ever materialised at a time; the full digit matrix
is written to disk once and memory-mapped afterwards.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def rank_digits_vector(x: np.ndarray, levels: int) -> np.ndarray:
    """Mid-rank banding of one vector into ``levels`` bands."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    order = np.argsort(x, kind="stable")
    r = np.empty(n)
    i = 0
    while i < n:
        k = i
        while k + 1 < n and x[order[k + 1]] == x[order[i]]:
            k += 1
        r[order[i:k + 1]] = 0.5 * (i + k)
        i = k + 1
    return np.clip(np.floor(r / max(n - 1, 1) * levels), 0,
                   levels - 1).astype(np.int8)


def chain_digits_matrix(X: np.ndarray, n_res_per, levels: int) -> np.ndarray:
    """In-memory banding. For tests and small inputs, not for the cache."""
    X = np.asarray(X, dtype=np.float64)
    out = np.empty(X.shape, dtype=np.int8)
    off = 0
    for n in n_res_per:
        n = int(n)
        blk = X[off:off + n]
        for j in range(X.shape[1]):
            out[off:off + n, j] = rank_digits_vector(blk[:, j], levels)
        off += n
    return out


def digitise_to_file(cache_path: str | Path, out_path: str | Path,
                     levels: int) -> Path:
    """Band the whole cache chain by chain into a .npy digit file."""
    from .data import load_cache
    out_path = Path(out_path)
    z = load_cache(cache_path)
    try:
        X, n_res = z["X"], z["n_res_per"]
        n_rows, n_wires = X.shape
        out = np.lib.format.open_memmap(out_path, mode="w+",
                                        dtype=np.int8,
                                        shape=(n_rows, n_wires))
        off = 0
        for n in n_res:
            n = int(n)
            blk = np.asarray(X[off:off + n], dtype=np.float64)
            for j in range(n_wires):
                out[off:off + n, j] = rank_digits_vector(blk[:, j], levels)
            off += n
            del blk
        out.flush()
    finally:
        z.close()
    return out_path


def load_digits(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Memory-map a digit file written by :func:`digitise_to_file`."""
    d = np.lib.format.open_memmap(path, mode="r")
    if d.shape != tuple(shape) or d.dtype != np.int8:
        raise ValueError(f"digit file is {d.shape} {d.dtype}, "
                         f"want {tuple(shape)} int8")
    return d

"""Within-class edge counts on the 10 Å centroid ball.

The empty graph, the path on three vertices and K_3 all have three
vertices of one class. Their edge counts are 0, 2 and 3. That split is
what makes an edge count a different functor from the class headcount.
"""
from __future__ import annotations

import numpy as np

BALL_R2 = 100.0
AA20 = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
CHARGED = frozenset({1, 3, 6, 8, 11})
HPHOB = frozenset({0, 9, 10, 12, 13, 17, 19})
POLAR = frozenset({2, 4, 5, 15, 16, 18})
GLY = frozenset({7})
PRO = frozenset({14})
CLASS_OF = np.full(20, -1, dtype=np.int8)
for _ids, _k in ((CHARGED, 0), (HPHOB, 1), (POLAR, 2), (GLY, 3), (PRO, 4)):
    for _i in _ids:
        CLASS_OF[_i] = _k
BANK = ("charged", "hphob", "polar")
CLASS_ID = {"charged": 0, "hphob": 1, "polar": 2}
WIRE_NAMES = (
    "e_charged@10", "e_hphob@10", "e_polar@10",
    "ctrl~n_res@10", "ctrl~n_c_charged@10", "ctrl~n_c_hphob@10",
    "ctrl~n_c_polar@10",
)
COL = {n: j for j, n in enumerate(WIRE_NAMES)}


def contact_graph(xyz: np.ndarray) -> np.ndarray:
    """Symmetric 10 Å relation with the self-loop removed."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.size and not np.isfinite(xyz).all():
        raise ValueError("non-finite centroid")
    d2 = ((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1)
    A = d2 <= BALL_R2
    np.fill_diagonal(A, False)
    if not np.array_equal(A, A.T):
        raise AssertionError("10 Å relation is not symmetric")
    return A


def induced_edges(A: np.ndarray, S: np.ndarray) -> int:
    """Unordered edges of the subgraph induced by ``S``. Zero is a count."""
    if int(np.asarray(S).size) < 2:
        return 0
    two = int(A[np.ix_(S, S)].sum())
    if two % 2:
        raise AssertionError("induced adjacency is not symmetric")
    return two // 2


def chain_ecc(xyz: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """One chain, columns in ``WIRE_NAMES`` order."""
    xyz = np.asarray(xyz, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)
    n = len(xyz)
    W = np.zeros((n, len(WIRE_NAMES)), dtype=np.int64)
    if n == 0:
        return W
    if len(codes) != n or (codes < 0).any() or (codes >= 20).any():
        raise ValueError("codes are not a length-matched amino-acid index")
    A = contact_graph(xyz)
    cls = CLASS_OF[codes]
    for i in range(n):
        members = np.append(np.flatnonzero(A[i]), i)
        cm = cls[members]
        W[i, COL["ctrl~n_res@10"]] = members.size
        for name in BANK:
            S = members[cm == CLASS_ID[name]]
            W[i, COL[f"e_{name}@10"]] = induced_edges(A, S)
            W[i, COL[f"ctrl~n_c_{name}@10"]] = S.size
            n_c = int(S.size)
            ceiling = n_c * (n_c - 1) // 2
            if int(W[i, COL[f"e_{name}@10"]]) > ceiling:
                raise AssertionError(f"e_{name} exceeds C(n_c, 2)")
    return W


def stack_chains(ctr, codes, n_res_per) -> np.ndarray:
    """ECC matrix for a whole fold, one chain at a time."""
    out = np.empty((len(codes), len(WIRE_NAMES)), dtype=np.int64)
    off = 0
    for n in n_res_per:
        n = int(n)
        out[off:off + n] = chain_ecc(np.asarray(ctr[off:off + n]),
                                     np.asarray(codes[off:off + n]))
        off += n
    if off != len(codes):
        raise AssertionError("n_res_per does not sum to the code column")
    return out


def spearman(a, b) -> float:
    """Average-rank Spearman. Constant inputs are undefined, not zero."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")

    def _rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        i = 0
        while i < len(x):
            k = i
            while k + 1 < len(x) and x[order[k + 1]] == x[order[i]]:
                k += 1
            r[order[i:k + 1]] = 0.5 * (i + k)
            i = k + 1
        return r
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))

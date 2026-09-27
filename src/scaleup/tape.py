"""Residue tapes. One chain is codes, centroids and a cryptic label.

The PDB reader keeps the first conformation of each CA atom and refuses
a non-finite coordinate. Unknown residue names are dropped, not assigned
a class.
"""
from __future__ import annotations

import numpy as np

from .ecc import AA20

AA_INDEX = {name: i for i, name in enumerate(AA20)}


def parse_ca(text: str, chain: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return residue numbers, codes and CA coordinates for one chain."""
    seen: set[tuple[int, str]] = set()
    resseq: list[int] = []
    codes: list[int] = []
    xyz: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        if len(line) < 54 or line[21] != chain:
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        name = line[17:20].strip()
        if name not in AA_INDEX:
            continue
        icode = line[26] if len(line) > 26 else " "
        key = (int(line[22:26]), icode)
        if key in seen:
            continue
        seen.add(key)
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        if not np.isfinite((x, y, z)).all():
            raise ValueError("non-finite CA coordinate")
        resseq.append(key[0])
        codes.append(AA_INDEX[name])
        xyz.append((x, y, z))
    return (np.asarray(resseq, dtype=np.int64),
            np.asarray(codes, dtype=np.int64),
            np.asarray(xyz, dtype=np.float64).reshape(-1, 3))

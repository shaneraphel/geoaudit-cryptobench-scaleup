"""Pinned input loading. Mmap only; nothing is copied into RAM wholesale.

The training cache is read through a memory map so a run's resident set
is set by the block size, not by the 600 MB wire matrix. Every loader
verifies what it opened before returning it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .constants import (CACHE_N_ROWS, CACHE_SHA256, MANIFEST_SHA256,
                        N_TRAIN_UNITS, N_WIRES)


def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cache(path: str | Path):
    """Open the training wide cache as read-only memmaps.

    Returns the open NpzFile; the caller must keep it alive while the
    arrays are used. Refuses any file whose sha256 differs from the pin.
    """
    path = Path(path)
    if "test" in path.name:
        raise ValueError("refusing a cache whose name says test fold")
    digest = sha256_of_file(path)
    if digest != CACHE_SHA256:
        raise ValueError(f"cache sha256 {digest} does not match the pin; "
                         f"this supplement is defined on exactly one matrix")
    z = np.load(path, allow_pickle=False, mmap_mode="r")
    want = {"X", "y", "ctr", "n_res_per", "units", "names"}
    if set(z.files) < want:
        raise ValueError(f"cache lacks {sorted(want - set(z.files))}")
    if z["X"].shape != (CACHE_N_ROWS, N_WIRES):
        raise ValueError(f"unexpected wire matrix {z['X'].shape}")
    if len(z["units"]) != N_TRAIN_UNITS:
        raise ValueError(f"unexpected unit count {len(z['units'])}")
    return z


def load_manifest(path: str | Path) -> dict:
    """Load the training manifest; refuses any file that is not the pin."""
    path = Path(path)
    digest = sha256_of_file(path)
    if digest != MANIFEST_SHA256:
        raise ValueError(f"manifest sha256 {digest} does not match the pin")
    doc = json.loads(path.read_text())
    entries = doc.get("entries") or []
    if not entries:
        raise ValueError("manifest has no entries")
    return doc


def cluster_of_manifest(doc: dict) -> dict[str, str]:
    return {f"{e['pdb']}_{e['chain']}": str(e["cluster_id"])
            for e in doc["entries"]}


def check_units_match_cache(units: list[str], doc: dict) -> None:
    """Cache row order must equal manifest order, exactly."""
    ids = [f"{e['pdb']}_{e['chain']}" for e in doc["entries"]]
    if list(units) != ids:
        raise ValueError("cache unit order differs from manifest order; "
                         "cluster ids would attach to the wrong chains")

"""Download an apo bridge and score it with the integer field.

The filters, the column list and the split seed are fixed in
docs/BRIDGE_PREREG.md. ``collect`` only downloads and writes tapes.
``score`` refuses to run until that file exists. The OSF test fold is
read solely to exclude its PDB ids.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import http.client
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .constants import (FANOUT_CAP, LEVELS, N_BOOT, PAIRING_SEED, RIDGE,
                        ROUNDS)
from .data import load_manifest
from .digits import chain_digits_matrix
from .ecc import COL, chain_ecc
from .field import (cell_offsets, compile_cells, integer_fanout,
                    partition_tables, score_pick)
from .gates import gate_search
from .logistic import apply_logistic, fit_logistic, standardise_stats
from .metrics import paired_bootstrap_ci, per_unit_auc_vector
from .splits import cluster_halving, pick_vectors, row_masks
from .tape import parse_ca

API = "https://apoholo.cz/api/db"
RCSB = "https://files.rcsb.org/download/{}.pdb"
UA = {"User-Agent": "geoaudit-scaleup/0.3", "Accept-Encoding": "identity"}
UNIPROTS = (
    "P00533", "P04637", "P24941", "P00918", "P56817", "P04150",
    "P15056", "P12931", "P00519", "P06401", "P37231", "P35354",
    "P00742", "P00734", "P15121", "P00441", "P11802", "P42336",
    "P10275", "P03372", "P03951", "P09871", "P00746", "P35869",
)
IGNORED = {"HOH", "DOD", "WAT", "UNK", "ABA", "MPD", "GOL", "SO4", "PO4",
           "ZN", "MG", "CA", "NA", "CL"}
MAX_ENTRIES = 8
MAX_CHAINS = 480
MIN_RES, MAX_RES = 40, 800
MIN_POCKET = 8
SPLIT_SEED = 20260928
FIELD_ROUNDS = 4
COLUMNS = (
    "ctrl~n_res@10", "ctrl~n_c_charged@10", "ctrl~n_c_polar@10",
    "ctrl~n_c_hphob@10", "e_charged@10", "e_polar@10",
)
SCHEMA = "geoaudit.scaleup.ahoj_bridge.v1"


def _get(url: str) -> bytes:
    last = "none"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.IncompleteRead, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url} ({last})")


def _json(url: str) -> dict:
    return json.loads(_get(url))


def test_fold_pdb_ids(blob: bytes) -> set[str]:
    """PDB ids named by the OSF test fold. Used for exclusion only."""
    doc = json.loads(blob)
    ids = set(doc)
    for rows in doc.values():
        for row in rows:
            ids.add(str(row.get("holo_pdb_id", "")))
            text = str(row.get("apo_pymol_selection", ""))
            head = text.split(" ", 1)[0]
            if len(head) == 4:
                ids.add(head)
    return {i.lower() for i in ids if len(i) == 4}


def train_pdb_ids(manifest_path: Path) -> set[str]:
    doc = load_manifest(manifest_path)
    return {e["pdb"].lower() for e in doc["entries"]}


def _csv_residues(text: str) -> dict[tuple[str, str], set[int]]:
    """Map (pdb, chain) to residue numbers listed as cryptic."""
    out: dict[tuple[str, str], set[int]] = {}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        pdb = (row.get("chain") or "").strip().lower()
        for tok in (row.get("mapped_binding_residues") or "").split():
            if "_" not in tok:
                continue
            chain, num = tok.split("_", 1)
            if not chain or not num.lstrip("-").isdigit():
                continue
            out.setdefault((pdb, chain), set()).add(int(num))
    return out


def _accept_entry(entry: dict) -> bool:
    if entry.get("failed"):
        return False
    lig = str(entry.get("target_ligand") or "").upper()
    if lig in IGNORED or not lig:
        return False
    if int(entry.get("num_apo_chains") or 0) < 1:
        return False
    method = str(entry.get("target_experimental_method") or "")
    if method and "X-RAY" not in method.upper() and "XRAY" not in method.upper():
        return False
    res = entry.get("target_resolution")
    if res not in (None, "", "-"):
        try:
            if float(res) > 2.5:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _apo_ok(apo: dict) -> bool:
    if str(apo.get("apoholo_assignment") or "") != "A":
        return False
    if apo.get("is_alphafold"):
        return False
    method = str(apo.get("experimental_method") or "")
    if "X-RAY" not in method.upper() and "XRAY" not in method.upper():
        return False
    try:
        if float(apo.get("resolution")) > 2.5:
            return False
        if float(apo.get("pocket_rmsd")) <= 2.0:
            return False
        if int(apo.get("mapped_binding_residues_num") or 0) < MIN_POCKET:
            return False
    except (TypeError, ValueError):
        return False
    chains = apo.get("chains") or []
    return len(chains) == 1 and len(str(chains[0])) == 1


def collect(root: Path, manifest: Path, test_json: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    blocked = train_pdb_ids(manifest) | test_fold_pdb_ids(test_json.read_bytes())
    search_dir = root / "search"
    query_dir = root / "query"
    csv_dir = root / "residues"
    pdb_dir = root / "pdb"
    for d in (search_dir, query_dir, csv_dir, pdb_dir):
        d.mkdir(exist_ok=True)
    chain_dir = root / "chains"
    chain_dir.mkdir(exist_ok=True)
    tapes_code, tapes_xyz, tapes_y = [], [], []
    units, uniprots = [], []
    seen_pdb: set[str] = set()
    for path in sorted(chain_dir.glob("*.npz")):
        one = np.load(path, allow_pickle=False)
        tapes_code.append(one["codes"])
        tapes_xyz.append(one["ctr"])
        tapes_y.append(one["y"])
        units.append(str(one["unit"]))
        uniprots.append(str(one["uniprot"]))
        seen_pdb.add(str(one["unit"]).split("_", 1)[0])
    if units:
        print(f"resume {len(units)} chains already on disk", flush=True)
    n_skip = {"blocked": 0, "parse": 0, "short_label": 0, "length": 0}
    for acc in UNIPROTS:
        if len(units) >= MAX_CHAINS:
            break
        sp = search_dir / f"{acc}.json"
        if not sp.exists():
            url = f"{API}/search?{urllib.parse.urlencode({'uniprot_ids': acc})}"
            sp.write_bytes(_get(url))
            print(f"search {acc}", flush=True)
        doc = json.loads(sp.read_text())
        taken = 0
        for entry in doc.get("entries") or []:
            if taken >= MAX_ENTRIES or len(units) >= MAX_CHAINS:
                break
            if not _accept_entry(entry):
                continue
            key = str(entry["entry_key"])
            safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in key)
            qp = query_dir / f"{safe}.json"
            cp = csv_dir / f"{safe}.csv"
            qp.parent.mkdir(parents=True, exist_ok=True)
            if not qp.exists():
                try:
                    qp.write_bytes(_get(f"{API}/entry/{urllib.parse.quote(key)}/query-result"))
                except (RuntimeError, OSError) as exc:
                    print(f"  skip query {key}: {exc}", flush=True)
                    continue
            if not cp.exists():
                try:
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    blob = _get(f"{API}/entry/{urllib.parse.quote(key)}/download/{urllib.parse.quote(key)}.zip")
                    zf = zipfile.ZipFile(io.BytesIO(blob))
                    name = next(n for n in zf.namelist() if n.endswith("pocket_residues.csv"))
                    cp.write_bytes(zf.read(name))
                except (RuntimeError, zipfile.BadZipFile, StopIteration) as exc:
                    print(f"  skip zip {key}: {exc}", flush=True)
                    continue
            taken += 1
            q = json.loads(qp.read_text())
            residues = _csv_residues(cp.read_text())
            for apo in q.get("found_apo") or []:
                if len(units) >= MAX_CHAINS:
                    break
                if not _apo_ok(apo):
                    continue
                pdb = str(apo["pdb_id"]).lower()
                chain = str(apo["chains"][0])
                if pdb in blocked or pdb in seen_pdb:
                    n_skip["blocked"] += 1
                    continue
                labels = residues.get((pdb, chain))
                if not labels or len(labels) < MIN_POCKET:
                    n_skip["short_label"] += 1
                    continue
                pp = pdb_dir / f"{pdb}.pdb"
                if not pp.exists() or pp.stat().st_size == 0:
                    try:
                        pp.write_bytes(_get(RCSB.format(pdb)))
                    except RuntimeError:
                        n_skip["parse"] += 1
                        continue
                try:
                    raw = pp.read_bytes()
                    text = (gzip.decompress(raw).decode() if raw[:2] == b"\x1f\x8b"
                            else raw.decode(errors="replace"))
                    resseq, codes, xyz = parse_ca(text, chain)
                except (ValueError, UnicodeError):
                    n_skip["parse"] += 1
                    continue
                if not (MIN_RES <= len(codes) <= MAX_RES):
                    n_skip["length"] += 1
                    continue
                y = np.array([1 if int(r) in labels else 0 for r in resseq], dtype=np.int64)
                if int(y.sum()) < MIN_POCKET or int(y.sum()) == len(y):
                    n_skip["short_label"] += 1
                    continue
                unit = f"{pdb}_{chain}"
                np.savez_compressed(
                    chain_dir / f"{unit}.npz", codes=codes, ctr=xyz, y=y,
                    unit=np.array(unit), uniprot=np.array(acc))
                tapes_code.append(codes)
                tapes_xyz.append(xyz)
                tapes_y.append(y)
                units.append(unit)
                uniprots.append(acc)
                seen_pdb.add(pdb)
                print(f"  keep {pdb}_{chain} n={len(codes)} pos={int(y.sum())} "
                      f"({len(units)})", flush=True)
    if not units:
        raise SystemExit("no chain survived the bridge filters")
    codes = np.concatenate(tapes_code)
    ctr = np.concatenate(tapes_xyz)
    y = np.concatenate(tapes_y)
    n_res = np.array([len(c) for c in tapes_code], dtype=np.int64)
    out = root / "tapes.npz"
    np.savez_compressed(out, codes=codes, ctr=ctr, y=y, n_res_per=n_res,
                        units=np.array(units), uniprots=np.array(uniprots))
    receipt = {
        "n_chains": len(units), "n_rows": int(len(y)),
        "n_uniprots": len(set(uniprots)), "skipped": n_skip,
        "pdb_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "reads_test_fold": False,
    }
    (root / "COLLECT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"collected {len(units)} chains, {len(y)} residues", flush=True)
    return receipt


def _features(codes, ctr, n_res) -> np.ndarray:
    cols = [COL[n] for n in COLUMNS]
    blocks = []
    off = 0
    for n in n_res:
        n = int(n)
        w = chain_ecc(np.asarray(ctr[off:off + n]), np.asarray(codes[off:off + n]))
        blocks.append(w[:, cols].astype(np.float64))
        off += n
    return np.concatenate(blocks)


def _arm(X, y, ctr, n_res, units, cluster_of, kind: str):
    is_fit = cluster_halving(units, cluster_of, SPLIT_SEED)
    # A one-chain side makes the logistic undefined. Refuse rather than drop.
    if is_fit.sum() == 0 or (~is_fit).sum() == 0:
        raise SystemExit("the UniProt halving put every chain on one side")
    fit_rows, pick_rows = row_masks(is_fit, n_res)
    n_pick, ypick, ctr_pick, _ = pick_vectors(n_res, is_fit, y, ctr, units)
    if kind == "logistic":
        mean, sd = standardise_stats(X, fit_rows)
        w, moves = fit_logistic(X, y, fit_rows, mean, sd, RIDGE)
        raw_score = apply_logistic(X, pick_rows, mean, sd, w)
        extra = {"newton_steps": len(moves), "converged": bool(moves[-1] < 1e-6)}
    elif kind == "field":
        D = chain_digits_matrix(X, n_res, LEVELS)
        tables = partition_tables(X.shape[1], 2, FIELD_ROUNDS, PAIRING_SEED)
        offsets = cell_offsets(tables, LEVELS)
        frac, _tot = compile_cells(D, y, fit_rows, tables, offsets, LEVELS)
        mult = integer_fanout(D, y, fit_rows, tables, offsets, frac, RIDGE, FANOUT_CAP)
        raw_score = score_pick(D, pick_rows, tables, offsets, frac, mult)
        extra = {"n_tables": len(tables), "n_tables_used": int((mult != 0).sum())}
    else:
        raise ValueError(kind)
    raw, gated, gate, gated_score = gate_search(raw_score, ctr_pick, n_pick, ypick)
    vec = per_unit_auc_vector(gated_score, ypick, n_pick)
    return {"raw": raw, "gated": gated, "gate": gate, "per_unit": vec, **extra}


def score(root: Path) -> dict:
    z = np.load(root / "tapes.npz", allow_pickle=False)
    codes, ctr, y = z["codes"], z["ctr"], z["y"]
    n_res = z["n_res_per"]
    units = [str(u) for u in z["units"]]
    uniprots = [str(u) for u in z["uniprots"]]
    cluster_of = dict(zip(units, uniprots))
    X = _features(codes, ctr, n_res)
    count_idx = [i for i, n in enumerate(COLUMNS) if n.startswith("ctrl~")]
    full = _arm(X, y, ctr, n_res, units, cluster_of, "logistic")
    counts = _arm(X[:, count_idx], y, ctr, n_res, units, cluster_of, "logistic")
    field = _arm(X, y, ctr, n_res, units, cluster_of, "field")
    gap_field = paired_bootstrap_ci(
        [None if np.isnan(v) else round(float(v), 6) for v in field["per_unit"]],
        [None if np.isnan(v) else round(float(v), 6) for v in full["per_unit"]],
        N_BOOT, SPLIT_SEED + 1)
    gap_edges = paired_bootstrap_ci(
        [None if np.isnan(v) else round(float(v), 6) for v in full["per_unit"]],
        [None if np.isnan(v) else round(float(v), 6) for v in counts["per_unit"]],
        N_BOOT, SPLIT_SEED + 1)
    for arm in (full, counts, field):
        del arm["per_unit"]
    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "n_chains": len(units),
        "n_rows": int(len(y)),
        "n_uniprots": len(set(uniprots)),
        "split_seed": SPLIT_SEED,
        "columns": list(COLUMNS),
        "field_rounds": FIELD_ROUNDS,
        "logistic_full": full,
        "logistic_counts": counts,
        "integer_field": field,
        "field_minus_logistic": gap_field,
        "edges_minus_counts": gap_edges,
    }
    payload = json.dumps(doc, indent=2, allow_nan=False) + "\n"
    doc["artifact_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--root", required=True)
    c.add_argument("--manifest", required=True)
    c.add_argument("--test-json", required=True)
    s = sub.add_parser("score")
    s.add_argument("--root", required=True)
    s.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        collect(Path(a.root), Path(a.manifest), Path(a.test_json))
        return 0
    doc = score(Path(a.root))
    Path(a.json).write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(f"wrote {a.json}", flush=True)
    print(f"  field {doc['integer_field']['gated']:.4f}  "
          f"logistic {doc['logistic_full']['gated']:.4f}  "
          f"delta {doc['field_minus_logistic']['delta']:+.4f} "
          f"{doc['field_minus_logistic']['ci95']}", flush=True)
    print(f"  edges-counts {doc['edges_minus_counts']['delta']:+.4f} "
          f"{doc['edges_minus_counts']['ci95']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

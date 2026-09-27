"""Score surviving within-class edge counts against the 645-wire logistic.

The withdrawal rule and both augmented arms are fixed in
docs/ECC_CEILING_PREREG.md. This file does not choose a column after
seeing the ROC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from .constants import (BOOT_SEED, N_BOOT, N_WIRES, PAIRING_SEED,
                        PUBLISHED_LOGIT, REPRODUCE_TOL, RIDGE)
from .data import check_units_match_cache, cluster_of_manifest, load_cache, load_manifest
from .ecc import BANK, COL, WIRE_NAMES, spearman, stack_chains
from .gates import gate_search
from .logistic import apply_logistic, fit_logistic, standardise_stats
from .metrics import paired_bootstrap_ci, per_unit_auc_vector
from .splits import cluster_halving, pick_vectors, row_masks

WITHDRAW_AT = 0.95
SCHEMA = "geoaudit.scaleup.ecc_ceiling.v1"


def _round(vec):
    return [None if np.isnan(v) else round(float(v), 6) for v in vec]


def _logistic_gated(X, y, ctr, n_res, units, cluster_of):
    is_fit = cluster_halving(units, cluster_of, PAIRING_SEED)
    fit_rows, pick_rows = row_masks(is_fit, n_res)
    n_pick, ypick, ctr_pick, _ = pick_vectors(n_res, is_fit, y, ctr, units)
    mean, sd = standardise_stats(X, fit_rows)
    w, moves = fit_logistic(X, y, fit_rows, mean, sd, RIDGE)
    score = apply_logistic(X, pick_rows, mean, sd, w)
    raw, gated, gate, gated_score = gate_search(score, ctr_pick, n_pick, ypick)
    vec = per_unit_auc_vector(gated_score, ypick, n_pick)
    return {"raw": raw, "gated": gated, "gate": gate,
            "newton_steps": len(moves),
            "converged": bool(moves[-1] < 1e-6),
            "per_unit": _round(vec)}


def run(cache_path, manifest_path, cascade_path) -> dict:
    z = load_cache(cache_path)
    cas = np.load(cascade_path, allow_pickle=False)
    try:
        if "test" in Path(cascade_path).name:
            raise SystemExit("refusing a cascade whose name says test")
        units = [str(u) for u in z["units"]]
        doc = load_manifest(manifest_path)
        cluster_of = cluster_of_manifest(doc)
        check_units_match_cache(units, doc)
        if list(cas["units"]) != units or not np.array_equal(
                cas["n_res_per"], z["n_res_per"]):
            raise SystemExit("cascade residue universe is not the pinned cache")
        if not np.array_equal(cas["y"], z["y"]) or not np.allclose(
                cas["ctr"], z["ctr"]):
            raise SystemExit("cascade labels or centroids drifted from the cache")
        ecc = stack_chains(cas["ctr"], cas["codes"], cas["n_res_per"])
        withdrawal = {}
        kept = []
        for name in BANK:
            r = spearman(ecc[:, COL[f"e_{name}@10"]],
                         ecc[:, COL[f"ctrl~n_c_{name}@10"]])
            gone = abs(r) >= WITHDRAW_AT
            withdrawal[name] = {"spearman_vs_n_c": r, "withdrawn": gone}
            if not gone:
                kept.append(f"e_{name}@10")
        X = np.asarray(z["X"], dtype=np.float64)
        y, ctr, n_res = z["y"], z["ctr"], z["n_res_per"]
        base = _logistic_gated(X, y, ctr, n_res, units, cluster_of)
        if abs(base["gated"] - PUBLISHED_LOGIT) > REPRODUCE_TOL:
            raise SystemExit(
                f"baseline logistic {base['gated']:.6f} did not reproduce "
                f"{PUBLISHED_LOGIT}")
        arms = {"baseline_645": base}
        blocks = {
            "edges": [COL[n] for n in kept],
            "edges_and_controls": [COL[n] for n in kept] + [
                COL["ctrl~n_res@10"],
                *[COL[f"ctrl~n_c_{n.split('_')[1].split('@')[0]}@10"]
                  for n in kept],
            ],
        }
        for arm, cols in blocks.items():
            extra = ecc[:, cols].astype(np.float64)
            aug = _logistic_gated(np.hstack([X, extra]), y, ctr, n_res,
                                  units, cluster_of)
            paired = paired_bootstrap_ci(aug["per_unit"], base["per_unit"],
                                         N_BOOT, BOOT_SEED)
            aug["columns"] = [WIRE_NAMES[j] for j in cols]
            aug["paired_minus_baseline"] = paired
            del aug["per_unit"]
            arms[arm] = aug
        del base["per_unit"]
        return {
            "schema": SCHEMA,
            "clinical_grade": False,
            "reads_test_fold": False,
            "n_rows": int(len(y)),
            "n_wires_baseline": N_WIRES,
            "withdrawal": withdrawal,
            "kept": kept,
            "arms": arms,
            "environment": {"python": platform.python_version(),
                            "numpy": np.__version__},
            "what_this_does_not_say": (
                "A gap here is a training-half ceiling read. It does not "
                "open the official test fold and it does not attach the "
                "columns to a deployed field."),
        }
    finally:
        z.close()
        cas.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cascade", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    data = Path(a.data_dir)
    doc = run(data / "_wide_cache_train.npz", data / "train_manifest.json",
              a.cascade)
    payload = json.dumps(doc, indent=2, allow_nan=False) + "\n"
    doc["artifact_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    out = Path(a.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}", flush=True)
    for name, row in doc["withdrawal"].items():
        print(f"  e_{name}@10 spearman {row['spearman_vs_n_c']:.4f} "
              f"withdrawn {row['withdrawn']}", flush=True)
    for arm, row in doc["arms"].items():
        extra = ""
        if "paired_minus_baseline" in row:
            p = row["paired_minus_baseline"]
            extra = f"  delta {p['delta']:+.4f} {p['ci95']}"
        print(f"  {arm:<22} gated {row['gated']:.4f}{extra}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Post-hoc logistic on external Sets A and B+C.

Coefficients, standardisation and the spatial gate are fixed from the
CryptoBench training fold. See docs/EXTERNAL_LOGISTIC_POSTHOC.md.
The frozen paper repository is imported and not modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

FROZEN = Path("/Volumes/FOLIATION/pipeline_local/geoaudit-cryptobench")
sys.path.insert(0, str(FROZEN / "src"))

from pocket_bench.methods.algebraic_descriptors import (  # noqa: E402
    FEATURE_NAMES, algebraic_residue_features)
from pocket_bench.methods.table_field import TableField  # noqa: E402
from pocket_bench.methods.wide_descriptors import build_wide  # noqa: E402

from .constants import N_BOOT, RIDGE  # noqa: E402
from .data import load_cache  # noqa: E402
from .gates import spread_matched_gate  # noqa: E402
from .logistic import apply_logistic, fit_logistic, standardise_stats  # noqa: E402
from .metrics import paired_bootstrap_ci, roc_auc  # noqa: E402

TRAIN_CACHE = Path(__file__).resolve().parents[2] / "data" / "_wide_cache_train.npz"
FIELD_PATH = FROZEN / "data/cryptobench_apo/TABLE_FIELD.json"
LOGISTIC_GATE = (18.0, 0.5)
BOOT_SEED = 20260929
ACF_MATCH_ATOL = 1e-4
SCHEMA = "geoaudit.scaleup.external_logistic_posthoc.v1"
SETS = {
    "set_a": {
        "manifest": FROZEN / "data/external/external_manifest.json",
        "predictions": FROZEN / "results/external/predictions/table_field.json",
    },
    "set_bc": {
        "manifest": FROZEN / "data/external/setbc_manifest.json",
        "predictions": FROZEN / "results/external/setbc_predictions/table_field.json",
    },
}


def fit_training(cache_path: Path):
    """Mean, scale and coefficients on every training residue."""
    z = load_cache(cache_path)
    try:
        X, y = z["X"], np.asarray(z["y"])
        rows = np.ones(len(y), dtype=bool)
        mean, sd = standardise_stats(X, rows)
        w, moves = fit_logistic(X, y, rows, mean, sd, RIDGE)
        return {"mean": mean, "sd": sd, "w": w,
                "newton_steps": len(moves),
                "converged": bool(moves[-1] < 1e-6),
                "n_rows": int(len(y)),
                "n_chains": int(len(z["units"]))}
    finally:
        z.close()


def _wires(receptor: Path, chain: str, prop: np.ndarray):
    resseq, F, codes, ctr = algebraic_residue_features(receptor, chain=chain)
    n_res = np.asarray([len(resseq)], dtype=np.int64)
    X, names = build_wide(F, codes, ctr, n_res, tuple(FEATURE_NAMES), prop)
    if X.shape[1] != 645:
        raise SystemExit(f"{receptor.name} produced {X.shape[1]} wires, not 645")
    if list(names) and len(names) != 645:
        raise SystemExit("wire names are not the 645-column builder")
    return resseq, X, ctr


def _labels(path: Path) -> set[int]:
    doc = json.loads(path.read_text())
    return {int(r) for r in doc["cryptic_residues"]}


def _acf_map(unit_doc: dict) -> dict[int, float]:
    scores = (unit_doc.get("residue_scores")
              or (unit_doc.get("extra") or {}).get("residue_scores"))
    if not scores:
        raise SystemExit("archived counting-field unit has no residue scores")
    return {int(k): float(v) for k, v in scores.items()}


def _chain(resseq, X, ctr, fit, acf: dict[int, float], cryptic: set[int]):
    raw = apply_logistic(X, np.ones(len(resseq), dtype=bool),
                         fit["mean"], fit["sd"], fit["w"])
    gated = spread_matched_gate(raw, ctr, [len(resseq)], *LOGISTIC_GATE)
    y, s_acf, s_log = [], [], []
    for i, r in enumerate(resseq):
        r = int(r)
        if r not in acf:
            continue
        y.append(1 if r in cryptic else 0)
        s_acf.append(acf[r])
        s_log.append(float(gated[i]))
    return (np.asarray(y), np.asarray(s_acf, dtype=np.float64),
            np.asarray(s_log, dtype=np.float64))


def _summarise(rows: list[dict]) -> dict:
    acf = [r["acf_roc"] for r in rows]
    log = [r["logistic_roc"] for r in rows]
    paired = paired_bootstrap_ci(acf, log, N_BOOT, BOOT_SEED)
    return {"n_paired": paired["n_paired"],
            "acf_roc": None if not acf else round(float(np.mean(acf)), 6),
            "logistic_roc": None if not log else round(float(np.mean(log)), 6),
            "acf_minus_logistic": paired,
            "chains": rows}


def evaluate_set(name: str, field: TableField, fit: dict, *,
                 check_first: bool) -> dict:
    spec = SETS[name]
    manifest = json.loads(spec["manifest"].read_text())
    preds = json.loads(spec["predictions"].read_text())["units"]
    rows = []
    skipped = []
    checked = not check_first
    for entry in manifest["entries"]:
        unit = f"{entry['pdb']}_{entry['chain']}"
        pred = preds.get(unit)
        if pred is None or pred.get("status") not in (None, "OK"):
            skipped.append({"unit": unit, "reason": "no archived field score"})
            continue
        receptor = FROZEN / entry["receptor_path"]
        resseq, X, ctr = _wires(receptor, entry["chain"], field.prop)
        if not checked:
            recomputed = field.score_matrix(X, ctr, [len(resseq)])
            archived = _acf_map(pred)
            gap = [abs(float(recomputed[i]) - archived[int(r)])
                   for i, r in enumerate(resseq) if int(r) in archived]
            if not gap or max(gap) > ACF_MATCH_ATOL:
                raise SystemExit(
                    f"{unit}: recomputed field disagrees with the archive "
                    f"(max abs {max(gap) if gap else 'undefined'})")
            print(f"  {unit} field matches archive "
                  f"(max abs {max(gap):.2e})", flush=True)
            checked = True
        y, s_acf, s_log = _chain(resseq, X, ctr, fit, _acf_map(pred),
                                 _labels(FROZEN / entry["label_path"]))
        if y.sum() == 0 or y.sum() == len(y):
            skipped.append({"unit": unit, "reason": "single class"})
            continue
        rows.append({
            "unit": unit,
            "set": entry.get("set", "set_a"),
            "n_residues": int(len(y)),
            "n_positive": int(y.sum()),
            "acf_roc": round(float(roc_auc(s_acf, y)), 6),
            "logistic_roc": round(float(roc_auc(s_log, y)), 6),
        })
        print(f"  {name} {unit} acf {rows[-1]['acf_roc']:.4f} "
              f"logistic {rows[-1]['logistic_roc']:.4f}", flush=True)
    out = _summarise(rows)
    out["skipped"] = skipped
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    print("fitting logistic on the training fold", flush=True)
    fit = fit_training(TRAIN_CACHE)
    if not fit["converged"]:
        raise SystemExit("training logistic did not converge")
    print(f"  {fit['n_chains']} chains, {fit['n_rows']} rows, "
          f"{fit['newton_steps']} Newton steps", flush=True)
    field = TableField.load(FIELD_PATH)
    if not np.allclose(field.prop, np.load(TRAIN_CACHE)["propensity_table"]):
        raise SystemExit("field propensity is not the training cache propensity")
    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "post_hoc": True,
        "not_a_preregistered_confirmation": True,
        "reads_official_test_fold": False,
        "training": {
            "n_chains": fit["n_chains"], "n_rows": fit["n_rows"],
            "newton_steps": fit["newton_steps"], "converged": True,
            "logistic_gate": {"radius": LOGISTIC_GATE[0],
                              "weight": LOGISTIC_GATE[1]},
            "gate_selected_on": "training pick half, then frozen",
            "propensity": "compiled field table, matched to the training cache",
        },
        "sets": {},
    }
    for name in ("set_a", "set_bc"):
        print(name, flush=True)
        doc["sets"][name] = evaluate_set(
            name, field, fit, check_first=(name == "set_a"))
    payload = json.dumps(doc, indent=2, allow_nan=False) + "\n"
    doc["artifact_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(f"wrote {a.json}", flush=True)
    for name, block in doc["sets"].items():
        p = block["acf_minus_logistic"]
        print(f"  {name}: acf {block['acf_roc']:.4f} logistic "
              f"{block['logistic_roc']:.4f} delta {p['delta']:+.4f} "
              f"{p['ci95']} n={p['n_paired']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

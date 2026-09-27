"""Instruments the ROC ladder does not measure.

The manuscript already records that, on mean ROC-AUC, the counting field
and a logistic regression on the same wires are not separable. Competing
again on that average is the wrong experiment: logistic is a fitted
linear probability, and the field is an integer table. This module scores
the published halving once, then reads three facts off that same fit.

1. Exactness. The pick-half score is the integer combination of cell
   rates, and the reconstruction error is reported rather than assumed.
2. Top-decile capture. Of a chain's cryptic residues, how many sit in the
   top tenth of that chain's ranking. One cut, declared in constants.
3. Nested fit size. The same pick half, with the field and the logistic
   refit on 1/8, 2/8, 4/8 and all of the fit clusters. The gate chosen at
   the full fit is frozen, so the curve is about sample size alone.

Nothing here retunes ridge, cap, width, rounds, or the gate list. The
official test fold is not opened.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from .constants import (BOOT_SEED, CACHE_N_ROWS, FANOUT_CAP,
                        FIT_FRACTION_DENOM, FIT_FRACTION_NUMS, LEVELS,
                        MANIFEST_NAME, N_BOOT, N_WIRES, PAIRING_SEED,
                        PUBLISHED_CI, PUBLISHED_DELTA, PUBLISHED_FIELD,
                        PUBLISHED_LOGIT, PUBLISHED_N_PAIRED, REPRODUCE_TOL,
                        RIDGE, ROUNDS, SCHEMA_COMPATIBLE, SUBSAMPLE_SEED,
                        TOP_FRACTION, WIDTH)
from .data import (check_units_match_cache, cluster_of_manifest, load_cache,
                   load_manifest)
from .digits import load_digits
from .field import (cell_offsets, compile_cells, integer_fanout,
                    partition_tables, score_pick)
from .gates import gate_search, spread_matched_gate
from .logistic import apply_logistic, fit_logistic, standardise_stats
from .metrics import paired_bootstrap_ci, per_unit_auc, top_fraction_capture
from .run import assert_published_half
from .splits import cluster_halving, pick_vectors, row_masks


def reconstruction_error(D, pick_rows, tables, offsets, frac, mult,
                         score) -> float:
    """Max absolute gap between a stored score and a fresh table read."""
    again = score_pick(D, pick_rows, tables, offsets, frac, mult)
    return float(np.max(np.abs(again - np.asarray(score, dtype=np.float64))))


def nested_fit_clusters(units: list[str], is_fit: np.ndarray,
                        cluster_of: dict[str, str], numer: int,
                        denom: int, seed: int) -> np.ndarray:
    """Fit-chain mask using the first ``numer/denom`` of one cluster shuffle.

    The shuffle is seeded once, so numer=1 is a subset of numer=2, and
    numer=denom returns the full fit half. Pick chains stay pick chains.
    """
    if numer < 1 or numer > denom:
        raise ValueError("numer must lie in 1..denom")
    fit_clusters = sorted({cluster_of[u] for u, f in zip(units, is_fit) if f})
    rng = np.random.default_rng(seed)
    rng.shuffle(fit_clusters)
    keep_n = max(1, int(np.floor(len(fit_clusters) * numer / denom)))
    if numer == denom:
        keep_n = len(fit_clusters)
    keep = set(fit_clusters[:keep_n])
    return np.array([bool(f) and cluster_of[u] in keep
                     for u, f in zip(units, is_fit)])


def _gate_pair(name: str) -> tuple[float, float]:
    radius = float(name.split()[0][1:])
    weight = float(name.split()[1][1:])
    return radius, weight


def _arm_at_fraction(X, D, y, ctr, n_res, units, cluster_of, is_fit_full,
                     tables, offsets, numer: int, frozen_gate: dict):
    """Refit both arms on a nested subset and score the full pick half."""
    is_fit = nested_fit_clusters(units, is_fit_full, cluster_of, numer,
                                 FIT_FRACTION_DENOM, SUBSAMPLE_SEED)
    fit_rows, pick_rows = row_masks(is_fit_full, n_res)
    # The pick half is the published one. Only the fit mask shrinks, and it
    # shrinks inside the published fit half, so no pick residue is used.
    sub_rows, _ = row_masks(is_fit, n_res)
    n_pick, ypick, ctr_pick, _ = pick_vectors(n_res, is_fit_full, y, ctr,
                                              units)
    mean, sd = standardise_stats(X, sub_rows)
    w, moves = fit_logistic(X, y, sub_rows, mean, sd, RIDGE)
    s_log = apply_logistic(X, pick_rows, mean, sd, w)
    frac, _tot = compile_cells(D, y, sub_rows, tables, offsets, LEVELS)
    mult = integer_fanout(D, y, sub_rows, tables, offsets, frac, RIDGE,
                          FANOUT_CAP)
    s_field = score_pick(D, pick_rows, tables, offsets, frac, mult)
    out = {"numer": numer, "denom": FIT_FRACTION_DENOM,
           "n_fit_chains": int(is_fit.sum()),
           "n_fit_clusters": int(len({cluster_of[u]
                                      for u, f in zip(units, is_fit) if f})),
           "logistic_converged": bool(moves[-1] < 1e-6)}
    for arm, score in (("field", s_field), ("logistic", s_log)):
        radius, weight = _gate_pair(frozen_gate[arm])
        gated = spread_matched_gate(score, ctr_pick, n_pick, radius, weight)
        out[arm] = {
            "roc_auc": per_unit_auc(gated, ypick, n_pick),
            "top_decile_capture": float(np.nanmean(top_fraction_capture(
                gated, ypick, n_pick, TOP_FRACTION))),
        }
    return out


def run_compatible(cache_path, manifest_path, digits_path) -> dict:
    z = load_cache(cache_path)
    try:
        X, y = z["X"], z["y"]
        ctr, n_res = z["ctr"], z["n_res_per"]
        units = [str(u) for u in z["units"]]
        if X.shape[1] != N_WIRES or int(np.asarray(n_res).sum()) != CACHE_N_ROWS:
            raise ValueError(f"unexpected cache geometry {X.shape}")
        doc = load_manifest(manifest_path)
        cluster_of = cluster_of_manifest(doc)
        check_units_match_cache(units, doc)
        D = load_digits(digits_path, (CACHE_N_ROWS, N_WIRES))

        is_fit = cluster_halving(units, cluster_of, PAIRING_SEED)
        fit_rows, pick_rows = row_masks(is_fit, n_res)
        n_pick, ypick, ctr_pick, pick_units = pick_vectors(
            n_res, is_fit, y, ctr, units)

        mean, sd = standardise_stats(X, fit_rows)
        w_log, moves = fit_logistic(X, y, fit_rows, mean, sd, RIDGE)
        s_log = apply_logistic(X, pick_rows, mean, sd, w_log)
        log_raw, log_auc, log_gate, log_gated = gate_search(
            s_log, ctr_pick, n_pick, ypick)

        tables = partition_tables(N_WIRES, WIDTH, ROUNDS, PAIRING_SEED)
        offsets = cell_offsets(tables, LEVELS)
        frac, tot = compile_cells(D, y, fit_rows, tables, offsets, LEVELS)
        mult = integer_fanout(D, y, fit_rows, tables, offsets, frac, RIDGE,
                              FANOUT_CAP)
        s_field = score_pick(D, pick_rows, tables, offsets, frac, mult)
        err = reconstruction_error(D, pick_rows, tables, offsets, frac, mult,
                                   s_field)
        fld_raw, fld_auc, fld_gate, fld_gated = gate_search(
            s_field, ctr_pick, n_pick, ypick)

        from .metrics import per_unit_auc_vector
        fld_vec = [None if np.isnan(v) else round(float(v), 6)
                   for v in per_unit_auc_vector(fld_gated, ypick, n_pick)]
        log_vec = [None if np.isnan(v) else round(float(v), 6)
                   for v in per_unit_auc_vector(log_gated, ypick, n_pick)]
        paired = paired_bootstrap_ci(fld_vec, log_vec, N_BOOT, BOOT_SEED)
        published_row = {
            "split_seed": PAIRING_SEED, "direction": "forward",
            "logistic": {"gated": log_auc}, "field": {"gated": fld_auc},
            "paired_field_minus_logistic": paired,
        }
        assert_published_half(published_row)

        cap_f = top_fraction_capture(fld_gated, ypick, n_pick, TOP_FRACTION)
        cap_l = top_fraction_capture(log_gated, ypick, n_pick, TOP_FRACTION)
        cap_ci = paired_bootstrap_ci(
            [None if np.isnan(v) else round(float(v), 6) for v in cap_f],
            [None if np.isnan(v) else round(float(v), 6) for v in cap_l],
            N_BOOT, BOOT_SEED + 2)

        frozen = {"field": fld_gate, "logistic": log_gate}
        curve = [_arm_at_fraction(X, D, y, ctr, n_res, units, cluster_of,
                                  is_fit, tables, offsets, numer, frozen)
                 for numer in FIT_FRACTION_NUMS]
        return {
            "schema": SCHEMA_COMPATIBLE,
            "clinical_grade": False,
            "reads_test_fold": False,
            "published_half": {
                "field_gated": fld_auc, "field_raw": fld_raw,
                "logistic_gated": log_auc, "logistic_raw": log_raw,
                "field_gate": fld_gate, "logistic_gate": log_gate,
                "paired_roc": paired,
                "reproduced": True,
                "logistic_newton_steps": len(moves),
                "logistic_converged": bool(moves[-1] < 1e-6),
            },
            "exactness": {
                "max_abs_reconstruction_error": err,
                "n_tables": len(tables),
                "n_tables_used": int((mult != 0).sum()),
                "n_cells": int(len(frac)),
                "n_cells_empty": int((tot == 0).sum()),
                "integer_multiplicity_bytes": int(mult.nbytes),
                "logistic_coefficient_bytes": int(w_log.nbytes),
            },
            "top_decile": {
                "fraction": TOP_FRACTION,
                "field_capture": round(float(np.nanmean(cap_f)), 6),
                "logistic_capture": round(float(np.nanmean(cap_l)), 6),
                "paired_field_minus_logistic": cap_ci,
                "n_chains": int(np.sum(~np.isnan(cap_f))),
            },
            "fit_curve": curve,
            "frozen_gates": frozen,
            "environment": {"python": platform.python_version(),
                            "numpy": np.__version__,
                            "platform": platform.platform()},
            "what_this_does_not_say": (
                "Top-decile capture and the fit curve are training-fold "
                "reads of the published halving. They do not open the "
                "official test fold, and they do not replace the "
                "manuscript's ROC sentence."),
        }
    finally:
        z.close()


def write_compatible(doc: dict, json_path: Path, tex_path: Path) -> None:
    payload = json.dumps(doc, indent=2, allow_nan=False) + "\n"
    doc["artifact_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    pub = doc["published_half"]
    top = doc["top_decile"]
    ex = doc["exactness"]
    roc = pub["paired_roc"]
    cap = top["paired_field_minus_logistic"]
    rows = "\n".join(
        f"{c['numer']}/{c['denom']} & {c['n_fit_chains']} & "
        f"{c['field']['roc_auc']:.4f} & {c['logistic']['roc_auc']:.4f} & "
        f"{c['field']['top_decile_capture']:.4f} & "
        f"{c['logistic']['top_decile_capture']:.4f} \\\\"
        for c in doc["fit_curve"])
    lo, hi = cap["ci95"]
    tex_path.write_text("\n".join([
        "% Generated by scaleup.compatible. Do not edit numbers by hand.",
        f"% Artifact sha256 {doc['artifact_sha256']}.",
        "",
        "\\section*{Supplement: what the integer field is for}",
        "\\label{sec:compatible}",
        "",
        "On the published training half the counting field reaches "
        f"{pub['field_gated']:.4f} ROC-AUC and the logistic regression "
        f"{pub['logistic_gated']:.4f}. The paired gap is "
        f"{roc['delta']:+.4f}, interval "
        f"[{roc['ci95'][0]:+.4f}, {roc['ci95'][1]:+.4f}], which contains "
        "zero. That is the manuscript's sentence, reproduced here before "
        "anything else was read. Mean ROC-AUC is not where these two "
        "objects differ, so the rest of this note measures the places "
        "they do.",
        "",
        f"The field score on the pick half reconstructs from "
        f"{ex['n_tables_used']} integer multiplicities with maximum "
        f"absolute error {ex['max_abs_reconstruction_error']:.2e}. "
        f"Those multiplicities are {ex['integer_multiplicity_bytes']} bytes. "
        f"The logistic coefficient vector is "
        f"{ex['logistic_coefficient_bytes']} bytes and does not decompose "
        "into a cell.",
        "",
        f"Top-decile capture, the share of a chain's cryptic residues that "
        f"sit in the top tenth of its own ranking, is "
        f"{top['field_capture']:.4f} for the field and "
        f"{top['logistic_capture']:.4f} for the logistic, a paired gap of "
        f"{cap['delta']:+.4f} with interval [{lo:+.4f}, {hi:+.4f}] "
        f"over {top['n_chains']} chains. The tenth is fixed; it was not "
        "chosen after the fact.",
        "",
        "Refitting both arms on nested subsets of the same fit clusters, "
        "with each arm's gate frozen at the full-fit choice:",
        "",
        "\\begin{tabular}{@{}rrrrrr@{}}",
        "fit & chains & field ROC & logistic ROC & field top & "
        "logistic top \\\\",
        "\\hline",
        rows,
        "\\end{tabular}",
        "",
        "The official test fold was not read.",
        "",
    ]))


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--tex", required=True)
    a = ap.parse_args(argv)
    data = Path(a.data_dir)
    doc = run_compatible(data / "_wide_cache_train.npz",
                         data / MANIFEST_NAME,
                         data / "_digits_train.npy")
    # The reproduction gate already ran. Restate the tolerance check in the
    # artifact so a later reader can see the bound that was applied.
    roc = doc["published_half"]["paired_roc"]
    if abs(doc["published_half"]["field_gated"] - PUBLISHED_FIELD) > REPRODUCE_TOL:
        raise SystemExit("field AUC left the manuscript row")
    if abs(roc["delta"] - PUBLISHED_DELTA) > REPRODUCE_TOL:
        raise SystemExit("paired delta left the manuscript row")
    if roc["n_paired"] != PUBLISHED_N_PAIRED:
        raise SystemExit("paired count left the manuscript row")
    if abs(roc["ci95"][0] - PUBLISHED_CI[0]) > REPRODUCE_TOL:
        raise SystemExit("interval left the manuscript row")
    full = doc["fit_curve"][-1]
    if abs(full["field"]["roc_auc"] - doc["published_half"]["field_gated"]) > 1e-9:
        raise SystemExit("the full-fit curve row is not the published field")
    if abs(full["logistic"]["roc_auc"] - doc["published_half"]["logistic_gated"]) > 1e-9:
        raise SystemExit("the full-fit curve row is not the published logistic")
    write_compatible(doc, Path(a.json), Path(a.tex))
    print(f"wrote {a.json}", flush=True)
    print(f"  roc delta {roc['delta']:+.4f} {roc['ci95']}", flush=True)
    print(f"  top-decile delta "
          f"{doc['top_decile']['paired_field_minus_logistic']['delta']:+.4f} "
          f"{doc['top_decile']['paired_field_minus_logistic']['ci95']}",
          flush=True)
    print(f"  reconstruction {doc['exactness']['max_abs_reconstruction_error']:.2e}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

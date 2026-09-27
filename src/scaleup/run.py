"""Command line: score halves, summarise, check.

Each half is one cluster-disjoint split of the training fold in one
direction: fit the Newton logistic and compile the counting field on the
fit side, score both on the pick side, search the same three gates, and
append one JSON line. Halves are independent processes; a dead machine
loses at most one half, and ``run-all`` resumes from the jsonl.

The first half scored is always the published halving, and the run stops
unless it reproduces the manuscript row. The official test fold is never
opened: every path with "test" in its name is refused.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from .constants import (BOOT_SEED, CACHE_N_ROWS, CLUSTER_BOOT_DRAWS,
                        CLUSTER_BOOT_SEED, DIGIT_MMAP_NAME, FANOUT_CAP,
                        LEVELS, MANIFEST_NAME, N_BOOT, N_WIRES, PAIRING_SEED,
                        PUBLISHED_CI, PUBLISHED_DELTA, PUBLISHED_FIELD,
                        PUBLISHED_LOGIT, PUBLISHED_N_PAIRED, REPRODUCE_TOL,
                        RIDGE, ROUNDS, SCHEMA, SPLIT_SEEDS, WIDTH)
from .data import (check_units_match_cache, cluster_of_manifest, load_cache,
                   load_manifest)
from .digits import load_digits
from .field import (cell_offsets, compile_cells, integer_fanout,
                    partition_tables, score_pick)
from .gates import gate_search
from .logistic import apply_logistic, fit_logistic, standardise_stats
from .metrics import (cluster_bootstrap_ci, paired_bootstrap_ci,
                      per_unit_auc_vector)
from .splits import cluster_halving, pick_vectors, row_masks


def _rounded(vec: list[float]) -> list[float | None]:
    return [None if np.isnan(v) else round(float(v), 6) for v in vec]


def score_half(cache_path: str | Path, manifest_path: str | Path,
               digits_path: str | Path, split_seed: int,
               reverse: bool) -> dict:
    """Score one half. Returns the jsonl row; writes nothing."""
    t0 = time.perf_counter()
    z = load_cache(cache_path)
    try:
        X, y = z["X"], z["y"]
        ctr, n_res = z["ctr"], z["n_res_per"]
        units = [str(u) for u in z["units"]]
        if X.shape[1] != N_WIRES or int(n_res.sum()) != CACHE_N_ROWS:
            raise ValueError(f"unexpected cache geometry {X.shape}")
        doc = load_manifest(manifest_path)
        cluster_of = cluster_of_manifest(doc)
        check_units_match_cache(units, doc)
        D = load_digits(digits_path, (CACHE_N_ROWS, N_WIRES))

        is_fit = cluster_halving(units, cluster_of, split_seed)
        if reverse:
            is_fit = ~is_fit
        fit_rows, pick_rows = row_masks(is_fit, n_res)
        n_pick, ypick, ctr_pick, pick_units = pick_vectors(
            n_res, is_fit, y, ctr, units)

        mean, sd = standardise_stats(X, fit_rows)
        w_log, moves = fit_logistic(X, y, fit_rows, mean, sd, RIDGE)
        s_log = apply_logistic(X, pick_rows, mean, sd, w_log)
        log_raw, log_auc, log_gate, log_gated = gate_search(
            s_log, ctr_pick, n_pick, ypick)
        log_vec = per_unit_auc_vector(log_gated, ypick, n_pick)
        del s_log, log_gated
        gc.collect()

        tables = partition_tables(N_WIRES, WIDTH, ROUNDS, PAIRING_SEED)
        offsets = cell_offsets(tables, LEVELS)
        frac, _tot = compile_cells(D, y, fit_rows, tables, offsets, LEVELS)
        mult = integer_fanout(D, y, fit_rows, tables, offsets, frac,
                              RIDGE, FANOUT_CAP)
        s_field = score_pick(D, pick_rows, tables, offsets, frac, mult)
        fld_raw, fld_auc, fld_gate, fld_gated = gate_search(
            s_field, ctr_pick, n_pick, ypick)
        fld_vec = per_unit_auc_vector(fld_gated, ypick, n_pick)

        log_r = _rounded(log_vec)
        fld_r = _rounded(fld_vec)
        boot_seed = (BOOT_SEED if split_seed == PAIRING_SEED and not reverse
                     else split_seed + 1)
        paired = paired_bootstrap_ci(fld_r, log_r, N_BOOT, boot_seed)
        return {
            "split_seed": int(split_seed),
            "direction": "reverse" if reverse else "forward",
            "n_fit_chains": int(is_fit.sum()),
            "n_pick_chains": int((~is_fit).sum()),
            "n_fit_rows": int(fit_rows.sum()),
            "n_pick_rows": int(pick_rows.sum()),
            "logistic": {"raw": log_raw, "gated": log_auc,
                         "gate": log_gate,
                         "newton_steps": len(moves),
                         "converged": bool(moves[-1] < 1e-6)},
            "field": {"raw": fld_raw, "gated": fld_auc, "gate": fld_gate,
                      "n_tables_used": int((mult != 0).sum())},
            "paired_field_minus_logistic": paired,
            "pick_units": pick_units,
            "per_unit_field": fld_r,
            "per_unit_logistic": log_r,
            "seconds": round(time.perf_counter() - t0, 1),
            "reads_test_fold": False,
        }
    finally:
        z.close()


def assert_published_half(row: dict) -> None:
    """The published halving must reproduce the manuscript row, or stop."""
    if row["split_seed"] != PAIRING_SEED or row["direction"] != "forward":
        return
    log_auc = row["logistic"]["gated"]
    fld_auc = row["field"]["gated"]
    paired = row["paired_field_minus_logistic"]
    ok = (abs(log_auc - PUBLISHED_LOGIT) < REPRODUCE_TOL
          and abs(fld_auc - PUBLISHED_FIELD) < REPRODUCE_TOL
          and abs(paired["delta"] - PUBLISHED_DELTA) < REPRODUCE_TOL
          and abs(paired["ci95"][0] - PUBLISHED_CI[0]) < REPRODUCE_TOL
          and abs(paired["ci95"][1] - PUBLISHED_CI[1]) < REPRODUCE_TOL
          and paired["n_paired"] == PUBLISHED_N_PAIRED)
    if not ok:
        raise SystemExit(
            f"published half did not reproduce (field {fld_auc:.6f}, "
            f"logit {log_auc:.6f}, delta {paired['delta']} "
            f"ci {paired['ci95']}). Refusing to score further halves "
            f"against a different estimator.")
    print("published half reproduces the manuscript row", flush=True)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def cmd_run_half(a) -> int:
    data = Path(a.data_dir)
    out = Path(a.out)
    done = {(r["split_seed"], r["direction"]) for r in _read_jsonl(out)}
    if (a.seed, a.direction) in done:
        print(f"seed {a.seed} {a.direction} already scored")
        return 0
    # The published half goes first even when a later half was requested:
    # nothing else is scored until the estimator is proven.
    if (PAIRING_SEED, "forward") not in done and (
            a.seed != PAIRING_SEED or a.direction != "forward"):
        raise SystemExit("score the published half first (--seed "
                         f"{PAIRING_SEED} --direction forward)")
    row = score_half(data / "_wide_cache_train.npz",
                     data / MANIFEST_NAME,
                     data / DIGIT_MMAP_NAME, a.seed,
                     a.direction == "reverse")
    assert_published_half(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(row, allow_nan=False) + "\n")
    p = row["paired_field_minus_logistic"]
    print(f"seed {a.seed} {a.direction}: field {row['field']['gated']:.4f} "
          f"logit {row['logistic']['gated']:.4f} delta {p['delta']:+.4f} "
          f"ci {p['ci95']} {row['seconds']:.0f}s", flush=True)
    return 0


def _chain_summary(rows: list[dict], cluster_of: dict[str, str]) -> dict:
    acc: dict[str, list[float]] = {}
    for row in rows:
        for unit, f, g in zip(row["pick_units"], row["per_unit_field"],
                              row["per_unit_logistic"]):
            if f is None or g is None:
                continue
            acc.setdefault(unit, []).append(f - g)
    units = sorted(acc)
    mean_gap = {u: float(np.mean(acc[u])) for u in units}
    boot = cluster_bootstrap_ci(units, [mean_gap[u] for u in units],
                                cluster_of, CLUSTER_BOOT_DRAWS,
                                CLUSTER_BOOT_SEED)
    halves = [r["paired_field_minus_logistic"]["delta"] for r in rows]
    return {
        "n_chains_scored": len(units),
        "mean_scores_per_chain": round(
            float(np.mean([len(acc[u]) for u in units])), 2),
        "mean_paired_delta": boot["mean"],
        "cluster_bootstrap_ci95": boot["ci95"],
        "excludes_zero": boot["excludes_zero"],
        "sign": boot["sign"],
        "n_halves": len(rows),
        "n_halves_point_estimate_positive": sum(d > 0 for d in halves),
        "n_halves_ci_excludes_zero_above": sum(
            r["paired_field_minus_logistic"]["excludes_zero"]
            and r["paired_field_minus_logistic"]["delta"] > 0 for r in rows),
        "n_halves_ci_excludes_zero_below": sum(
            r["paired_field_minus_logistic"]["excludes_zero"]
            and r["paired_field_minus_logistic"]["delta"] < 0 for r in rows),
        "half_delta_min": round(float(min(halves)), 6),
        "half_delta_max": round(float(max(halves)), 6),
        "half_delta_median": round(float(np.median(halves)), 6),
    }


def _tex_close(sign: str) -> str:
    if sign == "field":
        return ("At this scale the training-fold gap resolves in favour of "
                "the counting field. It is still a training-fold statement: "
                "the official test fold was not read, and the manuscript's "
                "sentence about the single published half remains the one "
                "that half supports.")
    if sign == "logistic":
        return ("At this scale logistic regression leads on the training "
                "fold, and the interval excludes zero from below. The "
                "manuscript's statement that the two are not separable was "
                "the cautious reading of one half; this read is stronger, "
                "and it goes the other way.")
    return ("The interval still contains zero. Scoring every training chain "
            "across sixteen partitions does not separate the counting field "
            "from logistic regression on these wires. The accuracy, such as "
            "it is, remains in the wires.")


def _write_tex(doc: dict, path: Path) -> None:
    s = doc["chain_summary"]
    lo, hi = s["cluster_bootstrap_ci95"]
    pub = next(r for r in doc["halves_brief"]
               if r["split_seed"] == PAIRING_SEED
               and r["direction"] == "forward")
    lines = [
        "% Generated by scaleup.run summarise; do not edit numbers by hand.",
        f"% Artifact sha256 {doc['artifact_sha256']}.",
        "",
        "\\section*{Supplement: sixteen training partitions against "
        "logistic regression}",
        "\\label{sec:large-scale-logit}",
        "",
        "The readout ladder in the manuscript scores one cluster-disjoint",
        "half of the training fold. On that half an L2-penalised logistic",
        "regression on the same 645 wires reaches "
        f"{pub['logistic_gated']:.4f}, against the published field at "
        f"{pub['field_gated']:.4f}: a paired difference of "
        f"{pub['delta']:+.4f}, whose interval contains zero. This note",
        "repeats the comparison on sixteen partitions, each scored in both",
        "directions, so that the gap is not a property of one halving.",
        "",
        "Nothing in the estimator moves. The wires, the within-chain",
        "quartiles, the sixteen pair partitions, the integer fan-out, the",
        "Newton logistic and the three-gate search are the ones that",
        "produced the manuscript row. The first seed is that halving, and",
        "the run stops if it does not reproduce. The official test fold is",
        "not read. Every half is kept, including those on which logistic",
        "regression is ahead.",
        "",
        f"There are {s['n_chains_scored']} chains with both classes, each",
        f"scored {s['mean_scores_per_chain']:.0f} times. The mean paired",
        "gap, field minus logistic, is "
        f"{s['mean_paired_delta']:+.4f}, with a cluster-bootstrap 95\\%",
        f"interval $[{lo:+.4f}, {hi:+.4f}]$. "
        f"Of the {s['n_halves']} halves, "
        f"{s['n_halves_point_estimate_positive']} have a positive point",
        "estimate. The half-wise gaps run from "
        f"{s['half_delta_min']:+.4f} to {s['half_delta_max']:+.4f}",
        f"(median {s['half_delta_median']:+.4f}).",
        "",
        _tex_close(s["sign"]),
        "",
    ]
    path.write_text("\n".join(lines))


def cmd_summarise(a) -> int:
    out = Path(a.out)
    rows = _read_jsonl(Path(a.jsonl))
    want = {(s, d) for s in SPLIT_SEEDS for d in ("forward", "reverse")}
    got = {(r["split_seed"], r["direction"]) for r in rows}
    if got != want:
        missing = sorted(want - got)
        raise SystemExit(f"refusing to summarise an incomplete run; "
                         f"{len(missing)} halves missing, first {missing[:3]}")
    doc_manifest = load_manifest(Path(a.data_dir) / MANIFEST_NAME)
    brief = []
    for r in rows:
        p = r["paired_field_minus_logistic"]
        brief.append({
            "split_seed": r["split_seed"], "direction": r["direction"],
            "n_paired": p["n_paired"],
            "field_gated": r["field"]["gated"],
            "logistic_gated": r["logistic"]["gated"],
            "delta": p["delta"], "ci95": p["ci95"],
            "excludes_zero": p["excludes_zero"],
            "field_gate": r["field"]["gate"],
            "logistic_gate": r["logistic"]["gate"],
            "seconds": r["seconds"],
        })
    brief.sort(key=lambda r: (r["split_seed"], r["direction"]))
    chain = _chain_summary(rows, cluster_of_manifest(doc_manifest))
    import platform
    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "question": "whether the training-fold gap between the published "
                    "counting field and logistic regression on the same 645 "
                    "wires is a property of one halving",
        "protocol": {
            "cache_sha256": ("fef234d4d7b356d585df74ea90e55e1cc6dcb958"
                             "675c2b7fbcd75b352d9d7374"),
            "n_split_seeds": len(SPLIT_SEEDS),
            "split_seeds": list(SPLIT_SEEDS),
            "directions": ["forward", "reverse"],
            "pairing_seed": PAIRING_SEED,
            "levels": LEVELS, "fanout_cap": FANOUT_CAP, "ridge": RIDGE,
            "rounds": ROUNDS, "width": WIDTH,
            "gates_searched": [{"radius": r, "weight": w}
                               for r, w in
                               ((14.0, 1.0), (18.0, 0.5), (18.0, 1.0))],
            "cluster_bootstrap": {"draws": CLUSTER_BOOT_DRAWS,
                                  "seed": CLUSTER_BOOT_SEED,
                                  "unit": "MMseqs cluster"},
            "negative_halves_retained": True,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "published_half_reproduced": True,
        "halves_brief": brief,
        "chain_summary": chain,
        "what_this_does_not_say": (
            "A resolved gap here is a statement about the training fold. "
            "It does not open, and must not be quoted as, the official "
            "test fold."),
    }
    payload = json.dumps(doc, indent=2, allow_nan=False) + "\n"
    doc["artifact_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    _write_tex(doc, Path(a.tex))
    print(f"wrote {out}", flush=True)
    print(f"chain delta {chain['mean_paired_delta']:+.4f} "
          f"ci {chain['cluster_bootstrap_ci95']} sign={chain['sign']}",
          flush=True)
    return 0


def cmd_check(a) -> int:
    doc = json.loads(Path(a.json).read_text())
    bad = []
    if doc.get("schema") != SCHEMA:
        bad.append("schema")
    if doc.get("clinical_grade"):
        bad.append("clinical_grade must stay false")
    if doc.get("reads_test_fold"):
        bad.append("test fold was read")
    if doc.get("protocol", {}).get("split_seeds") != list(SPLIT_SEEDS):
        bad.append("split seeds differ from the predeclared list")
    halves = doc.get("halves_brief") or []
    if len(halves) != 2 * len(SPLIT_SEEDS):
        bad.append(f"expected {2 * len(SPLIT_SEEDS)} halves, "
                   f"got {len(halves)}")
    rows = _read_jsonl(Path(a.jsonl))
    jsonl_neg = any(r["paired_field_minus_logistic"]["delta"] < 0
                    for r in rows)
    if jsonl_neg and not any(h["delta"] < 0 for h in halves):
        bad.append("jsonl has a negative half the summary dropped")
    if not doc.get("published_half_reproduced"):
        bad.append("published half was not reproduced")
    pub = [h for h in halves if h["split_seed"] == PAIRING_SEED
           and h["direction"] == "forward"]
    if len(pub) != 1:
        bad.append("published half missing from the summary")
    elif abs(pub[0]["logistic_gated"] - PUBLISHED_LOGIT) > REPRODUCE_TOL:
        bad.append("published logistic AUC drifted")
    if bad:
        print("FAIL")
        for b in bad:
            print(" -", b)
        return 1
    s = doc["chain_summary"]
    print(f"OK {s['mean_paired_delta']:+.4f} "
          f"{s['cluster_bootstrap_ci95']} sign={s['sign']} "
          f"halves={s['n_halves']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run-half", help="score one half")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--direction", choices=("forward", "reverse"),
                   required=True)
    p = sub.add_parser("run-all", help="score every half, then summarise")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--tex", required=True)
    p = sub.add_parser("summarise", help="jsonl to summary json + tex")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--tex", required=True)
    p = sub.add_parser("check", help="verify a summary")
    p.add_argument("--jsonl", required=True)
    p.add_argument("--json", required=True)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "run-half":
        return cmd_run_half(a)
    if a.cmd == "run-all":
        for seed in SPLIT_SEEDS:
            for direction in ("forward", "reverse"):
                sub = argparse.Namespace(data_dir=a.data_dir, out=a.out,
                                         seed=seed, direction=direction)
                rc = cmd_run_half(sub)
                if rc:
                    return rc
        fin = argparse.Namespace(data_dir=a.data_dir, jsonl=a.out,
                                 json=a.json, tex=a.tex)
        return cmd_summarise(fin)
    if a.cmd == "summarise":
        return cmd_summarise(a)
    if a.cmd == "check":
        return cmd_check(a)
    raise SystemExit("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())

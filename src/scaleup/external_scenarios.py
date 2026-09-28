"""Scenarios for the post-hoc external logistic comparison.

Every cut here is taken from the difficulty note that predates this
logistic: under ten cryptic residues, more than twenty-two, and chain
length against Set A's published median of 336. Set B and Set C are the
sets already frozen. Nothing is chosen because its interval cleared zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .metrics import paired_bootstrap_ci

BOOT_SEED = 20260929
DRAWS = 4000
SCHEMA = "geoaudit.scaleup.external_logistic_scenarios.v1"


def _paired(rows: list[dict]) -> dict:
    if len(rows) < 5:
        return {"n_paired": len(rows), "too_small": True}
    acf = [r["acf_roc"] for r in rows]
    log = [r["logistic_roc"] for r in rows]
    paired = paired_bootstrap_ci(acf, log, DRAWS, BOOT_SEED)
    return {"n_paired": paired["n_paired"],
            "acf_roc": round(float(np.mean(acf)), 6),
            "logistic_roc": round(float(np.mean(log)), 6),
            "acf_minus_logistic": paired,
            "n_acf_ahead": int(sum(a > b for a, b in zip(acf, log)))}


def _middle(deltas: np.ndarray) -> dict:
    """The functional in tools/preregistered_read.py: drop round(0.2 n) each side."""
    k = int(round(0.20 * len(deltas)))
    ordered = np.sort(deltas)
    middle = ordered[k:len(ordered) - k]
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(deltas), size=(DRAWS, len(deltas)))
    stats = np.empty(DRAWS)
    for i, draw in enumerate(idx):
        s = np.sort(deltas[draw])
        stats[i] = s[k:len(s) - k].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"n": int(len(deltas)), "n_trimmed_each_side": k,
            "middle_60_mean": round(float(middle.mean()), 6),
            "ci95": [round(float(lo), 6), round(float(hi), 6)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "worst_losses_mean": round(float(ordered[:k].mean()), 6),
            "best_wins_mean": round(float(ordered[-k:].mean()), 6)}


def _scene(row: dict) -> dict:
    return {"unit": row["unit"], "pool": row["pool"], "set": row.get("set"),
            "n_residues": row["n_residues"], "n_positive": row["n_positive"],
            "acf_roc": row["acf_roc"], "logistic_roc": row["logistic_roc"],
            "acf_minus_logistic": round(row["acf_roc"] - row["logistic_roc"], 6)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    doc = json.loads(Path(a.src).read_text())
    rows = []
    for pool, block in doc["sets"].items():
        for chain in block["chains"]:
            chain = dict(chain)
            chain["pool"] = pool
            rows.append(chain)
    cuts = {
        "all_external": rows,
        "set_a": [r for r in rows if r["pool"] == "set_a"],
        "set_bc": [r for r in rows if r["pool"] == "set_bc"],
        "set_b": [r for r in rows if r.get("set") == "set_b"],
        "set_c": [r for r in rows if r.get("set") == "set_c"],
        "under_ten_cryptic": [r for r in rows if r["n_positive"] < 10],
        "ten_to_twenty_two_cryptic": [r for r in rows if 10 <= r["n_positive"] <= 22],
        "over_twenty_two_cryptic": [r for r in rows if r["n_positive"] > 22],
        "length_at_most_336": [r for r in rows if r["n_residues"] <= 336],
        "length_above_336": [r for r in rows if r["n_residues"] > 336],
    }
    ranked = sorted(rows, key=lambda r: r["acf_roc"] - r["logistic_roc"])
    out = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "post_hoc": True,
        "cuts_named_before_this_logistic": (
            "under ten and over twenty-two cryptic residues, and length 336, "
            "are the cuts in EXTERNAL_SET_DIFFICULTY.json. Set B and Set C "
            "are the frozen cryo-EM sets. No cut was added after seeing "
            "which interval excluded zero."),
        "mean_paired": {name: _paired(sub) for name, sub in cuts.items()},
        "middle_60": {
            name: _middle(np.array([r["acf_roc"] - r["logistic_roc"] for r in sub]))
            for name, sub in cuts.items() if len(sub) >= 5
        },
        "largest_acf_leads": [_scene(r) for r in ranked[-5:]],
        "largest_logistic_leads": [_scene(r) for r in ranked[:5]],
        "what_is_proved": (
            "No cut in this file has a mean paired interval that excludes "
            "zero. Individual chains differ by several tenths in both "
            "directions. That is a case list, not a proof that the counting "
            "field ranks cryptic residues better than logistic regression."),
    }
    payload = json.dumps(out, indent=2, allow_nan=False) + "\n"
    out["artifact_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    dest = Path(a.json)
    dest.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"wrote {dest}")
    for name, block in out["mean_paired"].items():
        gap = block.get("acf_minus_logistic") or {}
        if not gap:
            print(f"  {name}: n={block['n_paired']} too small")
            continue
        print(f"  {name}: n={gap['n_paired']} delta {gap['delta']:+.4f} "
              f"{gap['ci95']} excludes0={gap['excludes_zero']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

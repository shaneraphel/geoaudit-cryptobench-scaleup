"""Every public interface is invoked here, on synthetic data.

The point is the one the frozen repo's history demands: no function this
supplement depends on may go unwired. Each test calls the real entry point
and checks it against a slow implementation short enough to read.
"""
import json
import sys

import numpy as np
import pytest

from scaleup import constants as C
from scaleup import data as D
from scaleup import digits as DG
from scaleup import field as F
from scaleup import gates as G
from scaleup import logistic as LG
from scaleup import metrics as M
from scaleup import run as R
from scaleup import splits as S


def test_constants_predeclared():
    assert len(C.SPLIT_SEEDS) == 16
    assert C.SPLIT_SEEDS[0] == C.PAIRING_SEED == 20260725
    assert C.BOOT_SEED != C.PAIRING_SEED
    assert C.GATES == ((14.0, 1.0), (18.0, 0.5), (18.0, 1.0))


def test_roc_auc_known_values():
    assert M.roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(
        0.75)
    assert M.roc_auc([1, 2, 3, 4], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert M.roc_auc([4, 3, 2, 1], [0, 0, 1, 1]) == pytest.approx(0.0)
    assert M.roc_auc([1, 1, 1], [1, 1, 1]) is None
    # ties average ranks: one tied pair scores half
    assert M.roc_auc([0.5, 0.5, 0.0, 1.0], [0, 1, 0, 1]) == pytest.approx(
        0.875)


def test_per_unit_vector_skips_single_class():
    score = np.array([0.1, 0.9, 0.2, 0.3, 0.8])
    y = np.array([0, 1, 1, 0, 1])
    vec = M.per_unit_auc_vector(score, y, [2, 3])
    assert vec[0] == pytest.approx(1.0)
    assert M.per_unit_auc(score, y, [2, 3]) == pytest.approx(
        float(np.nanmean(vec)))
    vec2 = M.per_unit_auc_vector(score, np.array([1, 1, 1, 0, 1]), [2, 3])
    assert np.isnan(vec2[0])


def test_paired_bootstrap_deterministic_and_sane():
    rng = np.random.default_rng(0)
    a = list(0.6 + 0.1 * rng.standard_normal(50))
    b = list(0.5 + 0.1 * rng.standard_normal(50))
    c1 = M.paired_bootstrap_ci(a, b, 4000, 99)
    c2 = M.paired_bootstrap_ci(a, b, 4000, 99)
    assert c1 == c2
    assert c1["n_paired"] == 50
    assert c1["delta"] == pytest.approx(float(np.mean(a) - np.mean(b)),
                                        abs=1e-6)
    assert c1["excludes_zero"]


def test_cluster_bootstrap_signs():
    units = [f"u{i}" for i in range(8)]
    cof = {u: f"c{i // 2}" for i, u in enumerate(units)}
    pos = M.cluster_bootstrap_ci(units, [0.05] * 8, cof, 500, 7)
    assert pos["sign"] == "field" and pos["excludes_zero"]
    neg = M.cluster_bootstrap_ci(units, [-0.05] * 8, cof, 500, 7)
    assert neg["sign"] == "logistic" and neg["excludes_zero"]
    mix = M.cluster_bootstrap_ci(units,
                                 [0.2, -0.2, 0.2, -0.2,
                                  0.2, -0.2, 0.2, -0.2],
                                 cof, 500, 7)
    assert mix["sign"] == "unresolved" and not mix["excludes_zero"]


def test_halving_disjoint_and_covering():
    units = [f"p{i}_A" for i in range(20)]
    cof = {u: f"cl{i % 10}" for i, u in enumerate(units)}
    is_fit = S.cluster_halving(units, cof, 12345)
    assert is_fit.dtype == bool and len(is_fit) == 20
    # cluster-disjoint: no cluster on both sides
    fit_cl = {cof[u] for u, f in zip(units, is_fit) if f}
    pick_cl = {cof[u] for u, f in zip(units, is_fit) if not f}
    assert fit_cl.isdisjoint(pick_cl)
    assert len(fit_cl) == 5 and len(pick_cl) == 5
    with pytest.raises(ValueError):
        S.cluster_halving(units + ["ghost_X"], cof, 1)


def test_row_masks_expand_chains():
    n_res = [3, 5, 2]
    fit, pick = S.row_masks(np.array([True, False, True]), n_res)
    assert fit.tolist() == [True] * 3 + [False] * 5 + [True] * 2
    assert (fit ^ pick).all()
    n_pick, ypick, ctr_pick, punits = S.pick_vectors(
        n_res, np.array([True, False, True]),
        np.arange(10), np.zeros((10, 3)), ["a", "b", "c"])
    assert n_pick.tolist() == [5] and punits == ["b"]
    assert ypick.tolist() == [3, 4, 5, 6, 7]


def test_rank_digits_ties_share_midrank():
    assert DG.rank_digits_vector(np.array([9.0]), 4).tolist() == [0]
    out = DG.rank_digits_vector(np.array([1.0, 1.0, 5.0, 9.0]), 4)
    # tied pair shares the mid-rank band
    assert out[0] == out[1] and out.tolist() == [0, 0, 2, 3]


def test_chain_digits_matches_vector(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((9, 5))
    D = DG.chain_digits_matrix(X, [4, 5], 4)
    assert D.dtype == np.int8 and D.min() >= 0 and D.max() <= 3
    for j in range(5):
        assert (D[:4, j] == DG.rank_digits_vector(X[:4, j], 4)).all()
        assert (D[4:, j] == DG.rank_digits_vector(X[4:, j], 4)).all()
    # digit file round trip through a memmap
    p = tmp_path / "d.npy"
    m = np.lib.format.open_memmap(p, mode="w+", dtype=np.int8,
                                  shape=D.shape)
    m[:] = D
    m.flush()
    del m
    assert (DG.load_digits(p, D.shape) == D).all()
    with pytest.raises(ValueError):
        DG.load_digits(p, (D.shape[0], D.shape[1] + 1))


def test_data_refuses_wrong_inputs(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert D.sha256_of_file(p) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    with pytest.raises(ValueError):
        D.load_cache(p)
    with pytest.raises(ValueError):
        D.load_manifest(p)
    doc = {"entries": [{"pdb": "1a", "chain": "A", "cluster_id": 3}]}
    assert D.cluster_of_manifest(doc) == {"1a_A": "3"}
    D.check_units_match_cache(["1a_A"], doc)
    with pytest.raises(ValueError):
        D.check_units_match_cache(["1a_B"], doc)


def test_partitions_cover_each_wire_once():
    tables = F.partition_tables(11, 2, 3, 20260725)
    assert len(tables) == 3 * 5  # 11 wires -> five pairs + one dropped single
    for r in range(3):
        seen = [w for t in tables[r * 5:(r + 1) * 5] for w in t]
        assert sorted(seen) != sorted(set(seen)) or len(seen) == 10
        assert len(set(seen)) == 10  # each wire at most once per round
    off = F.cell_offsets([[0, 1], [2, 3, 4]], 4)
    assert off.tolist() == [0, 16, 16 + 64]


def test_field_matches_naive_loop():
    rng = np.random.default_rng(2)
    n, p = 300, 9
    Dm = rng.integers(0, 4, size=(n, p)).astype(np.int8)
    y = rng.integers(0, 2, size=n)
    fit = np.zeros(n, dtype=bool)
    fit[:200] = True
    tables = F.partition_tables(p, 2, 2, 20260725)
    offsets = F.cell_offsets(tables, 4)
    frac, tot = F.compile_cells(Dm, y, fit, tables, offsets, 4, block=64)
    # naive rates
    K = len(tables)
    slow_tot = np.zeros(int(offsets[-1]), dtype=np.int64)
    slow_pos = np.zeros(int(offsets[-1]))
    for i in np.flatnonzero(fit):
        for k, (a, b) in enumerate(tables):
            addr = offsets[k] + int(Dm[i, a]) + 4 * int(Dm[i, b])
            slow_tot[addr] += 1
            slow_pos[addr] += y[i]
    rate = y[fit].mean()
    slow_frac = np.where(slow_tot > 0, slow_pos / np.maximum(slow_tot, 1),
                         rate)
    assert np.array_equal(tot, slow_tot)
    assert np.allclose(frac, slow_frac)
    # naive fan-out on the same rates
    V = np.empty((200, K))
    for r, i in enumerate(np.flatnonzero(fit)):
        for k, (a, b) in enumerate(tables):
            V[r, k] = slow_frac[offsets[k] + int(Dm[i, a])
                                + 4 * int(Dm[i, b])]
    yf = y[fit]
    mu1, mu0 = V[yf == 1].mean(0), V[yf == 0].mean(0)
    C_ = np.where((yf == 1)[:, None], V - mu1, V - mu0)
    S_ = C_.T @ C_ / (len(yf) - 2)
    S_.flat[::K + 1] += 0.03 * np.trace(S_) / K + 1e-12
    w = np.linalg.solve(S_, mu1 - mu0)
    want = np.round(w / np.abs(w).max() * 32).astype(np.int64)
    assert np.array_equal(
        F.integer_fanout(Dm, y, fit, tables, offsets, frac, 0.03, 32,
                         block=64), want)
    # naive scores
    pick = ~fit
    got = F.score_pick(Dm, pick, tables, offsets, frac, want, block=64)
    slow_s = np.array([
        sum(slow_frac[offsets[k] + int(Dm[i, a]) + 4 * int(Dm[i, b])]
            * want[k] for k, (a, b) in enumerate(tables))
        for i in np.flatnonzero(pick)])
    assert np.allclose(got, slow_s)


def test_logistic_fits_and_matches_manual():
    rng = np.random.default_rng(3)
    n, p = 400, 6
    X = rng.standard_normal((n, p))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) * 0.5 > 0
         ).astype(np.int64)
    fit = np.zeros(n, dtype=bool)
    fit[:300] = True
    mean, sd = LG.standardise_stats(X, fit, block=64)
    assert np.allclose(mean, X[fit].mean(0))
    assert np.allclose(sd, X[fit].std(0, ddof=1))
    w, moves = LG.fit_logistic(X, y, fit, mean, sd, 0.03, block=64)
    assert moves[-1] < 1e-6  # converged
    s = LG.apply_logistic(X, ~fit, mean, sd, w, block=64)
    z = (X[~fit] - mean) / sd
    assert np.allclose(s, z @ w[:-1] + w[-1])
    assert M.roc_auc(s, y[~fit]) > 0.85


def test_gates_match_naive_and_search_picks_best():
    rng = np.random.default_rng(4)
    n = 60
    ctr = rng.standard_normal((n, 3)) * 10
    s = rng.standard_normal(n)
    g = G.neighbourhood_mean(s, ctr, [n], 14.0)
    slow = np.array([
        s[np.sum((ctr - ctr[i]) ** 2, 1) <= 14.0 ** 2].mean()
        for i in range(n)])
    assert np.allclose(g, slow)
    gm = G.spread_matched_gate(s, ctr, [n], 14.0, 0.5)
    assert np.std(gm - s) == pytest.approx(0.5 * np.std(s))
    y = (rng.random(n) > 0.7).astype(int)
    raw, best, name, gated = G.gate_search(s, ctr, [n], y)
    assert best >= raw - 1e-12
    assert name in ("r14 w1.0", "r18 w0.5", "r18 w1.0")
    assert M.per_unit_auc(gated, y, [n]) == pytest.approx(best)


def test_published_half_gate_passes_and_fails():
    good = {"split_seed": 20260725, "direction": "forward",
            "logistic": {"gated": C.PUBLISHED_LOGIT},
            "field": {"gated": C.PUBLISHED_FIELD},
            "paired_field_minus_logistic": {
                "delta": C.PUBLISHED_DELTA,
                "ci95": list(C.PUBLISHED_CI), "n_paired": 384}}
    R.assert_published_half(good)  # must not raise
    bad = json.loads(json.dumps(good))
    bad["paired_field_minus_logistic"]["delta"] = 0.999
    with pytest.raises(SystemExit):
        R.assert_published_half(bad)
    # other halves pass through untouched
    other = dict(good, split_seed=1)
    R.assert_published_half(other)


def test_no_torch_anywhere():
    for mod in list(sys.modules):
        assert mod.split(".")[0] != "torch", "torch got imported"

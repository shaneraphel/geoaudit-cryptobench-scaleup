# Large-scale field-vs-logistic readout comparison

A computationally reproducible supplement to the frozen paper
[`shaneraphel/geoaudit-cryptobench`](https://github.com/shaneraphel/geoaudit-cryptobench)
(which this repository never modifies): is the training-fold gap between
the counting field and logistic regression on the same 645 wires a
property of one halving?

- Sixteen predeclared cluster-disjoint partitions, each scored both ways.
- Same wires, same quartiles, same pair tables, same integer fan-out,
  same Newton logistic, same three-gate search. Nothing retuned.
- First half must reproduce the manuscript row or the run stops.
- Every half kept, including ones where logistic regression is ahead.

`clinical_grade=false`. No torch, no sklearn, no deep learning: numpy
only. The test suite fails if `torch` is ever imported.

## Reproduce

```bash
make selftest    # every public interface, invoked on synthetic data
make compatible  # reproduce the manuscript row, then exactness, top-decile, fit curve
make data       # pinned cache + manifest, sha256-verified
make digits     # within-chain quartiles, streamed to a memmap
make run        # 32 halves, resume-safe jsonl, summary json + tex
make check      # verify the summary (seeds, signs, published row)
```

Inputs are pinned by sha256 (see `docs/PROTOCOL.md`). The 465 MB
training cache is fetched from the
[v0.1-data](https://github.com/shaneraphel/geoaudit-cryptobench-scaleup/releases/tag/v0.1-data)
release asset; the manifest comes from the frozen repo at a pinned
commit. Results land in `results/`, the supplement section in
`paper/supplement_large_scale_logit.tex`.

## Layout

| path | what |
|---|---|
| `src/scaleup/constants.py` | every constant, one place |
| `src/scaleup/data.py` | pinned mmap loading + verification |
| `src/scaleup/splits.py` | cluster-disjoint halvings |
| `src/scaleup/digits.py` | within-chain quartile banding |
| `src/scaleup/field.py` | pair tables, cell rates, integer fan-out |
| `src/scaleup/logistic.py` | Newton L2 logistic, chunked |
| `src/scaleup/gates.py` | spread-matched spatial gate + search |
| `src/scaleup/metrics.py` | ROC-AUC, paired and cluster bootstraps |
| `src/scaleup/run.py` | CLI: run-half, run-all, summarise, check |
| `src/scaleup/compatible.py` | exact integer reconstruction, top-decile capture, nested fit size |
| `tests/test_interfaces.py` | each interface invoked against a naive loop |
| `docs/PROTOCOL.md` | predeclared protocol (written before the run) |

## License

Apache-2.0, matching the frozen paper repository.

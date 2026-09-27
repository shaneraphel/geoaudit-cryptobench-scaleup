# Predeclared protocol

Written before any half was scored in this repository. The frozen paper's
readout ladder scores one cluster-disjoint half (seed 20260725, 384 chains)
and finds the counting field ahead of L2 logistic regression on the same
645 wires by +0.0038 ROC-AUC, interval containing zero. This supplement
asks whether that gap is a property of one halving.

## Fixed before the run

- Inputs: training wide cache sha256 `fef234d4…d7374` (234838 rows,
  645 wires, 770 chains) and manifest sha256 `c1f01970…bd9d0`. Any other
  file is refused. The official test fold is never opened.
- Estimators: within-chain quartiles, sixteen pair partitions at pairing
  seed 20260725, integer fan-out (cap 32, ridge 0.03), Newton logistic at
  the same ridge, same three-gate search. Nothing moves.
- Splits: sixteen seeds, `20260725 + 104729·k`, each scored forward and
  reverse, so every training chain is scored once per seed. The list is
  in `src/scaleup/constants.py` and is not edited after the signs.
- Order: the published half is scored first and must reproduce the
  manuscript row (field 0.804536, logistic 0.800773, delta +0.003762,
  CI [-0.005702, +0.012504], n 384) within 5e-5, or the run stops.
- Summary: mean, over chains, of each chain's mean paired gap, with a
  cluster bootstrap (4000 draws, seed 20260927). Halves are not treated
  as new proteins. Every half is kept, including negative ones; the
  checker fails if the jsonl holds a negative half the summary drops.

## What this cannot say

A resolved gap is a training-fold statement. It does not open, and must
not be quoted as, the official test fold. An unresolved gap means the
manuscript's sentence stands at this larger scale as well.

# Preregistration: within-class edge counts against the logistic ceiling

Written before this repository scores `e_charged@10` or `e_polar@10`.
The definitions are the ones already pinned for the empty graph, the
path on three vertices, and `K_3`: those three graphs all have three
vertices of one class and split as 0, 2, 3 edges. A column monotone in
the class count `n_c` is the same weak order as `n_c`. An edge count is
not that order.

## Columns

Radius 10 Å on residue centroids, self included in the ball, edges the
same predicate with the self-loop removed. Classes, unchanged: charged
`{ARG,ASP,GLU,HIS,LYS}`, polar `{ASN,CYS,GLN,SER,THR,TYR}`. Hydrophobic
is recomputed only to reapply its withdrawal rule; it is not a candidate.

Controls in the same matrix: `ctrl~n_res@10` and `ctrl~n_c` of the same
class. `e <= C(n_c, 2)` is an identity, checked, not a result.

## Withdrawal, fixed before the Spearman is computed

A column is withdrawn when its Spearman against its own `n_c` has
absolute value at least 0.95. That column is the class count relabelled.
The earlier census withdrew `e_hphob@10` on this rule (0.9794) and kept
`e_charged@10` (0.9217) and `e_polar@10` (0.9419). Those three numbers
are not copied. This cache has 234838 rows, not 235148, so the rule is
applied to the matrix this run builds.

## The ceiling question

On the published training half only (seed 20260725), the same Newton
logistic and the same three-gate search:

- baseline: the 645 wires, which must reproduce gated ROC-AUC 0.800773
  within 5e-5 or the run stops
- edges: baseline plus every column that survived withdrawal
- edges and controls: that set plus `ctrl~n_res@10` and the matching
  `ctrl~n_c`

Both augmented arms are reported. The paired gap is augmented minus
baseline, chains, 4000 draws, seed 20260726. The earlier preregistration
predicted these columns would not carry a residual past `n_c`. A gap
whose interval excludes zero from above would be the informative
surprise. An interval that contains zero means the ceiling did not move.

The official test fold is not opened. Nothing is attached to a deployed
field.

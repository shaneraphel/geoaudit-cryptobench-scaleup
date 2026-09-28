# Post-hoc external logistic comparison

This is not a pre-registered confirmation. Sets A and B+C had already been
scored by the counting field before this file existed. The only new
object is a logistic regression that was not allowed to see them.

## Fit, training fold only

Standardisation and the Newton logistic are fit on every residue of the
pinned training wide cache (sha256 `fef234d4…d7374`, 770 chains). The
spatial gate is the one the training-half search already chose for
logistic, radius 18 Å and weight 0.5. It is not searched again on the
external sets. The propensity table is the one stored in the compiled
field, which matches the training cache.

## External wires

Each external receptor is passed through the same 645-wire builder the
compiled field uses, with that frozen propensity. Before any logistic
number is kept, the field's own score on the first Set A chain is
recomputed and required to match the archived counting-field scores.
A mismatch stops the run.

## Report

Set A, then Sets B and C pooled. For each chain, ROC-AUC of the archived
counting-field scores against this logistic, on the same residues.
The paired difference is counting field minus logistic, 4000 chain
resamples, seed 20260929. Chains with a single class are omitted. The
official CryptoBench test fold is not read.

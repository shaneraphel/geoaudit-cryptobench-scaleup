# Preregistration: a downloaded apo bridge

Written before any chain from this download is scored. This is not another
read of the 645-wire training half. That comparison is finished.

## What is downloaded

AHoJ-DB query results at `https://apoholo.cz/api/db`, then the apo PDB
from RCSB. The UniProt list, the caps and the filters below are fixed
here. A chain is kept only when all of these hold:

- the entry did not fail
- the ligand is not in {HOH, DOD, WAT, UNK, ABA, MPD, GOL, SO4, PO4, ZN, MG, CA, NA, CL}
- X-ray, resolution at most 2.5 Å
- apo assignment, pocket RMSD greater than 2.0 Å
- at least 8 mapped binding residues
- chain length between 40 and 800
- the PDB id is in neither the pinned CryptoBench training manifest nor the OSF test-fold file `test.json`

The test-fold file is downloaded only so those PDB ids can be excluded.
It is not scored. At most 8 entries are taken per UniProt accession, and
at most one chain per PDB id, until 480 chains are kept. The accession
order is the list in `scaleup.ahoj_bridge.UNIPROTS`.

## The tape

Each kept chain becomes a residue tape: amino-acid code, centroid,
cryptic-residue label. The same tape feeds every arm. No learned
encoder is fit.

Columns, all at 10 Å: `n_res`, `n_charged`, `n_polar`, `n_hphob`,
`e_charged`, `e_polar`. `e_hphob` is not a column; it already failed
the 0.95 withdrawal against its own class count.

## The question

On one cluster-disjoint halving of these chains, clusters equal to
UniProt accessions, seed 20260928:

- an L2 logistic regression on the six columns
- the integer pair field on the same six columns (4-level within-chain
  digits, width 2, four rounds, cap 32, ridge 0.03)
- a logistic regression on the four count columns alone, edges removed

The paired gap is field minus full logistic, and edges-logistic minus
counts-logistic. Both intervals are reported. A gap whose interval
contains zero is a tie at this larger sample, not a miss to be
re-queried. The official CryptoBench test fold is not scored.

"""Constants for the large-scale field-vs-logistic comparison.

Every value here is either the frozen paper's published configuration or a
predeclared choice of this supplement. Nothing is tuned after a half is
scored; tools that need a constant import it from here so a drift has
exactly one place to hide.
"""
from __future__ import annotations

SCHEMA = "geoaudit.scaleup.logit_field.v1"

# Published configuration of the counting field (frozen paper).
PAIRING_SEED = 20260725
LEVELS = 4
RANKING = "within-chain"
FANOUT_CAP = 32
RIDGE = 0.03
ROUNDS = 16
WIDTH = 2
N_WIRES = 645
N_TRAIN_UNITS = 770
GATES: tuple[tuple[float, float], ...] = ((14.0, 1.0), (18.0, 0.5), (18.0, 1.0))

# Logistic readout: Newton on standardised wires, same relative ridge.
NEWTON_STEPS = 25
NEWTON_TOL = 1e-6

# Resampling. BOOT_SEED differs from the split seed on purpose.
N_BOOT = 4000
BOOT_SEED = PAIRING_SEED + 1

# Sixteen predeclared split seeds. The first is the published halving;
# 104729 is an offset, not a tuned stride, and the list is never edited
# after the signs are known.
N_SPLIT_SEEDS = 16
SPLIT_OFFSET = 104729
SPLIT_SEEDS: tuple[int, ...] = tuple(
    PAIRING_SEED + SPLIT_OFFSET * k for k in range(N_SPLIT_SEEDS)
)
CLUSTER_BOOT_SEED = 20260927
CLUSTER_BOOT_DRAWS = 4000

# Pinned inputs. The run refuses any cache whose sha256 differs.
CACHE_NAME = "_wide_cache_train.npz"
CACHE_SHA256 = (
    "fef234d4d7b356d585df74ea90e55e1cc6dcb958675c2b7fbcd75b352d9d7374"
)
CACHE_N_ROWS = 234838
MANIFEST_NAME = "train_manifest.json"
MANIFEST_SHA256 = (
    "c1f01970f7b7bf596785b8364c425827b2af36f05cedb6b76f9a23393c6bd9d0"
)
MANIFEST_COMMIT = "7245cfb279b0aa9587c2a914e52b6cd077868d1c"

# The manuscript row the first half must reproduce (full precision).
PUBLISHED_LOGIT = 0.8007733428798982
PUBLISHED_FIELD = 0.8045356538898498
PUBLISHED_DELTA = 0.003762
PUBLISHED_CI = (-0.005702, 0.012504)
PUBLISHED_N_PAIRED = 384
REPRODUCE_TOL = 5e-5

# Row block for the Gram and the Newton Hessian. 8192 is table_bank.BLOCK
# in the frozen paper; a different block changes float64 accumulation order
# and the published half stops reproducing. The wire matrix itself stays
# memory-mapped, so the resident set is one block, not the whole cache.
BLOCK_ROWS = 8192
GATE_CHUNK = 512
DIGIT_MMAP_NAME = "_digits_train.npy"

# Compatible instruments, declared before they are scored. They do not
# replace the ROC ladder and they are not searched.
# Top fraction of each chain's ranking. One cut, not a grid.
TOP_FRACTION = 0.10
# Nested fit-cluster fractions of the published fit half: 1/8, 2/8, 4/8, 8/8.
# Smaller fractions are prefixes of one seeded shuffle, so each larger fit
# contains the smaller one.
FIT_FRACTION_DENOM = 8
FIT_FRACTION_NUMS: tuple[int, ...] = (1, 2, 4, 8)
SUBSAMPLE_SEED = PAIRING_SEED + 17
SCHEMA_COMPATIBLE = "geoaudit.scaleup.compatible.v1"

"""Self-contained large-scale readout comparison.

Counting field vs L2 logistic regression on the same 645 wires, sixteen
cluster-disjoint training partitions, each scored both ways. Numpy only.
No torch, no sklearn, no import from the frozen paper repository.
"""
from __future__ import annotations

from . import (compatible, constants, data, digits, field, gates, logistic,
                   metrics, splits)

__version__ = "0.2.0"
__all__ = ["compatible", "constants", "data", "digits", "field", "gates",
           "logistic", "metrics", "splits"]

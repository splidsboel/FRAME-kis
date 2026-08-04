"""FRAME — a Filtered-ANN benchmark suite for Known-Item Search over V3C.

Adapter + shared-harness design (see the vault: Benchmark suite planning.md,
Architecture section). Add a system by implementing a VectorDBAdapter; the Runner
and Analyzer are shared.
"""

from .core.adapter import VectorDBAdapter
from .core.analyzer import Analyzer
from .core.dataset import Dataset
from .core.profile import Profiler, SelectivityProfile
from .core.runner import Runner
from .core.sweep import Sweeper, SweepCell
from .core.schema import (
    CONDITION_NAMES,
    CONDITIONS,
    Condition,
    GroundTruth,
    Metrics,
    Predicate,
    QueryItem,
    RawResult,
    RawResults,
    load_query_set,
)

__all__ = [
    "VectorDBAdapter",
    "Condition",
    "CONDITIONS",
    "CONDITION_NAMES",
    "Dataset",
    "Runner",
    "Analyzer",
    "Profiler",
    "SelectivityProfile",
    "Sweeper",
    "SweepCell",
    "Predicate",
    "QueryItem",
    "GroundTruth",
    "RawResult",
    "RawResults",
    "Metrics",
    "load_query_set",
]

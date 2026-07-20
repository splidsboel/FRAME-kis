"""
The API contract: the abstract base every system-under-test implements.

An adapter author writes exactly TWO things — `setup()` (ingest the shared
LOGICAL schema into this system's physical layout + build the vector index) and
`search()` (translate the abstract predicates into native filter syntax and run
filtered k-NN). The shared Runner and Analyzer do the rest, unchanged, for every
system.

`setup()` IS the multi-table-workaround under study: pgvector can hold the
normalized schema and JOIN; Chroma/Milvus must denormalize into one collection.
Keep that mapping visible here rather than hiding it in a shared loader.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from .schema import Predicate


class VectorDBAdapter(ABC):
    #: short system name used in output filenames / plots, e.g. "pgvector"
    name: str = "unnamed"

    @abstractmethod
    def setup(self) -> None:
        """Bring the system to a queryable state: connect, ensure the shared
        logical schema is materialised in this system's physical layout, and
        ensure the vector index exists. Called once before any search.

        For pgvector the V3C data is already loaded (see V3C Schema), so this is
        mostly a connection + index check. For Chroma/Milvus this is where the
        denormalisation happens."""

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        filters: Sequence[Predicate],
        k: int,
    ) -> list[str]:
        """Run filtered k-NN for one query and return k keyframe_ids ranked
        best-first. `filters` are AND-ed; an empty sequence means no filter
        (the no-filter condition). `query_vector` is pre-encoded by the shared
        encoder so every system searches the SAME embedding (fairness)."""

    def teardown(self) -> None:
        """Optional cleanup (close connections, drop temp collections)."""

    # convenience so `with adapter:` does setup/teardown
    def __enter__(self) -> "VectorDBAdapter":
        self.setup()
        return self

    def __exit__(self, *exc) -> None:
        self.teardown()

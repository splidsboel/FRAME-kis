"""
The API contract: the abstract base every system-under-test implements.

An adapter author writes exactly THREE things:

  * `load_data(dataset)` — ONE-TIME ingest. Materialise the shared LOGICAL schema
    (a Tier-2 canonical shard) into this system's PHYSICAL layout and build the
    vector index. Expensive; run once per system per shard, out of band.
  * `setup()` — PER-RUN. Connect, verify the index is there, apply search-time
    knobs, pin planner statistics. Cheap; runs before every benchmark run.
  * `search()` — translate the abstract predicates into native filter syntax and
    run filtered k-NN.

The shared Runner and Analyzer do the rest, unchanged, for every system.

`load_data()` IS the multi-table-workaround under study: pgvector can hold the
normalised schema and JOIN; Chroma/Milvus must denormalise into one collection.
Keep that mapping visible in each adapter rather than hiding it in a shared
loader.

Why the split (see [[Data pipeline and adapter load refactor]]): ingest used to
live in `setup()`, which meant every benchmark run risked re-ingesting millions
of rows, and every new adapter would have re-run GPU processing to get data at
all. Splitting them lets the expensive, backend-neutral work happen once (Tier 1
-> Tier 2, shared) and the per-backend materialisation happen once per system
(Tier 2 -> Tier 3), leaving runs cheap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from .dataset import Dataset
from .schema import Predicate


class VectorDBAdapter(ABC):
    #: short system name used in output filenames / plots, e.g. "pgvector"
    name: str = "unnamed"

    @abstractmethod
    def load_data(self, dataset: Dataset) -> None:
        """ONE-TIME ingest: materialise a Tier-2 canonical shard into THIS
        system's physical layout, then build the vector index.

        This method IS the multi-table workaround under study. pgvector loads the
        normalised tables and JOINs at query time; Chroma/Milvus must denormalise
        the same neutral files into one flat collection. Keep that visible here.

        MUST be idempotent — re-running it on an already-loaded system is a no-op
        (or a resume), never a duplicate insert. It is NOT called by the Runner or
        by `with adapter:`; it is run out of band (scripts/load_dataset.py),
        because a benchmark run must never pay a multi-million-row ingest."""

    @abstractmethod
    def setup(self) -> None:
        """PER-RUN: bring an already-loaded system to a queryable state — connect,
        verify the vector index exists, apply search-time knobs, pin planner
        statistics. Called once before any search, and cheap by construction.

        Ingest does NOT belong here (see `load_data`). If the data is missing,
        raise rather than loading it: a run that silently ingests is a run whose
        timings mean nothing."""

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

"""
Runner — shared orchestration, written once, identical for every system.

Standalone object that takes an adapter (decided 2026-07-20) rather than a method
on the ABC. For each item it runs BOTH conditions and times the adapter's search:

  * filtered   : encode(vector_query)     + apply filters
  * no-filter  : encode(raw_query_text)   + no predicate

Note the two conditions embed DIFFERENT text (the semantic remainder vs the full
original query) — this is the "push the attribute into a filter vs leave it in the
CLIP query" comparison, not the same vector with/without a WHERE clause.

Retrieves k=1000 once per condition; the Analyzer slices @5/@25/@50/@100/@1000 out
of that single ranked list, so retrieval is paid once.
"""

from __future__ import annotations

from time import perf_counter
from typing import Sequence

from .adapter import VectorDBAdapter
from .encode import Encoder
from .schema import QueryItem, RawResult, RawResults

DEFAULT_K = 1000


class Runner:
    def __init__(self, adapter: VectorDBAdapter, encoder: Encoder):
        self.adapter = adapter
        self.encoder = encoder

    def run(self, items: Sequence[QueryItem], k: int = DEFAULT_K) -> RawResults:
        results: list[RawResult] = []
        for item in items:
            # filtered condition — semantic remainder + structured predicates
            vec_f = self.encoder.encode(item.vector_query)
            t0 = perf_counter()
            filtered_ids = self.adapter.search(vec_f, item.filters, k)
            lat_f = (perf_counter() - t0) * 1000.0

            # no-filter condition — full original query, no predicate
            vec_nf = self.encoder.encode(item.raw_query_text)
            t0 = perf_counter()
            unfiltered_ids = self.adapter.search(vec_nf, [], k)
            lat_nf = (perf_counter() - t0) * 1000.0

            results.append(RawResult(
                query_id=item.query_id,
                filtered_ids=filtered_ids,
                unfiltered_ids=unfiltered_ids,
                latency_filtered_ms=lat_f,
                latency_unfiltered_ms=lat_nf,
            ))
        return RawResults(system=self.adapter.name, k=k, results=results)

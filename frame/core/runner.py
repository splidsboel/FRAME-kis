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

LATENCY: each search is timed as the MEDIAN of `repeat` trials after `warmup`
untimed passes. The warmup primes the plan cache and OS/postgres buffers for that
query's pages, so a cold first-hit (which we measured at ~40x the warm cost) does
not masquerade as the query's real cost. Ranked ids are taken from the last run and
are stable across trials (deterministic given a fixed ef_search).
"""

from __future__ import annotations

from statistics import median
from time import perf_counter
from typing import Sequence

from .adapter import VectorDBAdapter
from .encode import Encoder
from .schema import Predicate, QueryItem, RawResult, RawResults

DEFAULT_K = 1000
DEFAULT_WARMUP = 1
DEFAULT_REPEAT = 5


class Runner:
    def __init__(
        self,
        adapter: VectorDBAdapter,
        encoder: Encoder,
        warmup: int = DEFAULT_WARMUP,
        repeat: int = DEFAULT_REPEAT,
    ):
        self.adapter = adapter
        self.encoder = encoder
        self.warmup = max(0, warmup)
        self.repeat = max(1, repeat)

    def run(self, items: Sequence[QueryItem], k: int = DEFAULT_K) -> RawResults:
        results: list[RawResult] = []
        for item in items:
            # filtered condition — semantic remainder + structured predicates
            vec_f = self.encoder.encode(item.vector_query)
            filtered_ids, lat_f = self._timed_search(vec_f, item.filters, k)

            # no-filter condition — full original query, no predicate
            vec_nf = self.encoder.encode(item.raw_query_text)
            unfiltered_ids, lat_nf = self._timed_search(vec_nf, [], k)

            results.append(RawResult(
                query_id=item.query_id,
                filtered_ids=filtered_ids,
                unfiltered_ids=unfiltered_ids,
                latency_filtered_ms=lat_f,
                latency_unfiltered_ms=lat_nf,
            ))
        return RawResults(system=self.adapter.name, k=k, results=results)

    def _timed_search(
        self, vec, filters: Sequence[Predicate], k: int
    ) -> tuple[list[str], float]:
        """Warm the query, then return (ranked_ids, median latency in ms)."""
        ids: list[str] = []
        for _ in range(self.warmup):
            ids = self.adapter.search(vec, filters, k)
        samples: list[float] = []
        for _ in range(self.repeat):
            t0 = perf_counter()
            ids = self.adapter.search(vec, filters, k)
            samples.append((perf_counter() - t0) * 1000.0)
        return ids, median(samples)

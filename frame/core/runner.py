"""
Runner — shared orchestration, written once, identical for every system.

Standalone object that takes an adapter (decided 2026-07-20) rather than a method
on the ABC. For each item it runs the full 2x2 condition matrix (schema.CONDITIONS,
approved by Omar 28-07-2026) and times the adapter's search in each cell:

                  | no predicate       | predicate applied
    raw_query_text| raw+nofilter       | raw+filter
    vector_query  | semantic+nofilter  | semantic+filter

The two axes isolate two different deltas: the PREDICATE axis is "push the attribute
into a filter vs leave it in the query", and the TEXT axis is "does isolating the
semantic remainder help or hurt". The old two-condition run measured only the
diagonal, which confounds them — a difference between semantic+filter and
raw+nofilter could come from either axis.

ITEMS WITH NO PREDICATE only run the two no-filter cells: with an empty predicate
the filter cells are the same search by construction, and running them would
manufacture two identical numbers that then flatter any filtered-vs-unfiltered
aggregate. They are absent from the result rather than duplicated.

Retrieves k=1000 once per condition; the Analyzer slices @5/@25/@50/@100/@1000 out
of that single ranked list, so retrieval is paid once. Cost note: the full matrix
is ~2x the searches of the old diagonal run (4 cells per filtered item instead of
2), but only 2 encodes per item — the shared CachingEncoder sees each text once.

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
from .schema import CONDITIONS, Predicate, QueryItem, RawResult, RawResults
from .version import HARNESS_CONTRACT, BenchmarkVersion  # noqa: F401  (annotation)

DEFAULT_K = 1000
DEFAULT_WARMUP = 1
DEFAULT_REPEAT = 5


def timed_search(
    adapter: VectorDBAdapter,
    vec,
    filters: Sequence[Predicate],
    k: int,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
) -> tuple[list[str], float]:
    """Warm the query, then return (ranked_ids, median latency in ms).

    Shared by the Runner and the cutover Sweeper so both measure latency the same
    way: `warmup` untimed passes prime the plan cache + OS/postgres buffers, then
    the median of `repeat` timed passes is the query's warm cost. Ranked ids are
    from the last pass and are stable across trials (deterministic given fixed
    ef_search)."""
    ids: list[str] = []
    for _ in range(max(0, warmup)):
        ids = adapter.search(vec, filters, k)
    samples: list[float] = []
    for _ in range(max(1, repeat)):
        t0 = perf_counter()
        ids = adapter.search(vec, filters, k)
        samples.append((perf_counter() - t0) * 1000.0)
    return ids, median(samples)


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

    def run(self, items: Sequence[QueryItem], k: int = DEFAULT_K,
            benchmark: "BenchmarkVersion | None" = None) -> RawResults:
        """Run every condition for every item. `benchmark` is the version marker of
        the query set being run, stamped into the results so they can later be shown
        comparable (or not) — see frame/core/version.py."""
        results: list[RawResult] = []
        for item in items:
            ids: dict[str, list[str]] = {}
            latency: dict[str, float] = {}
            # One encode per DISTINCT text, reused by that text's two cells, so the
            # filter axis is measured against an IDENTICAL query vector — any
            # difference is the predicate, not encoder noise. (Comprehending over
            # CONDITIONS directly would call encode() four times: dict keys dedupe
            # only after every value is evaluated.)
            vectors = {attr: self.encoder.encode(getattr(item, attr))
                       for attr in sorted({c.text_attr for c in CONDITIONS})}

            for cond in CONDITIONS:
                if cond.filtered and not item.filters:
                    continue        # empty predicate: identical to the no-filter cell
                filters = item.filters if cond.filtered else []
                ranked, ms = self._timed_search(vectors[cond.text_attr], filters, k)
                ids[cond.name] = ranked
                latency[cond.name] = ms

            results.append(RawResult(query_id=item.query_id, ids=ids,
                                     latency_ms=latency))
        return RawResults(system=self.adapter.name, k=k, results=results,
                          benchmark=benchmark, harness_contract=HARNESS_CONTRACT)

    def _timed_search(
        self, vec, filters: Sequence[Predicate], k: int
    ) -> tuple[list[str], float]:
        """Warm the query, then return (ranked_ids, median latency in ms)."""
        return timed_search(
            self.adapter, vec, filters, k, warmup=self.warmup, repeat=self.repeat
        )

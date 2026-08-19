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

import json
from statistics import median
from time import perf_counter
from typing import Sequence

from .adapter import VectorDBAdapter
from .encode import Encoder
from .schema import CONDITIONS, Condition, Predicate, QueryItem, RawResult, RawResults, VariantResult
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
        grade_variants: bool = False,
        conditions: "Sequence[Condition] | None" = None,
    ):
        self.adapter = adapter
        self.encoder = encoder
        self.warmup = max(0, warmup)
        self.repeat = max(1, repeat)
        # Which of the 2x2 cells to run. Defaults to the full matrix; the k×ef sweep
        # passes a LEAN subset (semantic+filter, semantic+nofilter) — ef is a recall
        # dial only on the HNSW path, so the two semantic cells carry the signal and
        # the two raw cells would only add filtered-search cost. An item with no
        # predicate still drops its filter cells (see _run_items), as always.
        self.conditions = tuple(conditions) if conditions is not None else CONDITIONS
        # Opt-in: also run every real human phrasing of each item as its own query
        # (schema.VariantResult). Off by default — the extra searches are non-trivial
        # (a filtered phrasing pays the same filter cost as a 2x2 filter cell), and a
        # plain run's results file is then byte-identical to before.
        self.grade_variants = grade_variants

    def run(self, items: Sequence[QueryItem], k: int = DEFAULT_K,
            benchmark: "BenchmarkVersion | None" = None,
            progress_path: "str | None" = None) -> RawResults:
        """Run every condition for every item. `benchmark` is the version marker of
        the query set being run, stamped into the results so they can later be shown
        comparable (or not) — see frame/core/version.py.

        If `progress_path` is given, each item's result is flushed to that JSONL file
        the moment it completes, so a wall-clock kill (SLURM time limit) keeps every
        query already finished instead of losing the whole run — results were
        otherwise only written after all N items. The streamed file is byte-identical
        to RawResults.write_jsonl (same header + rows) so it reads back with
        RawResults.read_jsonl unchanged, whether the run finished or was cut short."""
        sink = None
        if progress_path is not None:
            sink = open(progress_path, "w")
            header: dict = {"system": self.adapter.name, "k": k,
                            "harness_contract": HARNESS_CONTRACT}
            if benchmark is not None:
                header["benchmark"] = benchmark.to_dict()
            # Keep this header byte-identical to RawResults.write_jsonl (same keys,
            # same order) so a streamed partial file reads back unchanged — ef_search
            # last, and omitted when the adapter left its default in place.
            ef = getattr(self.adapter, "ef_search", None)
            if ef is not None:
                header["ef_search"] = ef
            sink.write(json.dumps(header) + "\n")
            sink.flush()
        try:
            results = self._run_items(items, k, sink)
        finally:
            if sink is not None:
                sink.close()
        return RawResults(system=self.adapter.name, k=k, results=results,
                          benchmark=benchmark, harness_contract=HARNESS_CONTRACT,
                          ef_search=getattr(self.adapter, "ef_search", None))

    def _run_items(self, items, k, sink):
        results: list[RawResult] = []
        n = len(items)
        for i, item in enumerate(items, 1):
            # Progress line per item: a long run used to stall silently (the summary
            # only prints at the very end), so a runaway query was invisible until the
            # wall-clock limit killed the job. Print start + per-item wall time,
            # flushed, so the log shows exactly which query is slow.
            t_item = perf_counter()
            print(f"[run] {i}/{n} {item.query_id}  filters={len(item.filters)} ...",
                  flush=True)
            ids: dict[str, list[str]] = {}
            latency: dict[str, float] = {}
            # One encode per DISTINCT text, reused by that text's two cells, so the
            # filter axis is measured against an IDENTICAL query vector — any
            # difference is the predicate, not encoder noise. (Comprehending over
            # CONDITIONS directly would call encode() four times: dict keys dedupe
            # only after every value is evaluated.)
            vectors = {attr: self.encoder.encode(getattr(item, attr))
                       for attr in sorted({c.text_attr for c in CONDITIONS})}

            for cond in self.conditions:
                if cond.filtered and not item.filters:
                    continue        # empty predicate: identical to the no-filter cell
                filters = item.filters if cond.filtered else []
                ranked, ms = self._timed_search(vectors[cond.text_attr], filters, k)
                ids[cond.name] = ranked
                latency[cond.name] = ms

            variants = self._grade_variants(item, k) if self.grade_variants else []

            result = RawResult(query_id=item.query_id, ids=ids, latency_ms=latency,
                               variants=variants)
            results.append(result)
            if sink is not None:
                # Flush per item so a killed job keeps this query. flush() hands the
                # bytes to the OS, which survives the process being torn down; the
                # next job's Analyzer can read the partial file as-is.
                sink.write(json.dumps(result.to_dict()) + "\n")
                sink.flush()
            lat = "  ".join(f"{name}={ms:.0f}ms" for name, ms in latency.items())
            nvar = f"  +{len(variants)} variants" if variants else ""
            print(f"[run] {i}/{n} {item.query_id}  done in "
                  f"{perf_counter() - t_item:.1f}s  ({lat}){nvar}", flush=True)
        return results

    def _grade_variants(self, item: QueryItem, k: int) -> list[VariantResult]:
        """Run each real human phrasing of `item` as its own query: "nofilter"
        always, and "filter" too when the item carries a predicate (the task's
        SHARED filter — the filter is task-level, so this is the same predicate under
        many wordings, not a new filter measurement).

        A SINGLE search per (phrasing, cond), no warmup/repeat: variants are scored on
        target RANK, which is deterministic given a fixed ef_search, so there is
        nothing to average — and this is the semantic axis, not the latency headline,
        which the 2x2 cells own. The shared CachingEncoder means a phrasing repeated
        across systems is embedded once."""
        out: list[VariantResult] = []
        for vi, var in enumerate(item.user_query_variants):
            text = var.get("text", "")
            vec = self.encoder.encode(text)
            ids: dict[str, list[str]] = {"nofilter": self.adapter.search(vec, [], k)}
            if item.filters:
                ids["filter"] = self.adapter.search(vec, item.filters, k)
            out.append(VariantResult(index=vi, team=var.get("team", ""),
                                     action=var.get("action", ""), text=text, ids=ids))
        return out

    def _timed_search(
        self, vec, filters: Sequence[Predicate], k: int
    ) -> tuple[list[str], float]:
        """Warm the query, then return (ranked_ids, median latency in ms)."""
        return timed_search(
            self.adapter, vec, filters, k, warmup=self.warmup, repeat=self.repeat
        )

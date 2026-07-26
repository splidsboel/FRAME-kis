"""
sweep.py — dense recall-vs-selectivity sweep across the exact↔approximate cutover.

Standalone diagnostic (like the Profiler and Analyzer): given the loaded query
items and a LIVE pgvector adapter, it turns the *bracketed* cutover claim ("the
seqscan→HNSW flip sits around ~5–10% selectivity") into a *measured curve*. For
each filtered item it sweeps an `ef_search × iterative_scan` grid at a moderate
run depth `k`, and records — per grid cell — the plan the planner picked, how many
rows came back, and recall@k against the oracle's EXACT filtered k-NN (gt_filtered).

Why a separate entry point from run_benchmark: pgvector requires ef_search ≥ k and
caps it at 1000, so a top-1000 run pins ef_search=1000 (no headroom → recall never
degrades). Dropping to a moderate k (default 100) opens a real ef_search range and
lets the approximate-path recall degrade — which is the effect we characterize.

Two plan modes:
  * "auto"       — the planner's real choice (answers WHERE the cutover flips).
  * "force-hnsw" — SET enable_seqscan=off, pushing EVERY filter onto the approximate
                   index path (answers WHAT the approximate path costs at each
                   selectivity, densely, even for filters the planner keeps exact).

Scored against gt_filtered for ALL filtered items — including the harm exemplars
whose filter excludes their own target. Recall@k vs the exact filtered k-NN is a
purely geometric measure (does the HNSW walk miss what filter-then-scan returns?),
independent of whether the KIS target survives the filter, so those items are
in-scope here even though the Analyzer marks them unscorable for KIS.

Needs the DB, so it runs on the HPC (like build_gt / run_benchmark / profile). Plan
detection is pgvector-specific; the recall + selectivity parts are system-agnostic.
Requires an adapter exposing corpus_size / count_passing / plan_choice /
set_session_knobs (pgvector has them).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

# The canonical metric primitives live in the Analyzer — import them rather than
# reimplement, so the sweep's recall/rank match the main benchmark exactly.
from .analyzer import _first_rank, _recall_at_k
from .encode import Encoder
from .profile import _summarize_filters
from .runner import DEFAULT_REPEAT, DEFAULT_WARMUP, timed_search
from .schema import QueryItem

DEFAULT_K = 100
DEFAULT_EF_GRID = (100, 150, 200, 300, 500, 1000)
DEFAULT_MODES = ("off", "relaxed_order", "strict_order")
DEFAULT_SCORE_KS = (10, 25, 50, 100)
DEFAULT_PLAN_MODES = ("auto", "force-hnsw")


@dataclass
class SweepCell:
    """One (query, ef_search, iterative_scan, plan_mode) measurement."""
    query_id: str
    filter_summary: str
    n_filters: int
    global_selectivity: float           # true global passing set / corpus (per item)
    plan_mode: str                      # "auto" | "force-hnsw"
    plan: str                           # observed plan in this cell: hnsw|seqscan|other
    planner_est_rows: int | None        # planner's estimated passing set (EXPLAIN)
    ef_search: int
    iterative_scan: str
    k: int                              # run depth (LIMIT); rows_returned ≤ k
    rows_returned: int                  # < k ⇒ HNSW post-filter truncation (recall ceiling)
    recall: dict[int, float]            # score-k -> recall@k vs gt_filtered
    target_rank: int | None             # 1-based rank of best target kf (None if excluded/absent)
    target_passes_filter: bool | None   # False ⇒ harm exemplar (geometric recall still valid)
    latency_ms: float                   # warm median of the filtered search

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "filter_summary": self.filter_summary,
            "n_filters": self.n_filters,
            "global_selectivity": self.global_selectivity,
            "plan_mode": self.plan_mode,
            "plan": self.plan,
            "planner_est_rows": self.planner_est_rows,
            "ef_search": self.ef_search,
            "iterative_scan": self.iterative_scan,
            "k": self.k,
            "rows_returned": self.rows_returned,
            "recall": {str(sk): v for sk, v in self.recall.items()},
            "target_rank": self.target_rank,
            "target_passes_filter": self.target_passes_filter,
            "latency_ms": self.latency_ms,
        }


class Sweeper:
    """Sweeps each filtered item's recall across the ef_search × iterative_scan grid.

    `adapter` must expose corpus_size / count_passing / plan_choice /
    set_session_knobs (pgvector). `encoder` is the shared text encoder (a
    CachingEncoder makes each vector_query cost one encode, reused across cells).
    """

    def __init__(
        self,
        adapter,
        encoder: Encoder,
        k: int = DEFAULT_K,
        ef_grid: Sequence[int] = DEFAULT_EF_GRID,
        modes: Sequence[str] = DEFAULT_MODES,
        plan_modes: Sequence[str] = DEFAULT_PLAN_MODES,
        score_ks: Sequence[int] = DEFAULT_SCORE_KS,
        warmup: int = DEFAULT_WARMUP,
        repeat: int = DEFAULT_REPEAT,
    ):
        self.adapter = adapter
        self.encoder = encoder
        self.k = k
        # pgvector: ef_search must be ≥ k and ≤ 1000. Drop illegal points (sorted, unique).
        self.ef_grid = sorted({e for e in ef_grid if k <= e <= 1000})
        self.modes = tuple(modes)
        self.plan_modes = tuple(plan_modes)
        # can only score at depths actually retrieved (≤ k)
        self.score_ks = tuple(sk for sk in score_ks if sk <= k)
        self.warmup = warmup
        self.repeat = repeat

    def run(self, items: Sequence[QueryItem]) -> list[SweepCell]:
        # In scope: filtered items that already have oracle filtered GT. (Semantic-only
        # items have no filter; person-family single-label anchors have no oracle GT yet
        # — a follow-on that needs on-the-fly exact GT.)
        filtered = [
            it for it in items
            if it.filters and it.ground_truth and it.ground_truth.gt_filtered
        ]
        cells: list[SweepCell] = []
        with self.adapter:                      # one connection, one pinned ANALYZE
            corpus = self.adapter.corpus_size()
            # selectivity is constant per item — compute once.
            sel = {it.query_id: self.adapter.count_passing(it.filters) for it in filtered}

            for plan_mode in self.plan_modes:
                # force-hnsw is a session toggle; reset to the honest planner between blocks.
                self.adapter.set_session_knobs(enable_seqscan=(plan_mode != "force-hnsw"))
                for mode in self.modes:
                    # plan choice depends on (mode, plan_mode, k) but NOT ef_search — cache it.
                    plan_cache: dict[str, tuple[str, int | None]] = {}
                    for ef in self.ef_grid:
                        self.adapter.set_session_knobs(ef_search=ef, iterative_scan=mode)
                        for it in filtered:
                            vec = self.encoder.encode(it.vector_query)  # cached after 1st cell
                            if it.query_id not in plan_cache:
                                plan_cache[it.query_id] = self.adapter.plan_choice(
                                    vec, it.filters, self.k, iterative_scan=mode)
                            plan, est = plan_cache[it.query_id]
                            ids, lat = timed_search(
                                self.adapter, vec, it.filters, self.k,
                                warmup=self.warmup, repeat=self.repeat)
                            cells.append(self._cell(
                                it, sel[it.query_id], corpus, plan_mode, plan, est,
                                ef, mode, ids, lat))
            # leave the connection as the honest planner found it
            self.adapter.set_session_knobs(enable_seqscan=True)
        return cells

    def _cell(self, it, global_count, corpus, plan_mode, plan, est, ef, mode, ids, lat):
        gt = it.ground_truth
        recall = {sk: _recall_at_k(ids, gt.gt_filtered, sk) for sk in self.score_ks}
        targets = set(gt.target_keyframe_ids or [])
        return SweepCell(
            query_id=it.query_id,
            filter_summary=_summarize_filters(it.filters),
            n_filters=len(it.filters),
            global_selectivity=(global_count / corpus) if corpus else 0.0,
            plan_mode=plan_mode,
            plan=plan,
            planner_est_rows=est,
            ef_search=ef,
            iterative_scan=mode,
            k=self.k,
            rows_returned=len(ids),
            recall=recall,
            target_rank=_first_rank(ids, targets),
            target_passes_filter=gt.target_passes_filter,
            latency_ms=lat,
        )

    def summary(self, cells: Sequence[SweepCell]) -> str:
        """A recall@k(run k) matrix (rows = items by selectivity, cols = ef_search)
        per (plan_mode, iterative_scan) — the cutover curve at a glance. The full
        grid (all score-ks, latency, rows_returned) is in the JSONL."""
        if not cells:
            return "no filtered items with GT to sweep"
        k = self.k
        # pick the pinned mode for the compact view if present, else the first swept.
        view_mode = "relaxed_order" if "relaxed_order" in self.modes else self.modes[0]
        lines: list[str] = []
        for plan_mode in self.plan_modes:
            block = [c for c in cells
                     if c.plan_mode == plan_mode and c.iterative_scan == view_mode]
            if not block:
                continue
            by_qid: dict[str, dict[int, SweepCell]] = {}
            for c in block:
                by_qid.setdefault(c.query_id, {})[c.ef_search] = c
            order = sorted(by_qid, key=lambda q: next(iter(by_qid[q].values())).global_selectivity)
            header = (f"{'qid':>6} | {'sel':>6} | {'plan':>7} | "
                      + " | ".join(f"ef{ef:>4}" for ef in self.ef_grid))
            lines += [
                "",
                f"recall@{k}  ·  plan_mode={plan_mode}  ·  iterative_scan={view_mode}",
                header,
                "-" * len(header),
            ]
            for q in order:
                row = by_qid[q]
                any_c = next(iter(row.values()))
                cellstrs = []
                for ef in self.ef_grid:
                    c = row.get(ef)
                    cellstrs.append(f"{c.recall.get(k, 0.0):6.3f}" if c else "   -  ")
                lines.append(
                    f"{q:>6} | {any_c.global_selectivity:6.3f} | "
                    f"{any_c.plan:>7} | " + " | ".join(cellstrs))
        return "\n".join(lines)


def write_jsonl(cells: Sequence[SweepCell], path: str, system: str, k: int) -> None:
    with open(path, "w") as f:
        f.write(json.dumps({"system": system, "kind": "cutover_sweep", "k": k}) + "\n")
        for c in cells:
            f.write(json.dumps(c.to_dict()) + "\n")

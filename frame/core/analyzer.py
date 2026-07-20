"""
Analyzer — shared scoring, written once. Reads system output (RawResults) + the
oracle ground truth (from each item's `computed` block) and produces metrics.

Two families of metric, mirroring the thesis' two lenses:
  * geometric correctness — Recall@k of the system's ranking vs the oracle's exact
    filtered / unfiltered k-NN.
  * task success (KIS) — rank of the known target keyframe(s) in the system ranking,
    summarised as MRR.

Only items with computed, self-consistent GT are scored (see GroundTruth.is_scorable):
filtered GT present AND the target survives its own filter. Items whose filter
excludes their target are reported as unscorable rather than silently counted as
Recall@k = 0.
"""

from __future__ import annotations

from typing import Sequence

from .schema import (
    GroundTruth,
    Metrics,
    QueryItem,
    QueryMetrics,
    RawResult,
    RawResults,
)

DEFAULT_KS = (5, 25, 50, 100, 1000)


class Analyzer:
    def __init__(self, ks: Sequence[int] = DEFAULT_KS):
        self.ks = tuple(ks)

    def analyze(self, raw: RawResults, items: Sequence[QueryItem]) -> Metrics:
        gt_by_id = {it.query_id: it.ground_truth for it in items}
        per_query = [
            self._score_one(r, gt_by_id.get(r.query_id))
            for r in raw.results
        ]
        return Metrics(system=raw.system, ks=self.ks, per_query=per_query)

    def _score_one(self, r: RawResult, gt: GroundTruth | None) -> QueryMetrics:
        scorable = gt is not None and gt.is_scorable
        recall_f = {k: 0.0 for k in self.ks}
        recall_nf = {k: 0.0 for k in self.ks}
        rank_f = rank_nf = None

        if scorable and gt is not None:
            for k in self.ks:
                recall_f[k] = _recall_at_k(r.filtered_ids, gt.gt_filtered, k)
                recall_nf[k] = _recall_at_k(r.unfiltered_ids, gt.gt_nofilter, k)
            targets = set(gt.target_keyframe_ids or [])
            rank_f = _first_rank(r.filtered_ids, targets)
            rank_nf = _first_rank(r.unfiltered_ids, targets)

        return QueryMetrics(
            query_id=r.query_id,
            scorable=scorable,
            recall_filtered=recall_f,
            recall_unfiltered=recall_nf,
            target_rank_filtered=rank_f,
            target_rank_unfiltered=rank_nf,
            latency_filtered_ms=r.latency_filtered_ms,
            latency_unfiltered_ms=r.latency_unfiltered_ms,
        )

    def summary(self, m: Metrics) -> str:
        n = len(m.per_query)
        n_ok = sum(1 for q in m.per_query if q.scorable)
        lines = [
            f"system: {m.system}   items: {n}   scorable: {n_ok}",
            "",
            f"{'k':>6} | {'recall(filt)':>12} | {'recall(nofilt)':>14}",
            "-" * 40,
        ]
        for k in m.ks:
            lines.append(
                f"{k:>6} | {m.mean_recall_filtered(k):>12.3f} | "
                f"{m.mean_recall_unfiltered(k):>14.3f}"
            )
        lines += [
            "",
            f"MRR  filtered: {m.mrr_filtered():.3f}   "
            f"no-filter: {m.mrr_unfiltered():.3f}   "
            f"(Δ = {m.mrr_filtered() - m.mrr_unfiltered():+.3f})",
            "",
            f"median latency  filtered: {m.median_latency_filtered():8.1f} ms   "
            f"no-filter: {m.median_latency_unfiltered():8.1f} ms   "
            f"(all {n} items, warm)",
        ]
        return "\n".join(lines)


def _recall_at_k(retrieved: list[str], gt: list[str] | None, k: int) -> float:
    """Fraction of the oracle's top-k that the system retrieved in its top-k."""
    if not gt:
        return 0.0
    truth = set(gt[:k])
    if not truth:
        return 0.0
    hits = sum(1 for kid in retrieved[:k] if kid in truth)
    return hits / len(truth)


def _first_rank(retrieved: list[str], targets: set[str]) -> int | None:
    """1-based rank of the first retrieved id that is a target keyframe."""
    for i, kid in enumerate(retrieved, start=1):
        if kid in targets:
            return i
    return None

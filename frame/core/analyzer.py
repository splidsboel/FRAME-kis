"""
Analyzer — shared scoring, written once. Reads system output (RawResults) + the
oracle ground truth (from each item's `computed` block) and produces metrics.

Two families of metric, mirroring the thesis' two lenses:
  * geometric correctness — Recall@k of the system's ranking vs the oracle's exact
    filtered / unfiltered k-NN, sliced at k ∈ DEFAULT_KS.
  * task success (KIS) — rank of the known target keyframe(s) in the system ranking,
    summarised as MRR at the rank caps in DEFAULT_MRR_CAPS (a target deeper than the
    cap is a miss, not a small reciprocal).

Latency is reported as the median AND the p95 across queries — the median is the
typical cost, the p95 is where a planner cutover or a broad filter shows up, and
reporting only the median hides exactly the queries the thesis is about.

The Analyzer produces numbers, never figures. Plotting lives in
scripts/plot_metrics.py, which reads the metrics jsonl this writes.

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

# Rank caps for MRR (Omar, 28-07-2026): a target found deeper than the cap counts
# as a MISS. Deliberately a different set from DEFAULT_KS — those are fine-grained
# recall slices, these are "how deep would a VBS user realistically look".
DEFAULT_MRR_CAPS = (1000, 100, 50, 10)


class Analyzer:
    def __init__(self, ks: Sequence[int] = DEFAULT_KS,
                 mrr_caps: Sequence[int] = DEFAULT_MRR_CAPS):
        self.ks = tuple(ks)
        self.mrr_caps = tuple(mrr_caps)

    def analyze(self, raw: RawResults, items: Sequence[QueryItem]) -> Metrics:
        gt_by_id = {it.query_id: it.ground_truth for it in items}
        per_query = [
            self._score_one(r, gt_by_id.get(r.query_id))
            for r in raw.results
        ]
        return Metrics(system=raw.system, ks=self.ks, per_query=per_query,
                       retrieval_k=raw.k)

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
            f"{'MRR@cap':>7} | {'filtered':>8} | {'no-filter':>9} | {'Δ':>7}",
            "-" * 40,
        ]
        for cap in self.mrr_caps:
            f, nf = m.mrr_filtered(cap), m.mrr_unfiltered(cap)
            # A cap at or past the retrieval depth cannot bite: no rank beyond k
            # exists, so that row is the uncapped MRR. Say so rather than let it
            # read as a fourth data point.
            note = "" if m.cap_is_meaningful(cap) else f"  (= uncapped, run k={m.retrieval_k})"
            lines.append(f"{cap:>7} | {f:>8.3f} | {nf:>9.3f} | {f - nf:>+7.3f}{note}")

        lines += [
            "",
            f"{'latency':>7} | {'filtered':>10} | {'no-filter':>11}",
            "-" * 40,
            f"{'median':>7} | {m.median_latency_filtered():>7.1f} ms | "
            f"{m.median_latency_unfiltered():>8.1f} ms",
            f"{'p95':>7} | {m.latency_percentile_filtered(95):>7.1f} ms | "
            f"{m.latency_percentile_unfiltered(95):>8.1f} ms",
            f"(across all {n} items, warm; each item is itself the median of the "
            f"Runner's repeat trials)",
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

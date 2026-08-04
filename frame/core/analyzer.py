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

Everything is scored PER CONDITION (the 2x2 matrix — schema.CONDITIONS), each cell
against its own oracle answer: semantic+filter vs gt_filtered, raw+filter vs
gt_raw_filtered, raw+nofilter vs gt_nofilter, semantic+nofilter vs gt_vec_nofilter.
No cell is scored against a stand-in for its own ground truth.

Scorability is also per condition (GroundTruth.scorable_for): a cell needs its own
GT, and the two FILTER cells additionally need the target to survive the filter,
else Recall@k is 0 by construction. Such items are reported unscorable for that
cell rather than silently counted as Recall@k = 0. Aggregates then default to the
items scorable in EVERY condition (Metrics.comparable) so the four cells are
compared over one common subset — see the note on Metrics.comparable for why.
"""

from __future__ import annotations

from typing import Sequence

from .schema import (
    CONDITIONS,
    GroundTruth,
    Metrics,
    QueryItem,
    QueryMetrics,
    RawResult,
    RawResults,
)
from .version import HARNESS_CONTRACT, BenchmarkVersion, Compatibility, compare


class VersionMismatch(RuntimeError):
    """Results cannot be scored against this query set / harness."""


def check_compatible(raw: RawResults, benchmark: BenchmarkVersion | None,
                     allow_mismatch: bool = False) -> Compatibility | None:
    """Raise unless `raw` may be scored against `benchmark` under this harness.

    Returns the Compatibility so the caller can act on an `additive` result. Returns
    None when there is nothing to check (no marker on either side and nothing to
    compare against), which only happens in unit tests and ad-hoc scoring.
    """
    problems: list[str] = []

    if raw.harness_contract != HARNESS_CONTRACT:
        known = raw.harness_contract or "unknown (predates versioning)"
        problems.append(
            f"harness contract {known}, but this harness is {HARNESS_CONTRACT} — "
            f"what a results file means has changed (conditions and/or metric "
            f"definitions), so the numbers are not comparable")

    compat: Compatibility | None = None
    if benchmark is not None or raw.benchmark is not None:
        compat = compare(raw.benchmark, benchmark)
        if not compat.ok:
            problems.append(f"query set: {compat.reason}")

    if problems:
        detail = "\n  - ".join(problems)
        if not allow_mismatch:
            raise VersionMismatch(
                f"refusing to score these results:\n  - {detail}\n"
                f"Re-run the benchmark against the current query set, or pass "
                f"--allow-mismatch / allow_mismatch=True to score anyway (the "
                f"numbers will not be comparable).")
        print(f"[WARN] scoring despite a version mismatch:\n  - {detail}")
    elif compat is not None:
        if compat.status == "additive":
            print(f"[note] query set moved since the run: {compat.reason}")
        elif compat.reason:
            print(f"[note] {compat.reason}")

    return compat

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

    def analyze(self, raw: RawResults, items: Sequence[QueryItem],
                benchmark: "BenchmarkVersion | None" = None,
                allow_mismatch: bool = False) -> Metrics:
        """Score `raw` against `items`.

        Refuses by default when the results were not produced against this query set
        (or by this harness) — a silently wrong comparison is the expensive failure
        mode, an error message is the cheap one. `allow_mismatch=True` downgrades the
        refusal to a printed warning.
        """
        compat = check_compatible(raw, benchmark, allow_mismatch=allow_mismatch)

        gt_by_id = {it.query_id: it.ground_truth for it in items}
        results = raw.results
        if compat is not None and compat.status == "additive":
            # Every shared item is unchanged, so those remain valid; the rest are
            # dropped rather than scored against a query set that lacks them.
            shared = set(compat.shared)
            results = [r for r in results if r.query_id in shared]

        per_query = [self._score_one(r, gt_by_id.get(r.query_id)) for r in results]
        return Metrics(system=raw.system, ks=self.ks, per_query=per_query,
                       retrieval_k=raw.k,
                       benchmark=benchmark if benchmark is not None else raw.benchmark,
                       harness_contract=HARNESS_CONTRACT)

    def _score_one(self, r: RawResult, gt: GroundTruth | None) -> QueryMetrics:
        targets = set(gt.target_keyframe_ids or []) if gt else set()
        scorable: dict[str, bool] = {}
        recall: dict[str, dict[int, float]] = {}
        rank: dict[str, int | None] = {}

        for cond in CONDITIONS:
            if cond.name not in r.ids:
                continue            # the run did not produce this cell for this item
            ok = gt is not None and gt.scorable_for(cond)
            scorable[cond.name] = ok
            recall[cond.name] = {k: 0.0 for k in self.ks}
            rank[cond.name] = None
            if not ok or gt is None:
                continue
            ids = r.ids[cond.name]
            for k in self.ks:
                recall[cond.name][k] = _recall_at_k(ids, gt.gt_for(cond), k)
            rank[cond.name] = _first_rank(ids, targets)

        return QueryMetrics(
            query_id=r.query_id,
            scorable=scorable,
            recall=recall,
            target_rank=rank,
            latency_ms=dict(r.latency_ms),
        )

    def headline_cap(self, m: Metrics) -> int | None:
        """The deepest cap that actually bites for this run — the one worth putting
        on the 2x2 grid. None if every cap is at/beyond the retrieval depth."""
        biting = [c for c in self.mrr_caps if m.cap_is_meaningful(c)]
        return max(biting) if biting else None

    def summary(self, m: Metrics) -> str:
        conds = m.conditions()
        n, n_cmp = len(m.per_query), len(m.comparable())
        w = max((len(c) for c in conds), default=9)

        lines = [
            f"system: {m.system}   items: {n}   comparable "
            f"(scorable in all {len(conds)} conditions): {n_cmp}",
            f"conditions: {', '.join(conds)}",
            "",
            f"Recall@k vs the oracle's exact k-NN — over the {n_cmp} comparable items",
            f"{'condition':<{w}} | " + " | ".join(f"{k:>6}" for k in m.ks),
            "-" * (w + 3 + 9 * len(m.ks)),
        ]
        for c in conds:
            lines.append(f"{c:<{w}} | " +
                         " | ".join(f"{m.mean_recall(c, k):>6.3f}" for k in m.ks))

        lines += [
            "",
            f"MRR at rank caps — over the {n_cmp} comparable items",
            f"{'condition':<{w}} | " + " | ".join(f"{'@' + str(c):>7}" for c in self.mrr_caps),
            "-" * (w + 3 + 10 * len(self.mrr_caps)),
        ]
        for c in conds:
            lines.append(f"{c:<{w}} | " +
                         " | ".join(f"{m.mrr(c, cap):>7.3f}" for cap in self.mrr_caps))
        # A cap at or past the retrieval depth cannot bite: no rank beyond k exists,
        # so those columns repeat the uncapped MRR. Say so rather than let them read
        # as independent measurements.
        inert = [c for c in self.mrr_caps if not m.cap_is_meaningful(c)]
        if inert and m.retrieval_k:
            lines.append(f"(caps {', '.join('@' + str(c) for c in inert)} are at or "
                         f"beyond the run's retrieval depth k={m.retrieval_k}, so they "
                         f"equal the uncapped MRR)")

        lines += self._grid(m)

        lines += [
            "",
            "Latency (warm; each item is itself the median of the Runner's "
            "repeat trials)",
            f"{'condition':<{w}} | {'median':>10} | {'p95':>10} | {'n':>4}",
            "-" * (w + 33),
        ]
        for c in conds:
            lines.append(f"{c:<{w}} | {m.median_latency(c):>7.1f} ms | "
                         f"{m.latency_percentile(c, 95):>7.1f} ms | "
                         f"{m.latency_n(c):>4}")
        return "\n".join(lines)

    def _grid(self, m: Metrics) -> list[str]:
        """The 2x2 read as a grid: the filter delta down one axis, the raw-vs-
        semantic delta across the other. Only drawn when all four cells ran —
        a partial matrix has no interpretable margins."""
        conds = m.conditions()
        if len(conds) < len(CONDITIONS) or not all(c.name in conds for c in CONDITIONS):
            return ["", "(2x2 grid omitted — this run produced "
                    f"{len(conds)} of {len(CONDITIONS)} conditions)"]
        cap = self.headline_cap(m)
        val = lambda text, filt: m.mrr(f"{text}+{'filter' if filt else 'nofilter'}", cap)
        label = f"MRR@{cap}" if cap else "MRR (uncapped)"
        out = [
            "",
            f"The 2x2 — {label}, over the {len(m.comparable())} comparable items",
            f"{'':<10} | {'no-filter':>10} | {'filter':>10} | {'Δ filter':>10}",
            "-" * 49,
        ]
        for text in ("raw", "semantic"):
            nf, f = val(text, False), val(text, True)
            out.append(f"{text:<10} | {nf:>10.3f} | {f:>10.3f} | {f - nf:>+10.3f}")
        d_nf = val("semantic", False) - val("raw", False)
        d_f = val("semantic", True) - val("raw", True)
        out.append(f"{'Δ semantic':<10} | {d_nf:>+10.3f} | {d_f:>+10.3f} |")
        out.append("(Δ filter = pushing the attribute into a predicate; "
                   "Δ semantic = isolating the semantic remainder)")
        return out


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

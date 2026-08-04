"""
Shared data types — the SEAM between the two halves of the suite.

Left half (oracle / query set, HPC): authors queries and computes ground truth,
persisting both into `data/benchmark.jsonl`. Right half (harness): reads the same
file, runs each system, scores against the computed GT. These dataclasses are the
one place the on-disk shapes are defined for the harness side.

The on-disk item shape is authored in `queryset/queries/*.json` and compiled +
GT-enriched into `data/benchmark.jsonl` (see queryset/build.py, oracle/build_gt.py).
GT lives INSIDE each item's `computed` block (enrich-in-place), not in a separate
file — the harness reads it from there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence


# ─────────────────────────────────────────────────────────────────────────────
# Authored side: an abstract, system-agnostic filter predicate.
# Same object is translated by the oracle (truth) and by each adapter (under test)
# into their own dialects — but via SEPARATE code paths (see architecture notes).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Predicate:
    filter_type: str          # "scene" | "object" | "pattern-match" | "video-category" | "video-tag"
    attribute: str            # e.g. "scene_label", "object_label", "ocr_text", "video_categories"
    op: str                   # "in" | "contains"
    value: Any                # list[str] for label/video-meta filters; str|list[str] for pattern
    vocab: str | None = None
    mapping_source: str | None = None
    verified: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "Predicate":
        return cls(
            filter_type=d["filter_type"],
            attribute=d["attribute"],
            op=d["op"],
            value=d["value"],
            vocab=d.get("vocab"),
            mapping_source=d.get("mapping_source"),
            verified=d.get("verified", False),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth for one item — read out of the item's `computed` block.
# Field names mirror queryset/build.py's computed_stub / oracle/build_gt.py output.
# `None` means "not yet computed on the HPC" (item still pending GT).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GroundTruth:
    query_id: str
    target_keyframe_ids: list[str] | None
    target_passes_filter: bool | None
    filter_selectivity: list[float | None]
    gt_filtered: list[str] | None      # exact filtered k-NN keyframe ids (ranked)
    gt_nofilter: list[str] | None      # exact unfiltered k-NN keyframe ids (ranked, RAW query text)
    gt_vec_nofilter: list[str] | None  # exact unfiltered k-NN of the vector_query (no filter);
                                       # the neighbour list the filtered HNSW walk actually traverses,
                                       # used by the profiler for near-query pass-rate

    @property
    def is_scorable(self) -> bool:
        """An item can be scored only once its filtered GT exists AND its own
        target survives the filter (else Recall@k is 0 by construction)."""
        return bool(self.gt_filtered) and self.target_passes_filter is True

    @classmethod
    def from_item(cls, item: dict) -> "GroundTruth":
        c = item.get("computed") or {}
        return cls(
            query_id=item["query_id"],
            target_keyframe_ids=c.get("target_keyframe_ids"),
            target_passes_filter=c.get("target_passes_filter"),
            filter_selectivity=c.get("filter_selectivity") or [],
            gt_filtered=c.get("geometric_gt_filtered"),
            gt_nofilter=c.get("geometric_gt_nofilter"),
            gt_vec_nofilter=c.get("geometric_gt_vec_nofilter"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# One benchmark item. Carries the authored decomposition + (optionally) its GT.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class QueryItem:
    query_id: str
    status: str                        # "draft" | "verified"
    raw_query_text: str                # no-filter condition embeds THIS (full text)
    vector_query: str                  # filtered condition embeds THIS (semantic part)
    filters: list[Predicate]           # AND-ed together; [] == no filter
    target: dict                       # {video_id, start_s, end_s}
    source: dict = field(default_factory=dict)
    notes: str = ""
    ground_truth: GroundTruth | None = None

    @classmethod
    def from_dict(cls, item: dict) -> "QueryItem":
        d = item["decomposition"]
        return cls(
            query_id=item["query_id"],
            status=item.get("status", "draft"),
            raw_query_text=item["raw_query_text"],
            vector_query=d["vector_query"],
            filters=[Predicate.from_dict(f) for f in d.get("filters", [])],
            target=item.get("target", {}),
            source=item.get("source", {}),
            notes=item.get("notes", ""),
            ground_truth=GroundTruth.from_item(item),
        )


def load_query_set(path: str) -> list[QueryItem]:
    """Load compiled + GT-enriched items from a benchmark.jsonl."""
    items: list[QueryItem] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(QueryItem.from_dict(json.loads(line)))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# What an adapter run produces, per item — RANKED IDS ONLY (decided 2026-07-20;
# scores/distances deliberately not carried — revisit if soft filters are added).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RawResult:
    query_id: str
    filtered_ids: list[str]            # ranked, best-first (filtered condition)
    unfiltered_ids: list[str]          # ranked, best-first (no-filter condition)
    latency_filtered_ms: float
    latency_unfiltered_ms: float

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "filtered_ids": self.filtered_ids,
            "unfiltered_ids": self.unfiltered_ids,
            "latency_filtered_ms": self.latency_filtered_ms,
            "latency_unfiltered_ms": self.latency_unfiltered_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RawResult":
        return cls(**d)


@dataclass
class RawResults:
    system: str                        # adapter.name, e.g. "pgvector"
    k: int                             # retrieval depth of the run
    results: list[RawResult] = field(default_factory=list)

    def __iter__(self) -> Iterator[RawResult]:
        return iter(self.results)

    def write_jsonl(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(json.dumps({"system": self.system, "k": self.k}) + "\n")
            for r in self.results:
                f.write(json.dumps(r.to_dict()) + "\n")

    @classmethod
    def read_jsonl(cls, path: str) -> "RawResults":
        with open(path) as f:
            header = json.loads(f.readline())
            rows = [RawResult.from_dict(json.loads(l)) for l in f if l.strip()]
        return cls(system=header["system"], k=header["k"], results=rows)


# ─────────────────────────────────────────────────────────────────────────────
# Scored output — per item, plus aggregates. Emitted by the Analyzer.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class QueryMetrics:
    query_id: str
    scorable: bool
    recall_filtered: dict[int, float]     # k -> recall@k vs oracle filtered GT
    recall_unfiltered: dict[int, float]   # k -> recall@k vs oracle unfiltered GT
    target_rank_filtered: int | None      # 1-based rank of best target keyframe
    target_rank_unfiltered: int | None
    latency_filtered_ms: float
    latency_unfiltered_ms: float

    @property
    def rr_filtered(self) -> float:
        """Uncapped reciprocal rank. For a capped MRR use Metrics.mrr_filtered(cap)."""
        return _reciprocal_rank(self.target_rank_filtered, None)

    @property
    def rr_unfiltered(self) -> float:
        return _reciprocal_rank(self.target_rank_unfiltered, None)

    def rr_filtered_at(self, cap: int | None) -> float:
        return _reciprocal_rank(self.target_rank_filtered, cap)

    def rr_unfiltered_at(self, cap: int | None) -> float:
        return _reciprocal_rank(self.target_rank_unfiltered, cap)

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "scorable": self.scorable,
            "recall_filtered": {str(k): v for k, v in self.recall_filtered.items()},
            "recall_unfiltered": {str(k): v for k, v in self.recall_unfiltered.items()},
            "target_rank_filtered": self.target_rank_filtered,
            "target_rank_unfiltered": self.target_rank_unfiltered,
            "rr_filtered": self.rr_filtered,
            "rr_unfiltered": self.rr_unfiltered,
            "latency_filtered_ms": self.latency_filtered_ms,
            "latency_unfiltered_ms": self.latency_unfiltered_ms,
        }


@dataclass
class Metrics:
    system: str
    ks: Sequence[int]
    per_query: list[QueryMetrics] = field(default_factory=list)
    # Retrieval depth of the run that produced these (RawResults.k). Carried so a
    # capped MRR can say whether its cap is meaningful: no rank beyond k exists,
    # so MRR@cap for cap >= k is just the uncapped MRR wearing a hat. 0 = unknown.
    retrieval_k: int = 0

    def _scorable(self) -> list[QueryMetrics]:
        return [m for m in self.per_query if m.scorable]

    def cap_is_meaningful(self, cap: int) -> bool:
        """False when the cap is at or above the run's retrieval depth."""
        return self.retrieval_k > 0 and cap < self.retrieval_k

    def mean_recall_filtered(self, k: int) -> float:
        rows = self._scorable()
        return _safe_mean([m.recall_filtered.get(k, 0.0) for m in rows])

    def mean_recall_unfiltered(self, k: int) -> float:
        rows = self._scorable()
        return _safe_mean([m.recall_unfiltered.get(k, 0.0) for m in rows])

    # MRR at a capped rank (Omar, 28-07): a target found beyond `cap` counts as a
    # MISS, not as a small reciprocal. cap=None is the uncapped MRR. The cap models
    # how deep a VBS user would actually look — a target at rank 800 is a miss in
    # practice, but contributes 0.00125 to an uncapped MRR and so hides there.
    def mrr_filtered(self, cap: int | None = None) -> float:
        return _safe_mean([m.rr_filtered_at(cap) for m in self._scorable()])

    def mrr_unfiltered(self, cap: int | None = None) -> float:
        return _safe_mean([m.rr_unfiltered_at(cap) for m in self._scorable()])

    # Latency is a system property independent of scorability, so it is summarised
    # over ALL items (a filtered search still has a real cost when the target fails
    # its own filter). Median over the per-query medians the Runner recorded.
    def median_latency_filtered(self) -> float:
        return _median([m.latency_filtered_ms for m in self.per_query])

    def median_latency_unfiltered(self) -> float:
        return _median([m.latency_unfiltered_ms for m in self.per_query])

    # Tail latency ACROSS queries, not within one. Each per-query number is already
    # the median of `repeat` warm trials (Runner), so this asks "which queries are
    # slow", not "how noisy is one query" — with ~30 items a within-query p95 off 5
    # trials would be noise, while the across-query tail is where a planner cutover
    # or a broad filter shows up.
    def latency_percentile_filtered(self, p: float = 95.0) -> float:
        return _percentile([m.latency_filtered_ms for m in self.per_query], p)

    def latency_percentile_unfiltered(self, p: float = 95.0) -> float:
        return _percentile([m.latency_unfiltered_ms for m in self.per_query], p)

    def write_jsonl(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(json.dumps({"system": self.system, "ks": list(self.ks),
                                "retrieval_k": self.retrieval_k}) + "\n")
            for m in self.per_query:
                f.write(json.dumps(m.to_dict()) + "\n")


def _safe_mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: Iterable[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def _percentile(xs: Iterable[float], p: float) -> float:
    """Linear-interpolated percentile (numpy's default), so p=50 == _median and the
    figures agree with the printed summary. Kept dependency-free: schema.py is the
    seam both halves of the suite import, and it stays stdlib-only."""
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return xs[0]
    pos = (n - 1) * max(0.0, min(100.0, p)) / 100.0
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _reciprocal_rank(rank: int | None, cap: int | None) -> float:
    """1/rank, or 0.0 when the target was never found (rank None) or was found
    deeper than `cap` — a capped MRR treats too-deep as a miss, not as a tiny hit."""
    if rank is None or rank < 1:
        return 0.0
    if cap is not None and rank > cap:
        return 0.0
    return 1.0 / rank

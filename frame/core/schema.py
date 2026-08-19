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
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence

from .version import HARNESS_CONTRACT

if TYPE_CHECKING:      # only for the annotation; avoids a runtime import cycle
    from .version import BenchmarkVersion


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
# The 2x2 condition matrix (Omar, 28-07-2026). Two axes:
#   TEXT      raw_query_text (the full user phrasing) vs vector_query (the isolated
#             semantic remainder, with the filterable attribute taken out)
#   PREDICATE the AND-ed structured filter applied, or not
#
# Every cell is one Condition, and everything downstream derives from this table:
# the Runner reads `text_attr` off the QueryItem to know what to embed, the
# Analyzer reads `gt_attr` off the GroundTruth to know what exact answer to score
# against. Adding a fifth condition is one row here plus its oracle GT — no
# changes in the Runner, Analyzer, or figures.
#
# The two DIAGONAL cells (semantic+filter, raw+nofilter) were the only ones the
# Runner produced before 2026-08-04; results files from then carry just those two.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Condition:
    name: str            # stable key in results files — do not rename casually
    text_attr: str       # QueryItem attribute holding the text to embed
    filtered: bool       # apply the item's predicate?
    gt_attr: str         # GroundTruth attribute holding this cell's exact answer


CONDITIONS: tuple[Condition, ...] = (
    Condition("raw+nofilter",      "raw_query_text", False, "gt_nofilter"),
    Condition("raw+filter",        "raw_query_text", True,  "gt_raw_filtered"),
    Condition("semantic+nofilter", "vector_query",   False, "gt_vec_nofilter"),
    Condition("semantic+filter",   "vector_query",   True,  "gt_filtered"),
)
CONDITION_NAMES: tuple[str, ...] = tuple(c.name for c in CONDITIONS)
BY_NAME: dict[str, Condition] = {c.name: c for c in CONDITIONS}

# The pair the suite led with before the full 2x2 — "push the attribute into a
# filter" vs "leave it in the CLIP query". Still the headline comparison, and the
# mapping legacy two-condition results files are read back as.
PRIMARY_FILTERED = "semantic+filter"
PRIMARY_UNFILTERED = "raw+nofilter"


# ─────────────────────────────────────────────────────────────────────────────
# Selectivity subdivision of the FILTER workload (Omar, 2026-08-10): split the
# filtered queries into a "selective" and a "relaxed" bucket, so the filter delta
# can be read separately for tight vs broad predicates (broad filters are also
# where pgvector's planner keeps the approximate HNSW path — see the profiler).
#
# The boundary is the CONJUNCTION selectivity (fraction of the corpus the full
# AND-ed predicate keeps), computed by the oracle and stored per item. It was set
# EMPIRICALLY from the v3c1 distribution: sorted, there is a clean gap between
# 2.66% and 5.19% (18 queries below, 12 above), so the cut sits in that gap.
# Selectivity is CORPUS-RELATIVE — re-derive this against the union distribution
# once the corpus changes. One number, one place.
SELECTIVE_MAX = 0.03
SELECTIVITY_CLASSES = ("selective", "relaxed")


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
    # One exact answer per condition (see CONDITIONS above). All four are computed
    # by the same oracle pass, so no cell is scored against a stand-in.
    gt_filtered: list[str] | None      # vector_query + filter    (semantic+filter)
    gt_nofilter: list[str] | None      # raw_query_text, no filter (raw+nofilter)
    gt_raw_filtered: list[str] | None  # raw_query_text + filter   (raw+filter)
    gt_vec_nofilter: list[str] | None  # vector_query, no filter   (semantic+nofilter);
                                       # also the neighbour list the filtered HNSW walk
                                       # traverses, used by the profiler for pass-rate
    # Corpus fraction the FULL (AND-ed) predicate keeps at the pinned thresholds —
    # the conjunction selectivity, computed by the oracle. None until GT is run
    # with thresholds (and for no-filter items). Drives selectivity_class().
    filter_selectivity_conjunction: float | None = None

    @property
    def is_scorable(self) -> bool:
        """Whether the PRIMARY filtered cell can be scored. Kept as the historical
        headline gate; per-cell scorability is `scorable_for`."""
        return self.scorable_for(BY_NAME[PRIMARY_FILTERED])

    def gt_for(self, cond: Condition) -> list[str] | None:
        return getattr(self, cond.gt_attr)

    @property
    def is_harm_exemplar(self) -> bool:
        """A FILTERED item whose own target is excluded by its own filter. Derived
        from GT, so it needs no re-authoring: target_passes_filter is None for a
        no-filter item and True when the target survives, so exactly `is False`
        marks the filter-harm exemplars (q0003/06/13/20, and any future item whose
        target its filter drops). These are the discriminator between exact and
        fuzzy filtering — a hard filter drops the target, a soft one could keep it
        (Omar, 2026-08-03/10)."""
        return self.target_passes_filter is False

    def selectivity_class(self) -> str | None:
        """"selective" | "relaxed" by conjunction selectivity, or None when it has
        not been computed (no-filter items, or GT run without thresholds). Boundary:
        schema.SELECTIVE_MAX (set empirically — see the note there)."""
        s = self.filter_selectivity_conjunction
        if s is None:
            return None
        return "selective" if s < SELECTIVE_MAX else "relaxed"

    def scorable_for(self, cond: Condition) -> bool:
        """A cell is scorable once its own exact answer exists and — for the two
        FILTER cells only — the target survives the filter, else Recall@k is 0 by
        construction. The no-filter cells are deliberately NOT gated on
        target_passes_filter: whether a predicate would have excluded the target
        says nothing about a run that applied no predicate, and gating them would
        throw away a valid measurement (e.g. the OCR harm exemplar q0006, whose
        target is rank 1 unfiltered)."""
        if not self.gt_for(cond):
            return False
        if cond.filtered and self.target_passes_filter is not True:
            return False
        return True

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
            gt_raw_filtered=c.get("geometric_gt_raw_filtered"),
            gt_vec_nofilter=c.get("geometric_gt_vec_nofilter"),
            filter_selectivity_conjunction=c.get("filter_selectivity_conjunction"),
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
    # Real human phrasings of THIS task, each {team, action, text}. Carried as
    # metadata for the 2x2 run; only graded as queries in their own right under the
    # opt-in variant facet (Runner.grade_variants — see VariantResult). Empty for
    # authored-only items.
    user_query_variants: list[dict] = field(default_factory=list)
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
            user_query_variants=item.get("user_query_variants") or [],
            ground_truth=GroundTruth.from_item(item),
        )


@dataclass
class QuerySet:
    """A benchmark.jsonl: its items plus the version marker identifying them.

    `version` is None only for a file written before versioning existed — which is
    itself a fact worth carrying, since results scored against it cannot be shown
    comparable to anything.
    """

    items: list[QueryItem] = field(default_factory=list)
    version: "BenchmarkVersion | None" = None

    def __iter__(self) -> Iterator[QueryItem]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


def load_benchmark(path: str) -> QuerySet:
    """Load a benchmark.jsonl with its version header (if it has one)."""
    from .version import BenchmarkVersion

    version = None
    items: list[QueryItem] = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # The header is the first line and has no query_id. Detecting it by
            # content rather than by position keeps pre-versioning files loadable.
            if i == 0 and "query_id" not in d:
                version = BenchmarkVersion.from_dict(d)
                continue
            items.append(QueryItem.from_dict(d))
    return QuerySet(items=items, version=version)


def load_query_set(path: str) -> list[QueryItem]:
    """Just the items. Use load_benchmark() when the version marker matters."""
    return load_benchmark(path).items


# ─────────────────────────────────────────────────────────────────────────────
# The per-variant facet (Omar, 2026-08-18): grade every REAL human phrasing of a
# task as its own query, so the semantic axis is exercised by many real wordings
# rather than the single authored decomposition. Each phrasing runs "nofilter" (does
# this wording find the target at all?) and, when the item has a predicate, "filter"
# (the task's shared filter — the filter is task-level, so this is not a new filter
# measurement, only the same filter under many phrasings). Ranked ids only, exactly
# like a Condition, but keyed "nofilter"/"filter" since here the VARYING thing is the
# text, not a QueryItem attribute. Scored to a target RANK only: there is no
# per-phrasing exact k-NN, so no Recall@k (that would need an oracle pass per
# phrasing). Opt-in (Runner.grade_variants) — off, the file is byte-identical to a
# 2x2-only run, so the harness contract is unchanged.
# ─────────────────────────────────────────────────────────────────────────────
VARIANT_CONDS: tuple[str, ...] = ("nofilter", "filter")


@dataclass
class VariantResult:
    index: int                          # position in the item's user_query_variants
    team: str
    action: str
    text: str
    ids: dict[str, list[str]] = field(default_factory=dict)  # VARIANT_CONDS -> ranked ids

    def to_dict(self) -> dict:
        return {"index": self.index, "team": self.team, "action": self.action,
                "text": self.text, "ids": self.ids}

    @classmethod
    def from_dict(cls, d: dict) -> "VariantResult":
        return cls(index=d["index"], team=d.get("team", ""), action=d.get("action", ""),
                   text=d.get("text", ""), ids=d.get("ids", {}))


# ─────────────────────────────────────────────────────────────────────────────
# What an adapter run produces, per item — RANKED IDS ONLY (decided 2026-07-20;
# scores/distances deliberately not carried — revisit if soft filters are added).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RawResult:
    query_id: str
    # condition name -> ranked ids / warm median latency. Keyed rather than four
    # flat fields so a condition can be absent (an item with no predicate has no
    # meaningful FILTER cell) and so adding a condition doesn't reshape the file.
    ids: dict[str, list[str]] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    # Per-phrasing rankings, present only under the opt-in variant facet. Absent (an
    # empty list) for a plain 2x2 run, and then dropped from to_dict entirely so the
    # results file is unchanged.
    variants: list[VariantResult] = field(default_factory=list)

    def conditions(self) -> list[str]:
        return [c for c in CONDITION_NAMES if c in self.ids]

    def to_dict(self) -> dict:
        d: dict = {
            "query_id": self.query_id,
            "ids": self.ids,
            "latency_ms": self.latency_ms,
        }
        if self.variants:
            d["variants"] = [v.to_dict() for v in self.variants]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RawResult":
        if "ids" in d:
            return cls(query_id=d["query_id"], ids=d["ids"],
                       latency_ms=d.get("latency_ms", {}),
                       variants=[VariantResult.from_dict(v) for v in d.get("variants", [])])
        # Legacy two-condition file (pre-2026-08-04): `filtered` meant
        # vector_query + predicate, `unfiltered` meant raw_query_text alone.
        # Read it back as exactly those two cells so old runs stay analysable —
        # they simply have two of the four.
        return cls(
            query_id=d["query_id"],
            ids={PRIMARY_FILTERED: d["filtered_ids"],
                 PRIMARY_UNFILTERED: d["unfiltered_ids"]},
            latency_ms={PRIMARY_FILTERED: d["latency_filtered_ms"],
                        PRIMARY_UNFILTERED: d["latency_unfiltered_ms"]},
        )


@dataclass
class RawResults:
    system: str                        # adapter.name, e.g. "pgvector"
    k: int                             # retrieval depth of the run
    results: list[RawResult] = field(default_factory=list)
    # What this run was measured against. Without these two a results file cannot
    # be shown comparable to any other — see frame/core/version.py.
    benchmark: "BenchmarkVersion | None" = None
    # Defaults to the CURRENT contract because an in-memory RawResults was, by
    # definition, just produced by this harness. Only a file can carry 0, and only
    # when it was written before versioning existed — which is the case worth
    # catching (see RawResults.read_jsonl).
    harness_contract: int = HARNESS_CONTRACT
    # Search-beam width the run used (hnsw.ef_search / hnswlib ef_search). Carried so
    # a k×ef sweep's artifacts are self-describing — one metrics/raw file names the
    # (k, ef) cell it came from without parsing the filename. None on older files and
    # on runs that left the adapter's default ef in place.
    ef_search: int | None = None

    def __iter__(self) -> Iterator[RawResult]:
        return iter(self.results)

    def write_jsonl(self, path: str) -> None:
        header: dict = {"system": self.system, "k": self.k,
                        "harness_contract": self.harness_contract}
        if self.benchmark is not None:
            header["benchmark"] = self.benchmark.to_dict()
        if self.ef_search is not None:
            header["ef_search"] = self.ef_search
        with open(path, "w") as f:
            f.write(json.dumps(header) + "\n")
            for r in self.results:
                f.write(json.dumps(r.to_dict()) + "\n")

    @classmethod
    def read_jsonl(cls, path: str) -> "RawResults":
        from .version import BenchmarkVersion

        with open(path) as f:
            header = json.loads(f.readline())
            rows = [RawResult.from_dict(json.loads(l)) for l in f if l.strip()]
        bench = header.get("benchmark")
        return cls(system=header["system"], k=header["k"], results=rows,
                   benchmark=BenchmarkVersion.from_dict(bench) if bench else None,
                   harness_contract=header.get("harness_contract", 0),
                   ef_search=header.get("ef_search"))


# ─────────────────────────────────────────────────────────────────────────────
# Scored output — per item, plus aggregates. Emitted by the Analyzer.
# ─────────────────────────────────────────────────────────────────────────────
# The default gate for "succeeding phrasings" (Omar, 2026-08-18): a phrasing counts
# as succeeding if it found the target within this rank WITHOUT a filter — i.e. the
# wording could at least locate the target on its own. The two-version MRR (all
# phrasings vs succeeding-only) separates "the query was too weak/vague" from a
# filter/system effect, so a task with many hopeless phrasings does not drag the
# MRR into looking like a system failure.
SUCCEEDING_CAP = 100


@dataclass
class VariantMetric:
    """One human phrasing's task-success score. Ranks keyed by VARIANT_CONDS
    ("nofilter"/"filter"); no Recall — there is no per-phrasing exact GT."""

    index: int
    team: str
    action: str
    text: str
    target_rank: dict[str, int | None] = field(default_factory=dict)

    def rr(self, cond: str, cap: int | None = None) -> float:
        return _reciprocal_rank(self.target_rank.get(cond), cap)

    def succeeds(self, cap: int = SUCCEEDING_CAP) -> bool:
        """Found the target within `cap` with NO filter — the succeeding-only gate."""
        r = self.target_rank.get("nofilter")
        return r is not None and r <= cap

    def to_dict(self) -> dict:
        return {"index": self.index, "team": self.team, "action": self.action,
                "text": self.text, "target_rank": self.target_rank,
                "rr": {c: self.rr(c) for c in self.target_rank}}


@dataclass
class QueryMetrics:
    """One item's scores, per condition. Every dict is keyed by condition name and
    holds only the conditions the run actually produced for this item."""

    query_id: str
    scorable: dict[str, bool]                   # condition -> cell has valid GT
    recall: dict[str, dict[int, float]]         # condition -> k -> recall@k
    target_rank: dict[str, int | None]          # condition -> 1-based rank of target
    latency_ms: dict[str, float]                # condition -> warm median latency
    # Grouping keys, carried so the metrics file is self-describing (plots and
    # subgroup summaries bucket without re-reading the benchmark GT). None when the
    # item has no filter / no computed selectivity.
    selectivity: float | None = None            # conjunction selectivity of the filter
    harm_exemplar: bool = False                 # target excluded by its own filter
    # Per-phrasing scores, present only under the variant facet (empty otherwise).
    variants: list[VariantMetric] = field(default_factory=list)

    def selectivity_class(self) -> str | None:
        if self.selectivity is None:
            return None
        return "selective" if self.selectivity < SELECTIVE_MAX else "relaxed"

    def conditions(self) -> list[str]:
        return [c for c in CONDITION_NAMES if c in self.target_rank]

    def is_scorable(self, cond: str) -> bool:
        return self.scorable.get(cond, False)

    def rr(self, cond: str, cap: int | None = None) -> float:
        """Reciprocal rank in one condition; a target deeper than `cap` is a miss."""
        return _reciprocal_rank(self.target_rank.get(cond), cap)

    def to_dict(self) -> dict:
        d = {
            "query_id": self.query_id,
            "scorable": self.scorable,
            "recall": {c: {str(k): v for k, v in ks.items()}
                       for c, ks in self.recall.items()},
            "target_rank": self.target_rank,
            "rr": {c: self.rr(c) for c in self.conditions()},
            "latency_ms": self.latency_ms,
            "selectivity": self.selectivity,
            "harm_exemplar": self.harm_exemplar,
        }
        if self.variants:
            d["variants"] = [v.to_dict() for v in self.variants]
        return d


@dataclass
class Metrics:
    system: str
    ks: Sequence[int]
    per_query: list[QueryMetrics] = field(default_factory=list)
    # Retrieval depth of the run that produced these (RawResults.k). Carried so a
    # capped MRR can say whether its cap is meaningful: no rank beyond k exists,
    # so MRR@cap for cap >= k is just the uncapped MRR wearing a hat. 0 = unknown.
    retrieval_k: int = 0
    # Provenance, carried through from the run so a metrics file alone says what it
    # may be compared with (frame/core/version.py).
    benchmark: "BenchmarkVersion | None" = None
    harness_contract: int = HARNESS_CONTRACT
    # Search-beam width of the run these scores came from (RawResults.ef_search).
    # Carried through analyze() so a metrics file names its (k, ef) sweep cell.
    ef_search: int | None = None

    def conditions(self) -> list[str]:
        """Conditions this run produced, in canonical order."""
        seen = {c for m in self.per_query for c in m.conditions()}
        return [c for c in CONDITION_NAMES if c in seen]

    def restricted(self, ids: Iterable[str]) -> "Metrics":
        """A view over just `ids` — same provenance, subset of per-query rows. Lets
        every aggregate (recall / MRR / latency / comparable) be reported over a
        named subgroup (a selectivity bucket, the harm exemplars) with no special
        casing: subset, then reuse the existing methods."""
        keep = set(ids)
        return Metrics(
            system=self.system, ks=self.ks,
            per_query=[m for m in self.per_query if m.query_id in keep],
            retrieval_k=self.retrieval_k, benchmark=self.benchmark,
            harness_contract=self.harness_contract, ef_search=self.ef_search,
        )

    def cap_is_meaningful(self, cap: int) -> bool:
        """False when the cap is at or above the run's retrieval depth."""
        return self.retrieval_k > 0 and cap < self.retrieval_k

    # ── which items an aggregate covers ────────────────────────────────────────
    # A cross-condition comparison is only honest over ONE common subset of items.
    # Conditions have different scorable sets: the two no-filter cells are scorable
    # for every item with GT, the two filter cells only for items that HAVE a filter
    # whose target survives it. Averaging each condition over its own subset would
    # compare a 37-item mean against a 30-item mean and read the difference as an
    # effect of the condition. So `common=True` (the default) restricts every
    # aggregate to items scorable in ALL of `conditions`.
    def comparable(self, conditions: Sequence[str] | None = None) -> list[QueryMetrics]:
        conds = list(conditions) if conditions is not None else self.conditions()
        return [m for m in self.per_query if all(m.is_scorable(c) for c in conds)]

    def _rows(self, cond: str, common: bool) -> list[QueryMetrics]:
        if common:
            return self.comparable()
        return [m for m in self.per_query if m.is_scorable(cond)]

    def mean_recall(self, cond: str, k: int, common: bool = True) -> float:
        return _safe_mean([m.recall.get(cond, {}).get(k, 0.0)
                           for m in self._rows(cond, common)])

    # MRR at a capped rank (Omar, 28-07): a target found beyond `cap` counts as a
    # MISS, not as a small reciprocal. cap=None is the uncapped MRR. The cap models
    # how deep a VBS user would actually look — a target at rank 800 is a miss in
    # practice, but contributes 0.00125 to an uncapped MRR and so hides there.
    def mrr(self, cond: str, cap: int | None = None, common: bool = True) -> float:
        return _safe_mean([m.rr(cond, cap) for m in self._rows(cond, common)])

    # ── per-variant aggregation (the "grade every phrasing" facet) ──────────────
    # These pool over PHRASINGS, not items: the unit is one real user wording. The
    # `common`/`comparable` item-subset logic does not apply — a variant is scored
    # on target rank alone (no per-cell GT to gate on), so the only gate is the
    # succeeding filter below. cond is a VARIANT_CONDS key ("nofilter"/"filter").
    def has_variants(self) -> bool:
        return any(m.variants for m in self.per_query)

    def variant_rows(self) -> list["VariantMetric"]:
        return [v for m in self.per_query for v in m.variants]

    def variant_mrr(self, cond: str, cap: int | None = None,
                    succeeding_cap: int | None = None) -> float:
        """MRR over all graded phrasings for `cond`. With `succeeding_cap` set,
        restrict to phrasings whose NO-FILTER rank is within it — the 'the query
        could at least find it unfiltered' subset (Omar's two-version MRR)."""
        rows = self.variant_rows()
        if succeeding_cap is not None:
            rows = [v for v in rows if v.succeeds(succeeding_cap)]
        return _safe_mean([v.rr(cond, cap) for v in rows])

    def variant_n(self, succeeding_cap: int | None = None) -> int:
        rows = self.variant_rows()
        if succeeding_cap is not None:
            rows = [v for v in rows if v.succeeds(succeeding_cap)]
        return len(rows)

    def variant_mrr_by_task(self, cond: str, cap: int | None = None,
                            succeeding_cap: int | None = None) -> dict[str, float]:
        """Per-task variant MRR — one number per item, the boxplot's central tendency."""
        out: dict[str, float] = {}
        for m in self.per_query:
            rows = m.variants
            if succeeding_cap is not None:
                rows = [v for v in rows if v.succeeds(succeeding_cap)]
            if rows:
                out[m.query_id] = _safe_mean([v.rr(cond, cap) for v in rows])
        return out

    def variant_task_rr(self, cond: str, cap: int | None = None) -> dict[str, list[float]]:
        """Per-task list of per-phrasing reciprocal ranks — the boxplot payload
        (one distribution per task)."""
        return {m.query_id: [v.rr(cond, cap) for v in m.variants]
                for m in self.per_query if m.variants}

    # Latency is a system property independent of scorability, so it is summarised
    # over every item that RAN the condition — a filtered search still has a real
    # cost when the target fails its own filter. Median over the per-query medians
    # the Runner recorded.
    def _latencies(self, cond: str) -> list[float]:
        return [m.latency_ms[cond] for m in self.per_query if cond in m.latency_ms]

    def latency_n(self, cond: str) -> int:
        """How many items actually ran this condition — differs between the filter
        and no-filter cells, since items without a predicate skip the filter cells."""
        return len(self._latencies(cond))

    def median_latency(self, cond: str) -> float:
        return _median(self._latencies(cond))

    # Tail latency ACROSS queries, not within one. Each per-query number is already
    # the median of `repeat` warm trials (Runner), so this asks "which queries are
    # slow", not "how noisy is one query" — with ~30 items a within-query p95 off 5
    # trials would be noise, while the across-query tail is where a planner cutover
    # or a broad filter shows up.
    def latency_percentile(self, cond: str, p: float = 95.0) -> float:
        return _percentile(self._latencies(cond), p)

    def write_jsonl(self, path: str) -> None:
        header: dict = {
            "system": self.system,
            "ks": list(self.ks),
            "retrieval_k": self.retrieval_k,
            "conditions": self.conditions(),
            "n_comparable": len(self.comparable()),
            "harness_contract": self.harness_contract,
        }
        if self.benchmark is not None:
            header["benchmark"] = self.benchmark.to_dict()
        if self.ef_search is not None:
            header["ef_search"] = self.ef_search
        with open(path, "w") as f:
            f.write(json.dumps(header) + "\n")
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

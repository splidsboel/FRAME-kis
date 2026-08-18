"""Unit tests for the per-variant grading facet (Omar, 2026-08-18): grade every
real human phrasing of a task as its own query, task-success (MRR) only.

Covers the whole seam — QueryItem loads variants, the Runner produces them behind
the opt-in flag, the Analyzer scores them to target ranks, and Metrics aggregates
the all-vs-succeeding MRR the report leads with.
"""

from __future__ import annotations

import copy

from frame.core.analyzer import Analyzer
from frame.core.runner import Runner
from frame.core.schema import (
    Metrics,
    QueryItem,
    QueryMetrics,
    RawResult,
    VariantMetric,
    VariantResult,
)

NOFILT_V, FILT_V = "nofilter", "filter"


def _with_variants(item: dict, texts) -> dict:
    """A copy of `item` carrying user_query_variants for the given phrasings."""
    d = copy.deepcopy(item)
    d["user_query_variants"] = [
        {"team": f"t{i}", "action": "textQuery", "text": t} for i, t in enumerate(texts)
    ]
    return d


# ─── schema: QueryItem carries variants, RawResult round-trips them ──────────

def test_queryitem_loads_variants(enriched_item):
    item = QueryItem.from_dict(_with_variants(enriched_item, ["a wording", "another"]))
    assert [v["text"] for v in item.user_query_variants] == ["a wording", "another"]


def test_queryitem_variants_default_empty(enriched_item):
    assert QueryItem.from_dict(enriched_item).user_query_variants == []


def test_rawresult_variants_roundtrip():
    r = RawResult(
        query_id="q1",
        ids={"raw+nofilter": ["a"]},
        variants=[VariantResult(0, "prak1", "textQuery", "hi", {NOFILT_V: ["x", "kf"]})],
    )
    d = r.to_dict()
    assert "variants" in d and d["variants"][0]["text"] == "hi"
    back = RawResult.from_dict(d)
    assert back.variants[0].ids == {NOFILT_V: ["x", "kf"]}
    assert back.variants[0].team == "prak1"


def test_rawresult_omits_variants_key_when_empty():
    # a plain 2x2 run must serialise byte-identically to before the facet existed
    assert "variants" not in RawResult(query_id="q1", ids={"raw+nofilter": ["a"]}).to_dict()


# ─── runner: produces one VariantResult per phrasing, opt-in ─────────────────

def test_runner_off_by_default_produces_no_variants(enriched_item, fake_adapter, fake_encoder):
    item = QueryItem.from_dict(_with_variants(enriched_item, ["one", "two"]))
    raw = Runner(fake_adapter, fake_encoder, warmup=0, repeat=1).run([item], k=5)
    assert raw.results[0].variants == []


def test_runner_grades_each_phrasing_nofilter_and_filter(
    enriched_item, fake_adapter, fake_encoder
):
    item = QueryItem.from_dict(_with_variants(enriched_item, ["one", "two", "three"]))
    raw = Runner(fake_adapter, fake_encoder, warmup=0, repeat=1,
                 grade_variants=True).run([item], k=5)
    vs = raw.results[0].variants
    assert [v.text for v in vs] == ["one", "two", "three"]
    # the item has a predicate, so each phrasing runs BOTH cells
    assert all(set(v.ids) == {NOFILT_V, FILT_V} for v in vs)
    # canned adapter: filtered vs no-filter rankings differ per its filters arg
    assert vs[0].ids[NOFILT_V] == ["kf_x", "kf_target", "kf_y"]
    assert vs[0].ids[FILT_V] == ["kf_target", "kf_a", "kf_z"]


def test_runner_variant_nofilter_only_without_predicate(
    enriched_item, fake_adapter, fake_encoder
):
    nofilt = _with_variants(enriched_item, ["one"])
    nofilt["decomposition"]["filters"] = []
    item = QueryItem.from_dict(nofilt)
    raw = Runner(fake_adapter, fake_encoder, warmup=0, repeat=1,
                 grade_variants=True).run([item], k=5)
    assert set(raw.results[0].variants[0].ids) == {NOFILT_V}


def test_runner_variant_encodes_each_phrasing(enriched_item, fake_adapter, fake_encoder):
    item = QueryItem.from_dict(_with_variants(enriched_item, ["alpha", "beta"]))
    Runner(fake_adapter, fake_encoder, warmup=0, repeat=1,
           grade_variants=True).run([item], k=5)
    assert "alpha" in fake_encoder.calls and "beta" in fake_encoder.calls


# ─── analyzer: scores variants to target ranks (end-to-end) ─────────────────

def test_analyzer_scores_variant_target_ranks(enriched_item, fake_adapter, fake_encoder):
    item = QueryItem.from_dict(_with_variants(enriched_item, ["one", "two"]))
    raw = Runner(fake_adapter, fake_encoder, warmup=0, repeat=1,
                 grade_variants=True).run([item], k=5)
    m = Analyzer().analyze(raw, [item])
    vms = m.per_query[0].variants
    assert len(vms) == 2
    # target "kf_target": no-filter ranking [kf_x, kf_target, kf_y] -> rank 2;
    # filtered ranking [kf_target, ...] -> rank 1
    assert vms[0].target_rank == {NOFILT_V: 2, FILT_V: 1}
    assert vms[0].rr(NOFILT_V) == 0.5 and vms[0].rr(FILT_V) == 1.0


# ─── aggregation: all vs succeeding-only MRR ────────────────────────────────

def _qm_with_variants(qid, ranks):
    """A QueryMetrics carrying only variant scores; `ranks` is a list of
    {nofilter, filter} rank dicts."""
    variants = [VariantMetric(i, "t", "textQuery", f"p{i}", r) for i, r in enumerate(ranks)]
    return QueryMetrics(qid, {}, {}, {}, {}, variants=variants)


def test_variant_mrr_all_vs_succeeding():
    # three phrasings: two find it unfiltered (r2, r4), one misses (None).
    m = Metrics(system="s", ks=(5,), retrieval_k=1000, per_query=[
        _qm_with_variants("q1", [
            {NOFILT_V: 2, FILT_V: 1},
            {NOFILT_V: 4, FILT_V: 2},
            {NOFILT_V: None, FILT_V: None},
        ])
    ])
    # ALL: mean of 1/2, 1/4, 0 = 0.25 ; SUCCEEDING (nofilter rank<=100): mean(1/2,1/4)=0.375
    assert m.variant_mrr(NOFILT_V) == (0.5 + 0.25 + 0.0) / 3
    assert m.variant_mrr(NOFILT_V, succeeding_cap=100) == (0.5 + 0.25) / 2
    assert m.variant_n() == 3
    assert m.variant_n(succeeding_cap=100) == 2


def test_variant_succeeding_cap_gates_on_nofilter_only():
    # a phrasing whose target is deep unfiltered (rank 500) does NOT succeed even if
    # its FILTER rank is 1 — the gate is "findable without the filter"
    v = VariantMetric(0, "t", "q", "p", {NOFILT_V: 500, FILT_V: 1})
    assert v.succeeds(100) is False


def test_variant_mrr_by_task_and_boxplot_payload():
    m = Metrics(system="s", ks=(5,), retrieval_k=1000, per_query=[
        _qm_with_variants("q1", [{NOFILT_V: 1}, {NOFILT_V: 2}]),   # RRs 1.0, 0.5
        _qm_with_variants("q2", [{NOFILT_V: 4}]),                  # RR 0.25
    ])
    by_task = m.variant_mrr_by_task(NOFILT_V)
    assert by_task["q1"] == 0.75 and by_task["q2"] == 0.25
    rr = m.variant_task_rr(NOFILT_V)
    assert rr["q1"] == [1.0, 0.5] and rr["q2"] == [0.25]


# ─── report: renders and is empty without variants ──────────────────────────

def test_variant_summary_empty_without_variants():
    m = Metrics(system="s", ks=(5,), retrieval_k=1000,
                per_query=[QueryMetrics("q1", {}, {}, {}, {})])
    assert Analyzer().variant_summary(m) == ""


def test_variant_summary_renders(enriched_item, fake_adapter, fake_encoder):
    item = QueryItem.from_dict(_with_variants(enriched_item, ["one", "two"]))
    raw = Runner(fake_adapter, fake_encoder, warmup=0, repeat=1,
                 grade_variants=True).run([item], k=1000)
    m = Analyzer().analyze(raw, [item])
    out = Analyzer().variant_summary(m)
    assert "Per-phrasing grading" in out
    assert "MRR succ" in out
    assert "q0001" in out

"""Unit tests for frame.core.schema — the on-disk contract + metric aggregation."""

from __future__ import annotations

from frame.core.version import HARNESS_CONTRACT
from frame.core.schema import (
    BY_NAME,
    CONDITION_NAMES,
    CONDITIONS,
    GroundTruth,
    Metrics,
    Predicate,
    QueryItem,
    QueryMetrics,
    RawResult,
    RawResults,
    _median,
    _safe_mean,
    load_query_set,
)


# ─── Predicate ──────────────────────────────────────────────────────────────

def test_predicate_from_dict_full(authored_item):
    f = authored_item["decomposition"]["filters"][0]
    p = Predicate.from_dict(f)
    assert p.filter_type == "scene"
    assert p.attribute == "scene_label"
    assert p.op == "in"
    assert p.value == ["night"]
    assert p.vocab == "places365"
    assert p.verified is True


def test_predicate_from_dict_defaults():
    p = Predicate.from_dict(
        {"filter_type": "object", "attribute": "object_label", "op": "in", "value": ["fish"]}
    )
    assert p.vocab is None
    assert p.mapping_source is None
    assert p.verified is False


def test_predicate_is_frozen():
    import dataclasses
    import pytest

    p = Predicate("scene", "scene_label", "in", ["night"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.filter_type = "object"  # type: ignore[misc]


# ─── GroundTruth ────────────────────────────────────────────────────────────

def test_ground_truth_from_item(enriched_item):
    gt = GroundTruth.from_item(enriched_item)
    assert gt.query_id == "q0001"
    assert gt.target_keyframe_ids == ["kf_target"]
    assert gt.target_passes_filter is True
    assert gt.filter_selectivity == [0.1]
    assert gt.gt_filtered == ["kf_target", "kf_a", "kf_b", "kf_c"]
    assert gt.gt_nofilter == ["kf_x", "kf_y", "kf_z"]
    assert gt.gt_raw_filtered == ["kf_target", "kf_a", "kf_m"]
    assert gt.gt_vec_nofilter == ["kf_target", "kf_p", "kf_q", "kf_r"]


def test_ground_truth_gt_for_every_condition(enriched_item):
    """Each 2x2 cell must resolve to its OWN oracle answer — a wrong wiring here
    would silently score a condition against another condition's truth."""
    gt = GroundTruth.from_item(enriched_item)
    got = {c.name: gt.gt_for(c) for c in CONDITIONS}
    assert got == {
        "raw+nofilter": ["kf_x", "kf_y", "kf_z"],
        "raw+filter": ["kf_target", "kf_a", "kf_m"],
        "semantic+nofilter": ["kf_target", "kf_p", "kf_q", "kf_r"],
        "semantic+filter": ["kf_target", "kf_a", "kf_b", "kf_c"],
    }
    assert len({tuple(v) for v in got.values()}) == 4    # all distinct


def test_scorable_for_gates_only_the_filter_cells(enriched_item):
    enriched_item["computed"]["target_passes_filter"] = False
    gt = GroundTruth.from_item(enriched_item)
    by_name = {c.name: gt.scorable_for(c) for c in CONDITIONS}
    assert by_name["semantic+filter"] is False
    assert by_name["raw+filter"] is False
    # no predicate was applied in these, so the target's filter fate is irrelevant
    assert by_name["raw+nofilter"] is True
    assert by_name["semantic+nofilter"] is True


def test_scorable_for_false_without_that_cells_gt(enriched_item):
    enriched_item["computed"]["geometric_gt_raw_filtered"] = None
    gt = GroundTruth.from_item(enriched_item)
    assert gt.scorable_for(BY_NAME["raw+filter"]) is False
    assert gt.scorable_for(BY_NAME["semantic+filter"]) is True


def test_ground_truth_from_item_no_computed(authored_item):
    gt = GroundTruth.from_item(authored_item)
    assert gt.gt_filtered is None
    assert gt.target_passes_filter is None
    assert gt.filter_selectivity == []


def test_is_scorable_true(enriched_item):
    assert GroundTruth.from_item(enriched_item).is_scorable is True


def test_is_scorable_false_when_target_fails_filter(enriched_item):
    enriched_item["computed"]["target_passes_filter"] = False
    assert GroundTruth.from_item(enriched_item).is_scorable is False


def test_is_scorable_false_when_no_filtered_gt(enriched_item):
    enriched_item["computed"]["geometric_gt_filtered"] = None
    assert GroundTruth.from_item(enriched_item).is_scorable is False


def test_is_scorable_false_when_target_passes_is_none(enriched_item):
    enriched_item["computed"]["target_passes_filter"] = None
    assert GroundTruth.from_item(enriched_item).is_scorable is False


# ─── QueryItem ──────────────────────────────────────────────────────────────

def test_query_item_from_dict(enriched_item):
    q = QueryItem.from_dict(enriched_item)
    assert q.query_id == "q0001"
    assert q.status == "verified"
    assert q.raw_query_text.startswith("a red car")
    assert q.vector_query == "a red car driving on a highway"
    assert len(q.filters) == 1
    assert isinstance(q.filters[0], Predicate)
    assert q.target["video_id"] == "00123"
    assert q.notes == "a note"
    assert q.ground_truth is not None
    assert q.ground_truth.is_scorable is True


def test_query_item_from_dict_defaults():
    item = {
        "query_id": "q0002",
        "raw_query_text": "text",
        "decomposition": {"vector_query": "v"},
        "target": {},
    }
    q = QueryItem.from_dict(item)
    assert q.status == "draft"       # default
    assert q.filters == []           # no filters => no-filter item
    assert q.source == {}
    assert q.notes == ""


# ─── RawResult / RawResults round-trips ─────────────────────────────────────

def _rr(qid, **by_cond):
    return RawResult(query_id=qid, ids=dict(by_cond),
                     latency_ms={c: 1.5 for c in by_cond})


def test_raw_result_roundtrip():
    r = _rr("q1", **{"semantic+filter": ["a", "b"], "raw+nofilter": ["c", "d"]})
    assert RawResult.from_dict(r.to_dict()) == r


def test_raw_result_conditions_in_canonical_order():
    r = _rr("q1", **{c: ["a"] for c in reversed(CONDITION_NAMES)})
    assert r.conditions() == list(CONDITION_NAMES)


def test_raw_result_reads_legacy_two_condition_file():
    """Pre-2026-08-04 runs stored filtered/unfiltered flat. Those meant
    vector_query+predicate and raw_query_text alone, so they map onto exactly two
    of the four cells — old runs stay analysable rather than becoming unreadable."""
    legacy = {"query_id": "q1", "filtered_ids": ["a"], "unfiltered_ids": ["b"],
              "latency_filtered_ms": 1.0, "latency_unfiltered_ms": 2.0}
    r = RawResult.from_dict(legacy)
    assert r.conditions() == ["raw+nofilter", "semantic+filter"]
    assert r.ids["semantic+filter"] == ["a"]
    assert r.ids["raw+nofilter"] == ["b"]
    assert r.latency_ms["raw+nofilter"] == 2.0


def test_raw_results_jsonl_roundtrip(tmp_path):
    rr = RawResults(
        system="pgvector", k=1000,
        results=[
            _rr("q1", **{c: ["a", "b"] for c in CONDITION_NAMES}),
            _rr("q2", **{"raw+nofilter": ["x"]}),   # item with no predicate
        ],
    )
    p = tmp_path / "raw.jsonl"
    rr.write_jsonl(str(p))
    back = RawResults.read_jsonl(str(p))
    assert back.system == "pgvector"
    assert back.k == 1000
    assert back.results == rr.results
    assert [r.query_id for r in back] == ["q1", "q2"]  # __iter__


# ─── QueryMetrics ───────────────────────────────────────────────────────────

def _qm(qid, *, scorable=None, recall=None, rank=None, lat=None):
    """QueryMetrics from per-condition dicts; conditions default to whatever the
    rank dict names."""
    rank = rank or {}
    scorable = {c: True for c in rank} if scorable is None else scorable
    return QueryMetrics(qid, scorable, recall or {c: {} for c in rank}, rank,
                        lat or {c: 1.0 for c in rank})


def test_query_metrics_reciprocal_rank():
    m = _qm("q1", rank={"semantic+filter": 4, "raw+nofilter": None})
    assert m.rr("semantic+filter") == 0.25
    assert m.rr("raw+nofilter") == 0.0        # None rank => RR 0
    assert m.rr("semantic+nofilter") == 0.0   # condition absent => RR 0


def test_query_metrics_rr_at_cap():
    m = _qm("q1", rank={"semantic+filter": 40})
    assert m.rr("semantic+filter", 50) == 0.025
    assert m.rr("semantic+filter", 10) == 0.0
    assert m.rr("semantic+filter") == 0.025    # uncapped


def test_query_metrics_to_dict_stringifies_k():
    m = _qm("q1", recall={"semantic+filter": {5: 1.0, 25: 0.4}},
            rank={"semantic+filter": 1})
    d = m.to_dict()
    assert d["recall"]["semantic+filter"] == {"5": 1.0, "25": 0.4}
    assert d["rr"]["semantic+filter"] == 1.0
    assert d["target_rank"]["semantic+filter"] == 1


# ─── Metrics aggregation ────────────────────────────────────────────────────

FILT, NOFILT = "semantic+filter", "raw+nofilter"


def test_metrics_conditions_in_canonical_order():
    m = Metrics(system="s", ks=(5,), per_query=[_qm("q1", rank={FILT: 1, NOFILT: 2})])
    assert m.conditions() == [NOFILT, FILT]


def test_metrics_aggregates_over_the_common_subset():
    """q2 is scorable in the no-filter cell only. The default aggregate must not
    average one condition over 2 items and the other over 1 — that difference
    would read as an effect of the condition."""
    m = Metrics(system="s", ks=(5,), per_query=[
        _qm("q1", recall={FILT: {5: 1.0}, NOFILT: {5: 0.5}}, rank={FILT: 1, NOFILT: 1}),
        _qm("q2", scorable={FILT: False, NOFILT: True},
            recall={FILT: {5: 0.0}, NOFILT: {5: 1.0}}, rank={FILT: None, NOFILT: 2}),
    ])
    assert [q.query_id for q in m.comparable()] == ["q1"]
    assert m.mean_recall(NOFILT, 5) == 0.5              # common subset: q1 only
    assert m.mean_recall(NOFILT, 5, common=False) == 0.75   # own subset: q1 + q2
    assert m.mrr(NOFILT) == 1.0
    assert m.mrr(NOFILT, common=False) == 0.75          # (1/1 + 1/2) / 2


def test_metrics_latency_over_every_item_that_ran_the_condition():
    # latency is a system property, independent of scorability
    m = Metrics(system="s", ks=(5,), per_query=[
        _qm("q1", rank={FILT: 1, NOFILT: 1}, lat={FILT: 10.0, NOFILT: 100.0}),
        _qm("q2", scorable={FILT: False, NOFILT: True}, rank={FILT: None, NOFILT: 1},
            lat={FILT: 30.0, NOFILT: 300.0}),
        _qm("q3", rank={NOFILT: 1}, lat={NOFILT: 200.0}),   # no predicate: no filter cell
    ])
    assert m.median_latency(FILT) == 20.0        # median(10, 30)
    assert m.latency_n(FILT) == 2
    assert m.median_latency(NOFILT) == 200.0
    assert m.latency_n(NOFILT) == 3


def test_metrics_empty_is_safe():
    m = Metrics(system="s", ks=(5,), per_query=[])
    assert m.conditions() == []
    assert m.mean_recall(FILT, 5) == 0.0
    assert m.mrr(FILT) == 0.0
    assert m.median_latency(FILT) == 0.0


def test_metrics_write_jsonl(tmp_path):
    import json

    m = Metrics(system="s", ks=(5, 25), retrieval_k=1000, per_query=[
        _qm("q1", recall={FILT: {5: 1.0, 25: 0.5}}, rank={FILT: 2})])
    p = tmp_path / "metrics.jsonl"
    m.write_jsonl(str(p))
    lines = p.read_text().splitlines()
    header = json.loads(lines[0])
    assert header == {"system": "s", "ks": [5, 25], "retrieval_k": 1000,
                      "conditions": [FILT], "n_comparable": 1,
                      "harness_contract": HARNESS_CONTRACT}
    row = json.loads(lines[1])
    assert row["query_id"] == "q1"
    assert row["rr"][FILT] == 0.5


# ─── capped MRR (Omar, 28-07-2026) ──────────────────────────────────────────

def test_capped_mrr_counts_too_deep_as_a_miss():
    m = Metrics(system="s", ks=(5,), per_query=[
        _qm("q1", rank={FILT: 1}), _qm("q2", rank={FILT: 100})])
    assert m.mrr(FILT) == (1.0 + 0.01) / 2          # uncapped
    assert m.mrr(FILT, 1000) == (1.0 + 0.01) / 2
    assert m.mrr(FILT, 100) == (1.0 + 0.01) / 2     # rank 100 is exactly at cap
    assert m.mrr(FILT, 50) == 0.5                   # rank 100 -> miss, not 0.01
    assert m.mrr(FILT, 10) == 0.5


def test_capped_mrr_never_found_stays_a_miss():
    m = Metrics(system="s", ks=(5,), per_query=[_qm("q1", rank={FILT: None})])
    assert m.mrr(FILT, 10) == 0.0
    assert m.mrr(FILT) == 0.0


def test_capped_mrr_is_per_condition():
    m = Metrics(system="s", ks=(5,), per_query=[_qm("q1", rank={FILT: 200, NOFILT: 2})])
    assert m.mrr(FILT, 50) == 0.0
    assert m.mrr(NOFILT, 50) == 0.5


def test_cap_is_meaningful_against_retrieval_depth():
    # nothing was retrieved past k, so a cap at/above k is the uncapped MRR
    m = Metrics(system="s", ks=(5,), per_query=[], retrieval_k=1000)
    assert m.cap_is_meaningful(100) is True
    assert m.cap_is_meaningful(1000) is False
    assert m.cap_is_meaningful(2000) is False
    # unknown depth (legacy metrics file) => cannot claim a cap is meaningful
    assert Metrics(system="s", ks=(5,), per_query=[]).cap_is_meaningful(10) is False


# ─── latency percentiles ────────────────────────────────────────────────────

def test_latency_percentile_across_queries():
    m = Metrics(system="s", ks=(5,), per_query=[
        _qm(f"q{i}", rank={FILT: 1, NOFILT: 1},
            lat={FILT: float(i), NOFILT: float(i) * 10})
        for i in range(1, 101)])          # 1..100 ms filtered, 10..1000 nofilter
    assert m.latency_percentile(FILT, 50) == m.median_latency(FILT)
    assert m.latency_percentile(FILT, 95) == 95.05
    assert m.latency_percentile(NOFILT, 95) == 950.5
    assert m.latency_percentile(FILT, 100) == 100.0
    assert m.latency_percentile(FILT, 0) == 1.0


def test_latency_percentile_degenerate_inputs():
    assert Metrics(system="s", ks=(5,), per_query=[]).latency_percentile(FILT, 95) == 0.0
    one = Metrics(system="s", ks=(5,),
                  per_query=[_qm("q1", rank={FILT: 1}, lat={FILT: 7.0})])
    assert one.latency_percentile(FILT, 95) == 7.0


# ─── helpers ────────────────────────────────────────────────────────────────

def test_safe_mean():
    assert _safe_mean([1.0, 2.0, 3.0]) == 2.0
    assert _safe_mean([]) == 0.0


def test_median_odd_even_empty():
    assert _median([3.0, 1.0, 2.0]) == 2.0        # odd
    assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5   # even => mean of middle two
    assert _median([]) == 0.0


# ─── load_query_set (file round-trip) ───────────────────────────────────────

def test_load_query_set(tmp_path, enriched_item):
    import json

    p = tmp_path / "benchmark.jsonl"
    p.write_text(json.dumps(enriched_item) + "\n\n")  # trailing blank line ignored
    items = load_query_set(str(p))
    assert len(items) == 1
    assert items[0].query_id == "q0001"
    assert items[0].ground_truth.is_scorable is True

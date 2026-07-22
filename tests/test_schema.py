"""Unit tests for frame.core.schema — the on-disk contract + metric aggregation."""

from __future__ import annotations

from frame.core.schema import (
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
    assert gt.gt_vec_nofilter == ["kf_target", "kf_p", "kf_q", "kf_r"]


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

def test_raw_result_roundtrip():
    r = RawResult("q1", ["a", "b"], ["c", "d"], 1.5, 2.5)
    assert RawResult.from_dict(r.to_dict()) == r


def test_raw_results_jsonl_roundtrip(tmp_path):
    rr = RawResults(
        system="pgvector",
        k=1000,
        results=[
            RawResult("q1", ["a", "b"], ["c"], 1.0, 2.0),
            RawResult("q2", ["x"], ["y", "z"], 3.0, 4.0),
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

def test_query_metrics_reciprocal_rank():
    m = QueryMetrics("q1", True, {5: 1.0}, {5: 0.5}, target_rank_filtered=4,
                     target_rank_unfiltered=None, latency_filtered_ms=1.0,
                     latency_unfiltered_ms=2.0)
    assert m.rr_filtered == 0.25
    assert m.rr_unfiltered == 0.0  # None rank => RR 0


def test_query_metrics_to_dict_stringifies_k():
    m = QueryMetrics("q1", True, {5: 1.0, 25: 0.4}, {5: 0.0}, 1, 2, 1.0, 2.0)
    d = m.to_dict()
    assert d["recall_filtered"] == {"5": 1.0, "25": 0.4}
    assert d["rr_filtered"] == 1.0


# ─── Metrics aggregation ────────────────────────────────────────────────────

def _qm(qid, scorable, rf, rnf, rank_f, lat_f, lat_nf):
    return QueryMetrics(qid, scorable, rf, rnf, rank_f, None, lat_f, lat_nf)


def test_metrics_means_only_over_scorable():
    m = Metrics(
        system="s",
        ks=(5,),
        per_query=[
            _qm("q1", True, {5: 1.0}, {5: 0.5}, 1, 10.0, 20.0),
            _qm("q2", True, {5: 0.0}, {5: 0.5}, None, 30.0, 40.0),
            _qm("q3", False, {5: 1.0}, {5: 1.0}, 1, 50.0, 60.0),  # excluded
        ],
    )
    assert m.mean_recall_filtered(5) == 0.5      # (1.0 + 0.0) / 2
    assert m.mean_recall_unfiltered(5) == 0.5    # (0.5 + 0.5) / 2
    assert m.mrr_filtered() == 0.5               # (1/1 + 0) / 2
    assert m.mrr_unfiltered() == 0.0             # both ranks None


def test_metrics_latency_over_all_items():
    # latency summarised over ALL items, scorable or not
    m = Metrics(
        system="s",
        ks=(5,),
        per_query=[
            _qm("q1", True, {5: 1.0}, {5: 1.0}, 1, 10.0, 100.0),
            _qm("q2", False, {5: 0.0}, {5: 0.0}, None, 30.0, 300.0),
        ],
    )
    assert m.median_latency_filtered() == 20.0    # median(10, 30)
    assert m.median_latency_unfiltered() == 200.0


def test_metrics_empty_is_safe():
    m = Metrics(system="s", ks=(5,), per_query=[])
    assert m.mean_recall_filtered(5) == 0.0
    assert m.mrr_filtered() == 0.0
    assert m.median_latency_filtered() == 0.0


def test_metrics_write_jsonl(tmp_path):
    import json

    m = Metrics(system="s", ks=(5, 25),
                per_query=[_qm("q1", True, {5: 1.0, 25: 0.5}, {5: 0.0, 25: 0.0}, 2, 1.0, 2.0)])
    p = tmp_path / "metrics.jsonl"
    m.write_jsonl(str(p))
    lines = p.read_text().splitlines()
    header = json.loads(lines[0])
    assert header == {"system": "s", "ks": [5, 25]}
    row = json.loads(lines[1])
    assert row["query_id"] == "q1"
    assert row["rr_filtered"] == 0.5


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

"""Unit tests for frame.core.analyzer — the shared scoring logic."""

from __future__ import annotations

from frame.core.analyzer import Analyzer, _first_rank, _recall_at_k
from frame.core.schema import QueryItem, RawResult, RawResults


# ─── _recall_at_k ───────────────────────────────────────────────────────────

def test_recall_perfect():
    assert _recall_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0


def test_recall_partial():
    # oracle top-3 = {a,b,c}; system top-3 = {a,x,b} => 2 hits / 3
    assert _recall_at_k(["a", "x", "b"], ["a", "b", "c"], 3) == 2 / 3


def test_recall_respects_k_on_both_sides():
    # k=2 => truth is {a,b}; system top-2 = {a,x} => 1/2
    assert _recall_at_k(["a", "x", "b"], ["a", "b", "c"], 2) == 0.5


def test_recall_empty_gt_is_zero():
    assert _recall_at_k(["a"], None, 5) == 0.0
    assert _recall_at_k(["a"], [], 5) == 0.0


def test_recall_no_overlap():
    assert _recall_at_k(["x", "y"], ["a", "b"], 5) == 0.0


# ─── _first_rank ────────────────────────────────────────────────────────────

def test_first_rank_is_one_based():
    assert _first_rank(["a", "b", "target"], {"target"}) == 3
    assert _first_rank(["target", "a"], {"target"}) == 1


def test_first_rank_none_when_absent():
    assert _first_rank(["a", "b"], {"target"}) is None


def test_first_rank_multiple_targets_returns_first():
    assert _first_rank(["a", "t2", "t1"], {"t1", "t2"}) == 2


# ─── Analyzer._score_one / analyze ──────────────────────────────────────────

def _raw(qid, filtered, unfiltered):
    return RawResult(qid, filtered, unfiltered, latency_filtered_ms=5.0,
                     latency_unfiltered_ms=7.0)


def test_analyze_scorable_item(enriched_item):
    item = QueryItem.from_dict(enriched_item)
    # system nails the filtered target at rank 1, misses on no-filter
    raw = RawResults(
        system="fake", k=5,
        results=[_raw("q0001", ["kf_target", "kf_a"], ["kf_zzz", "kf_target"])],
    )
    m = Analyzer(ks=(5,)).analyze(raw, [item])
    qm = m.per_query[0]
    assert qm.scorable is True
    assert qm.recall_filtered[5] > 0.0
    assert qm.target_rank_filtered == 1
    assert qm.target_rank_unfiltered == 2
    assert qm.latency_filtered_ms == 5.0


def test_analyze_unscorable_item_zeroed(enriched_item):
    enriched_item["computed"]["target_passes_filter"] = False
    item = QueryItem.from_dict(enriched_item)
    raw = RawResults(system="fake", k=5,
                     results=[_raw("q0001", ["kf_target"], ["kf_target"])])
    qm = Analyzer(ks=(5,)).analyze(raw, [item]).per_query[0]
    assert qm.scorable is False
    assert qm.recall_filtered[5] == 0.0
    assert qm.target_rank_filtered is None


def test_analyze_missing_gt_is_unscorable():
    # a raw result with no matching item => gt is None => unscorable, not a crash
    raw = RawResults(system="fake", k=5,
                     results=[_raw("ghost", ["a"], ["b"])])
    qm = Analyzer(ks=(5,)).analyze(raw, []).per_query[0]
    assert qm.scorable is False


def test_analyze_recall_at_multiple_ks(enriched_item):
    item = QueryItem.from_dict(enriched_item)
    # filtered GT = [kf_target, kf_a, kf_b, kf_c]; system returns 2 of the top-4
    raw = RawResults(
        system="fake", k=100,
        results=[_raw("q0001", ["kf_target", "zzz", "kf_b", "yyy"],
                      ["kf_x", "kf_y", "kf_z"])],
    )
    m = Analyzer(ks=(2, 5)).analyze(raw, [item]).per_query[0]
    assert m.recall_filtered[2] == 0.5   # {kf_target,kf_a} truth, hit kf_target
    assert m.recall_filtered[5] == 0.5   # {t,a,b,c} truth, hit t & b => 2/4


# ─── summary rendering ──────────────────────────────────────────────────────

def test_summary_contains_headline_numbers(enriched_item):
    item = QueryItem.from_dict(enriched_item)
    raw = RawResults(system="pgvector", k=5,
                     results=[_raw("q0001", ["kf_target"], ["kf_target"])])
    m = Analyzer(ks=(5,)).analyze(raw, [item])
    text = Analyzer(ks=(5,)).summary(m)
    assert "system: pgvector" in text
    assert "scorable: 1" in text
    assert "MRR" in text

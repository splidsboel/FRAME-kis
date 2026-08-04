"""Unit tests for frame.core.analyzer — the shared scoring logic."""

from __future__ import annotations

import copy

from frame.core.analyzer import Analyzer, _first_rank, _recall_at_k
from frame.core.schema import CONDITION_NAMES, QueryItem, RawResult, RawResults

FILT, NOFILT = "semantic+filter", "raw+nofilter"
RAW_FILT, SEM_NOFILT = "raw+filter", "semantic+nofilter"


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

def _raw(qid, **by_cond):
    """RawResult from condition -> ranked ids, with a dummy latency per cell."""
    return RawResult(query_id=qid, ids=dict(by_cond),
                     latency_ms={c: 5.0 for c in by_cond})


def _full(qid, ids):
    """Same ranked list in all four cells — for tests about plumbing, not ranking."""
    return _raw(qid, **{c: list(ids) for c in CONDITION_NAMES})


def test_analyze_scores_every_condition(enriched_item):
    item = QueryItem.from_dict(enriched_item)
    raw = RawResults(system="fake", k=5, results=[_full("q0001", ["kf_target", "kf_a"])])
    qm = Analyzer(ks=(5,)).analyze(raw, [item]).per_query[0]
    assert qm.conditions() == list(CONDITION_NAMES)
    assert all(qm.is_scorable(c) for c in CONDITION_NAMES)
    assert all(qm.target_rank[c] == 1 for c in CONDITION_NAMES)


def test_each_condition_scored_against_its_own_gt(enriched_item):
    # gt differs per cell; a system returning kf_m hits ONLY raw+filter's answer
    item = QueryItem.from_dict(enriched_item)
    raw = RawResults(system="fake", k=5, results=[_full("q0001", ["kf_m"])])
    qm = Analyzer(ks=(1,)).analyze(raw, [item]).per_query[0]
    assert qm.recall[RAW_FILT][1] == 0.0     # gt_raw_filtered[:1] is kf_target
    # kf_m is in gt_raw_filtered but not in the other three answers
    assert _recall_at_k(["kf_m"], item.ground_truth.gt_raw_filtered, 3) > 0.0
    assert _recall_at_k(["kf_m"], item.ground_truth.gt_filtered, 3) == 0.0


def test_filter_cells_unscorable_when_target_fails_filter(enriched_item):
    enriched_item["computed"]["target_passes_filter"] = False
    item = QueryItem.from_dict(enriched_item)
    raw = RawResults(system="fake", k=5, results=[_full("q0001", ["kf_target"])])
    qm = Analyzer(ks=(5,)).analyze(raw, [item]).per_query[0]
    # the two FILTER cells are compromised...
    assert qm.is_scorable(FILT) is False
    assert qm.is_scorable(RAW_FILT) is False
    assert qm.target_rank[FILT] is None
    # ...but the no-filter cells applied no predicate and stay valid
    assert qm.is_scorable(NOFILT) is True
    assert qm.is_scorable(SEM_NOFILT) is True
    assert qm.target_rank[NOFILT] == 1


def test_missing_gt_is_unscorable_everywhere():
    raw = RawResults(system="fake", k=5, results=[_full("ghost", ["a"])])
    qm = Analyzer(ks=(5,)).analyze(raw, []).per_query[0]
    assert not any(qm.is_scorable(c) for c in CONDITION_NAMES)


def test_analyze_only_scores_conditions_the_run_produced(enriched_item):
    item = QueryItem.from_dict(enriched_item)
    raw = RawResults(system="fake", k=5,
                     results=[_raw("q0001", **{NOFILT: ["kf_x"]})])
    qm = Analyzer(ks=(5,)).analyze(raw, [item]).per_query[0]
    assert qm.conditions() == [NOFILT]
    assert FILT not in qm.scorable


def test_analyze_recall_at_multiple_ks(enriched_item):
    item = QueryItem.from_dict(enriched_item)
    # filtered GT = [kf_target, kf_a, kf_b, kf_c]; system returns 2 of the top-4
    raw = RawResults(system="fake", k=100, results=[
        _raw("q0001", **{FILT: ["kf_target", "zzz", "kf_b", "yyy"]})])
    qm = Analyzer(ks=(2, 5)).analyze(raw, [item]).per_query[0]
    assert qm.recall[FILT][2] == 0.5   # {kf_target,kf_a} truth, hit kf_target
    assert qm.recall[FILT][5] == 0.5   # {t,a,b,c} truth, hit t & b => 2/4


# ─── the common comparable subset ───────────────────────────────────────────

def test_aggregates_use_the_common_subset(enriched_item):
    """An item whose target fails its filter is valid in the no-filter cells and
    invalid in the filter cells. Averaging each condition over its own scorable set
    would compare a 2-item mean against a 1-item mean."""
    ok = QueryItem.from_dict(enriched_item)
    bad_src = copy.deepcopy(enriched_item)
    bad_src["query_id"] = "q0002"
    bad_src["computed"]["target_passes_filter"] = False
    bad = QueryItem.from_dict(bad_src)

    raw = RawResults(system="fake", k=5, results=[
        _full("q0001", ["kf_target"]),          # rank 1 everywhere
        _full("q0002", ["zzz", "kf_target"]),   # rank 2 everywhere
    ])
    m = Analyzer(ks=(5,)).analyze(raw, [ok, bad])

    assert len(m.per_query) == 2
    assert [q.query_id for q in m.comparable()] == ["q0001"]
    # common subset: only q0001 counts, in every condition
    assert m.mrr(NOFILT) == 1.0
    # per-condition own subset: the no-filter cell also gets q0002's rank 2
    assert m.mrr(NOFILT, common=False) == (1.0 + 0.5) / 2


def test_conditions_reported_in_canonical_order(enriched_item):
    item = QueryItem.from_dict(enriched_item)
    raw = RawResults(system="fake", k=5, results=[_full("q0001", ["kf_target"])])
    assert Analyzer(ks=(5,)).analyze(raw, [item]).conditions() == list(CONDITION_NAMES)


# ─── capped MRR + retrieval depth ───────────────────────────────────────────

def test_analyze_carries_retrieval_depth(enriched_item):
    item = QueryItem.from_dict(enriched_item)
    raw = RawResults(system="fake", k=250, results=[_full("q0001", ["kf_target"])])
    assert Analyzer(ks=(5,)).analyze(raw, [item]).retrieval_k == 250


def test_capped_mrr_counts_too_deep_as_a_miss(enriched_item):
    item = QueryItem.from_dict(enriched_item)
    ids = [f"kf_{i}" for i in range(59)] + ["kf_target"]     # target at rank 60
    raw = RawResults(system="fake", k=1000, results=[_full("q0001", ids)])
    m = Analyzer(ks=(5,)).analyze(raw, [item])
    assert m.mrr(FILT, 100) == 1 / 60
    assert m.mrr(FILT, 50) == 0.0


def test_headline_cap_is_the_deepest_that_bites(enriched_item):
    item = QueryItem.from_dict(enriched_item)
    raw = RawResults(system="fake", k=100, results=[_full("q0001", ["kf_target"])])
    a = Analyzer(ks=(5,))
    assert a.headline_cap(a.analyze(raw, [item])) == 50   # @1000 and @100 are inert


# ─── summary rendering ──────────────────────────────────────────────────────

def test_summary_contains_headline_numbers(enriched_item):
    item = QueryItem.from_dict(enriched_item)
    raw = RawResults(system="pgvector", k=5, results=[_full("q0001", ["kf_target"])])
    text = Analyzer(ks=(5,)).summary(Analyzer(ks=(5,)).analyze(raw, [item]))
    assert "system: pgvector" in text
    assert "comparable" in text
    assert "median" in text and "p95" in text
    for c in CONDITION_NAMES:
        assert c in text


def test_summary_draws_the_grid_only_when_all_four_cells_ran(enriched_item):
    item = QueryItem.from_dict(enriched_item)
    full = RawResults(system="pgvector", k=1000, results=[_full("q0001", ["kf_target"])])
    assert "The 2x2" in Analyzer(ks=(5,)).summary(Analyzer(ks=(5,)).analyze(full, [item]))

    partial = RawResults(system="pgvector", k=1000,
                         results=[_raw("q0001", **{FILT: ["kf_target"],
                                                   NOFILT: ["kf_target"]})])
    text = Analyzer(ks=(5,)).summary(Analyzer(ks=(5,)).analyze(partial, [item]))
    assert "grid omitted" in text


def test_summary_flags_caps_beyond_retrieval_depth(enriched_item):
    item = QueryItem.from_dict(enriched_item)
    raw = RawResults(system="pgvector", k=100, results=[_full("q0001", ["kf_target"])])
    text = Analyzer(ks=(5,)).summary(Analyzer(ks=(5,)).analyze(raw, [item]))
    assert "at or beyond the run's retrieval depth k=100" in text

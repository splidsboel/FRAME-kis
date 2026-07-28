"""Unit tests for frame.core.profile — the selectivity / plan profiler."""

from __future__ import annotations

from frame.core.adapter import VectorDBAdapter
from frame.core.profile import Profiler, SelectivityProfile, _summarize_filters
from frame.core.schema import Predicate, QueryItem


class FakeProfilingAdapter(VectorDBAdapter):
    """Adapter exposing the profiling diagnostics the Profiler requires, with
    canned numbers so the arithmetic is checkable."""

    name = "fake-prof"

    def __init__(self, corpus=1000, passing=100, plan=("hnsw", 120), members=2):
        self._corpus = corpus
        self._passing = passing
        self._plan = plan
        self._members = members
        self.entered = False

    def setup(self) -> None:
        self.entered = True

    def search(self, query_vector, filters, k):
        return []

    # profiling diagnostics --------------------------------------------------
    def corpus_size(self) -> int:
        return self._corpus

    def count_passing(self, filters) -> int:
        return self._passing

    def plan_choice(self, vec, filters, k):
        return self._plan

    def count_members(self, ids, filters) -> int:
        return self._members


def _filtered_item(enriched_item):
    return QueryItem.from_dict(enriched_item)


# ─── _summarize_filters ─────────────────────────────────────────────────────

def test_summarize_single_list_filter():
    f = Predicate("object", "object_label", "in", ["fish", "hat"])
    assert _summarize_filters([f]) == "object:{fish,hat}"


def test_summarize_string_value_filter():
    f = Predicate("pattern-match", "ocr_text", "contains", "EXIT")
    assert _summarize_filters([f]) == "pattern-match:{EXIT}"


def test_summarize_conjunction():
    fs = [
        Predicate("scene", "scene_label", "in", ["night"]),
        Predicate("object", "object_label", "in", ["car"]),
    ]
    assert _summarize_filters(fs) == "scene:{night} AND object:{car}"


# ─── SelectivityProfile.divergence ──────────────────────────────────────────

def test_divergence_positive_when_global_overstates():
    p = SelectivityProfile(
        query_id="q", filter_summary="s", n_filters=1, corpus_size=1000,
        per_filter=[{"summary": "s", "count": 200, "selectivity": 0.2}],
        global_count=200, global_selectivity=0.2, plan="hnsw", planner_est_rows=200,
        near_query_n=100, near_query_passes=5, near_query_pass_rate=0.05,
    )
    assert abs(p.divergence - 0.15) < 1e-9


def test_divergence_none_without_near_query_rate():
    p = SelectivityProfile(
        query_id="q", filter_summary="s", n_filters=1, corpus_size=1000,
        per_filter=[{"summary": "s", "count": 200, "selectivity": 0.2}],
        global_count=200, global_selectivity=0.2, plan="hnsw", planner_est_rows=200,
        near_query_n=0, near_query_passes=0, near_query_pass_rate=None,
    )
    assert p.divergence is None
    assert p.to_dict()["divergence"] is None


# ─── Profiler.profile / _profile_one ────────────────────────────────────────

def test_profile_computes_selectivity_and_pass_rate(enriched_item, fake_encoder):
    item = _filtered_item(enriched_item)
    adapter = FakeProfilingAdapter(corpus=1000, passing=100, plan=("hnsw", 120), members=1)
    profiles = Profiler(adapter, fake_encoder, near_query_n=100).profile([item])

    assert len(profiles) == 1
    p = profiles[0]
    assert p.corpus_size == 1000
    assert p.global_count == 100
    assert p.global_selectivity == 0.1
    assert p.plan == "hnsw"
    assert p.planner_est_rows == 120
    # per-part breakdown: one filter, its own selectivity, conjunction tightens 1x
    assert len(p.per_filter) == 1
    assert p.per_filter[0]["selectivity"] == 0.1
    assert p.tightening == 1.0
    # enriched_item has 4 vec-nofilter neighbours; adapter says 1 passes
    assert p.near_query_n == 4
    assert p.near_query_passes == 1
    assert p.near_query_pass_rate == 0.25
    assert adapter.entered is True  # profiled inside the adapter context


def test_profile_skips_no_filter_items(enriched_item, fake_encoder):
    no_filter = QueryItem.from_dict({**enriched_item,
                                     "decomposition": {"vector_query": "v", "filters": []}})
    adapter = FakeProfilingAdapter()
    profiles = Profiler(adapter, fake_encoder).profile([no_filter])
    assert profiles == []


def test_profile_pass_rate_none_without_vec_nofilter_gt(enriched_item, fake_encoder):
    enriched_item["computed"]["geometric_gt_vec_nofilter"] = None
    item = _filtered_item(enriched_item)
    adapter = FakeProfilingAdapter()
    p = Profiler(adapter, fake_encoder).profile([item])[0]
    assert p.near_query_n == 0
    assert p.near_query_pass_rate is None


def test_profiler_summary_reports_risk(enriched_item, fake_encoder):
    item = _filtered_item(enriched_item)
    # global_selectivity 0.5 >> near-query pass-rate 0.25 on an HNSW plan => risk
    adapter = FakeProfilingAdapter(corpus=1000, passing=500, plan=("hnsw", 500), members=1)
    profiles = Profiler(adapter, fake_encoder).profile([item])
    text = Profiler(adapter, fake_encoder).summary(profiles)
    assert "filtered items: 1" in text
    assert "silent-recall-loss risk" in text


def test_profiler_summary_empty():
    adapter = FakeProfilingAdapter()
    assert "no filtered items" in Profiler(adapter, None).summary([])


def test_profile_write_jsonl(tmp_path, enriched_item, fake_encoder):
    import json

    from frame.core.profile import write_jsonl

    item = _filtered_item(enriched_item)
    adapter = FakeProfilingAdapter()
    profiles = Profiler(adapter, fake_encoder).profile([item])
    p = tmp_path / "prof.jsonl"
    write_jsonl(profiles, str(p), system="pgvector")
    lines = p.read_text().splitlines()
    assert json.loads(lines[0]) == {"system": "pgvector", "kind": "selectivity_profile"}
    assert json.loads(lines[1])["query_id"] == "q0001"

"""Unit tests for frame.core.runner — orchestration with fake collaborators."""

from __future__ import annotations

from frame.core.runner import Runner
from frame.core.schema import QueryItem


def test_run_produces_one_result_per_item(enriched_item, fake_adapter, fake_encoder):
    item = QueryItem.from_dict(enriched_item)
    runner = Runner(fake_adapter, fake_encoder, warmup=0, repeat=1)
    raw = runner.run([item], k=10)

    assert raw.system == "fake"
    assert raw.k == 10
    assert len(raw.results) == 1
    r = raw.results[0]
    assert r.query_id == "q0001"
    assert r.filtered_ids == ["kf_target", "kf_a", "kf_z"]
    assert r.unfiltered_ids == ["kf_x", "kf_target", "kf_y"]


def test_run_encodes_both_texts(enriched_item, fake_adapter, fake_encoder):
    item = QueryItem.from_dict(enriched_item)
    Runner(fake_adapter, fake_encoder, warmup=0, repeat=1).run([item], k=5)
    # filtered condition encodes vector_query, no-filter encodes raw_query_text
    assert item.vector_query in fake_encoder.calls
    assert item.raw_query_text in fake_encoder.calls


def test_run_filtered_gets_predicates_nofilter_gets_none(
    enriched_item, fake_adapter, fake_encoder
):
    item = QueryItem.from_dict(enriched_item)
    Runner(fake_adapter, fake_encoder, warmup=0, repeat=1).run([item], k=5)
    # two searches: one with the item's filters, one with zero filters
    n_filters_seen = sorted(nf for nf, _ in fake_adapter.search_calls)
    assert n_filters_seen == [0, 1]
    # both priced at the same k
    assert {k for _, k in fake_adapter.search_calls} == {5}


def test_warmup_and_repeat_control_search_count(enriched_item, fake_adapter, fake_encoder):
    item = QueryItem.from_dict(enriched_item)
    Runner(fake_adapter, fake_encoder, warmup=2, repeat=3).run([item], k=5)
    # per condition: warmup(2) + repeat(3) = 5 searches; two conditions => 10
    assert len(fake_adapter.search_calls) == 10


def test_latency_is_non_negative(enriched_item, fake_adapter, fake_encoder):
    item = QueryItem.from_dict(enriched_item)
    raw = Runner(fake_adapter, fake_encoder, warmup=1, repeat=3).run([item], k=5)
    r = raw.results[0]
    assert r.latency_filtered_ms >= 0.0
    assert r.latency_unfiltered_ms >= 0.0


def test_runner_clamps_degenerate_warmup_repeat(fake_adapter, fake_encoder):
    r = Runner(fake_adapter, fake_encoder, warmup=-5, repeat=0)
    assert r.warmup == 0    # clamped to >= 0
    assert r.repeat == 1    # clamped to >= 1

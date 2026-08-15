"""Unit tests for frame.core.runner — orchestration with fake collaborators."""

from __future__ import annotations

import copy

from frame.core.runner import Runner
from frame.core.schema import CONDITION_NAMES, QueryItem


def test_run_produces_one_result_per_item(enriched_item, fake_adapter, fake_encoder):
    item = QueryItem.from_dict(enriched_item)
    runner = Runner(fake_adapter, fake_encoder, warmup=0, repeat=1)
    raw = runner.run([item], k=10)

    assert raw.system == "fake"
    assert raw.k == 10
    assert len(raw.results) == 1
    r = raw.results[0]
    assert r.query_id == "q0001"
    assert r.conditions() == list(CONDITION_NAMES)          # full 2x2
    assert r.ids["semantic+filter"] == ["kf_target", "kf_a", "kf_z"]
    assert r.ids["raw+nofilter"] == ["kf_x", "kf_target", "kf_y"]


def test_run_covers_the_full_2x2(enriched_item, fake_adapter, fake_encoder):
    item = QueryItem.from_dict(enriched_item)
    raw = Runner(fake_adapter, fake_encoder, warmup=0, repeat=1).run([item], k=5)
    r = raw.results[0]
    assert set(r.ids) == set(CONDITION_NAMES)
    assert set(r.latency_ms) == set(CONDITION_NAMES)
    # two cells apply the predicate, two do not
    n_filters_seen = sorted(nf for nf, _ in fake_adapter.search_calls)
    assert n_filters_seen == [0, 0, 1, 1]
    assert {k for _, k in fake_adapter.search_calls} == {5}


def test_run_encodes_each_text_once_per_item(enriched_item, fake_adapter, fake_encoder):
    item = QueryItem.from_dict(enriched_item)
    Runner(fake_adapter, fake_encoder, warmup=0, repeat=1).run([item], k=5)
    # 4 cells but only 2 distinct texts — the filter axis must be measured against
    # an identical query vector, so each text is encoded once and shared
    assert fake_encoder.calls.count(item.vector_query) == 1
    assert fake_encoder.calls.count(item.raw_query_text) == 1


def test_item_without_predicate_skips_the_filter_cells(
    enriched_item, fake_adapter, fake_encoder
):
    # an empty predicate makes the filter cells identical to the no-filter cells;
    # running them would manufacture two duplicate numbers
    unfiltered = copy.deepcopy(enriched_item)
    unfiltered["decomposition"]["filters"] = []
    item = QueryItem.from_dict(unfiltered)
    raw = Runner(fake_adapter, fake_encoder, warmup=0, repeat=1).run([item], k=5)
    r = raw.results[0]
    assert r.conditions() == ["raw+nofilter", "semantic+nofilter"]
    assert len(fake_adapter.search_calls) == 2


def test_warmup_and_repeat_control_search_count(enriched_item, fake_adapter, fake_encoder):
    item = QueryItem.from_dict(enriched_item)
    Runner(fake_adapter, fake_encoder, warmup=2, repeat=3).run([item], k=5)
    # per cell: warmup(2) + repeat(3) = 5 searches; four cells => 20
    assert len(fake_adapter.search_calls) == 20


def test_latency_recorded_per_condition(enriched_item, fake_adapter, fake_encoder):
    item = QueryItem.from_dict(enriched_item)
    raw = Runner(fake_adapter, fake_encoder, warmup=1, repeat=3).run([item], k=5)
    r = raw.results[0]
    assert set(r.latency_ms) == set(CONDITION_NAMES)
    assert all(v >= 0.0 for v in r.latency_ms.values())


def test_runner_clamps_degenerate_warmup_repeat(fake_adapter, fake_encoder):
    r = Runner(fake_adapter, fake_encoder, warmup=-5, repeat=0)
    assert r.warmup == 0    # clamped to >= 0
    assert r.repeat == 1    # clamped to >= 1


def test_progress_path_streams_a_readable_file(
    enriched_item, fake_adapter, fake_encoder, tmp_path
):
    from frame.core.schema import RawResults
    from frame.core.version import BenchmarkVersion

    item = QueryItem.from_dict(enriched_item)
    path = tmp_path / "raw_results.fake.jsonl"
    bench = BenchmarkVersion(version="1.0.0", corpus="v3c1", digest="deadbeef",
                             n_items=1, n_with_gt=1)
    raw = Runner(fake_adapter, fake_encoder, warmup=0, repeat=1).run(
        [item], k=10, benchmark=bench, progress_path=str(path))

    # the streamed file is complete after the run and reads back identically to the
    # in-memory result — same header (system/k/benchmark) and same rows
    assert path.exists()
    reloaded = RawResults.read_jsonl(str(path))
    assert reloaded.system == raw.system == "fake"
    assert reloaded.k == raw.k == 10
    assert reloaded.benchmark is not None and reloaded.benchmark.label == bench.label
    assert [r.query_id for r in reloaded.results] == [r.query_id for r in raw.results]
    assert reloaded.results[0].ids == raw.results[0].ids


def test_progress_path_flushes_each_item_as_it_completes(
    enriched_item, fake_adapter, fake_encoder, tmp_path
):
    # the whole point is durability against a kill: item i's row must be on disk
    # BEFORE item i+1 runs, not batched at the end. Hook the adapter's search to read
    # the file back mid-run and record which query_ids are already persisted.
    from frame.core.schema import RawResults

    item1 = QueryItem.from_dict(enriched_item)
    other = copy.deepcopy(enriched_item)
    other["query_id"] = "q0002"
    item2 = QueryItem.from_dict(other)
    path = tmp_path / "raw_results.fake.jsonl"

    seen_ids_midway: set[str] = set()
    real_search = fake_adapter.search

    def spy(vec, filters, k):
        if path.exists():
            seen_ids_midway.update(
                r.query_id for r in RawResults.read_jsonl(str(path)).results
            )
        return real_search(vec, filters, k)

    fake_adapter.search = spy
    Runner(fake_adapter, fake_encoder, warmup=0, repeat=1).run(
        [item1, item2], k=5, progress_path=str(path))

    # q0001 was already flushed to disk while q0002's searches were still running
    assert "q0001" in seen_ids_midway

"""Unit tests for queryset/build.py — the authored-item validator + GT stub.

build.py is a standalone script (queryset/ is not an importable package), so it is
loaded here by file path.
"""

from __future__ import annotations

import copy
import importlib.util
import pathlib

import pytest

_BUILD_PATH = pathlib.Path(__file__).resolve().parents[1] / "queryset" / "build.py"
_spec = importlib.util.spec_from_file_location("queryset_build", _BUILD_PATH)
assert _spec is not None and _spec.loader is not None
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)


# ─── validate ───────────────────────────────────────────────────────────────

def test_valid_item_passes(authored_item):
    assert build.validate(authored_item) == []


def test_missing_required_top_level_key(authored_item):
    item = copy.deepcopy(authored_item)
    del item["raw_query_text"]
    errs = build.validate(item)
    assert any("raw_query_text" in e for e in errs)


def test_empty_vector_query_flagged(authored_item):
    item = copy.deepcopy(authored_item)
    item["decomposition"]["vector_query"] = ""
    errs = build.validate(item)
    assert any("vector_query" in e for e in errs)


def test_filter_missing_field_flagged(authored_item):
    item = copy.deepcopy(authored_item)
    del item["decomposition"]["filters"][0]["op"]
    errs = build.validate(item)
    assert any("filters[0]" in e and "op" in e for e in errs)


def test_scene_filter_empty_value_flagged(authored_item):
    item = copy.deepcopy(authored_item)
    item["decomposition"]["filters"][0]["value"] = []
    errs = build.validate(item)
    assert any("scene filter has empty value" in e for e in errs)


def test_empty_target_field_flagged(authored_item):
    item = copy.deepcopy(authored_item)
    item["target"]["end_s"] = None
    errs = build.validate(item)
    assert any("target.end_s" in e for e in errs)


def test_no_filter_item_is_valid(authored_item):
    item = copy.deepcopy(authored_item)
    item["decomposition"]["filters"] = []
    assert build.validate(item) == []


# ─── computed_stub ──────────────────────────────────────────────────────────

def test_computed_stub_selectivity_matches_filter_count(authored_item):
    stub = build.computed_stub(authored_item)
    assert stub["filter_selectivity"] == [None]  # one filter => one slot


def test_computed_stub_no_filters(authored_item):
    item = copy.deepcopy(authored_item)
    item["decomposition"]["filters"] = []
    stub = build.computed_stub(item)
    assert stub["filter_selectivity"] == []


def test_computed_stub_all_fields_pending(authored_item):
    stub = build.computed_stub(authored_item)
    for field in ("target_keyframe_ids", "target_passes_filter",
                  "geometric_gt_filtered", "geometric_gt_nofilter"):
        assert stub[field] is None
        assert field in stub["_pending"]


# ─── end-to-end main() over a temp query dir ────────────────────────────────

def test_main_compiles_items(tmp_path, authored_item, monkeypatch):
    import json

    qdir = tmp_path / "queries"
    qdir.mkdir()
    (qdir / "q0001.json").write_text(json.dumps(authored_item))
    # a template file (prefixed with _) must be ignored
    (qdir / "_TEMPLATE.json").write_text(json.dumps({"query_id": "TEMPLATE"}))
    out = tmp_path / "data" / "benchmark.jsonl"

    monkeypatch.setattr(build, "QDIR", str(qdir))
    monkeypatch.setattr(build, "OUT", str(out))

    with pytest.raises(SystemExit) as exc:
        build.main([])
    assert exc.value.code == 0  # no bad items

    lines = out.read_text().splitlines()
    assert len(lines) == 2                     # version header + one item
    header = json.loads(lines[0])
    assert header["corpus"] and header["digest"]
    assert header["n_items"] == 1
    assert header["n_with_gt"] == 0            # nothing enriched yet
    assert "q0001" in header["items"]
    compiled = json.loads(lines[1])
    assert compiled["query_id"] == "q0001"
    assert "computed" in compiled  # stub injected


def test_main_carries_ground_truth_forward(tmp_path, authored_item, enriched_item,
                                           monkeypatch):
    """GT costs an HPC job. A rebuild must not discard it for queries that did not
    change — and must discard it for one that did."""
    import json

    qdir = tmp_path / "queries"
    qdir.mkdir()
    (qdir / "q0001.json").write_text(json.dumps(authored_item))
    out = tmp_path / "data" / "benchmark.jsonl"
    out.parent.mkdir(parents=True)
    # a previous build, already enriched by the oracle
    out.write_text(json.dumps({"version": "1.0.0", "corpus": "v3c1", "digest": "x",
                               "items": {}, "gt_params": {}}) + "\n"
                   + json.dumps(enriched_item) + "\n")

    monkeypatch.setattr(build, "QDIR", str(qdir))
    monkeypatch.setattr(build, "OUT", str(out))
    with pytest.raises(SystemExit):
        build.main([])

    rebuilt = json.loads(out.read_text().splitlines()[1])
    assert rebuilt["computed"]["geometric_gt_filtered"] == \
        enriched_item["computed"]["geometric_gt_filtered"]

    # now edit the query itself — its GT is no longer valid and must be dropped
    edited = copy.deepcopy(authored_item)
    edited["decomposition"]["vector_query"] = "something else entirely"
    (qdir / "q0001.json").write_text(json.dumps(edited))
    with pytest.raises(SystemExit):
        build.main([])
    assert json.loads(out.read_text().splitlines()[1])["computed"][
        "geometric_gt_filtered"] is None


def test_main_rejects_invalid_item_nonzero_exit(tmp_path, authored_item, monkeypatch):
    import json

    qdir = tmp_path / "queries"
    qdir.mkdir()
    bad = copy.deepcopy(authored_item)
    del bad["target"]
    (qdir / "q0001.json").write_text(json.dumps(bad))

    monkeypatch.setattr(build, "QDIR", str(qdir))
    monkeypatch.setattr(build, "OUT", str(tmp_path / "data" / "benchmark.jsonl"))

    with pytest.raises(SystemExit) as exc:
        build.main([])
    assert exc.value.code == 1  # a rejected item => nonzero exit


def test_main_rejects_duplicate_ids(tmp_path, authored_item, monkeypatch):
    import json

    qdir = tmp_path / "queries"
    qdir.mkdir()
    (qdir / "a.json").write_text(json.dumps(authored_item))
    (qdir / "b.json").write_text(json.dumps(authored_item))  # same query_id

    monkeypatch.setattr(build, "QDIR", str(qdir))
    monkeypatch.setattr(build, "OUT", str(tmp_path / "data" / "benchmark.jsonl"))

    with pytest.raises(SystemExit) as exc:
        build.main([])
    assert exc.value.code == 1

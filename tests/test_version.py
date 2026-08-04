"""Unit tests for frame.core.version — what a run was measured against.

The point of this module is to REFUSE a wrong comparison, so most of these tests
assert that something is rejected. The digest tests matter most: if the digest
covers too much, harmless edits invalidate everyone's results; too little, and a
changed query slips through as comparable.
"""

from __future__ import annotations

import copy

import pytest

from frame.core.version import (
    HARNESS_CONTRACT,
    BenchmarkVersion,
    compare,
    item_digest,
    query_digest,
)


def _item(qid="q0001", text="a red car at night", vq="a red car",
          labels=("night",), video="00123", gt=True):
    it = {
        "query_id": qid,
        "status": "verified",
        "source": {"team": "verge", "timestamp": 123},
        "raw_query_text": text,
        "decomposition": {
            "vector_query": vq,
            "filters": [{"filter_type": "scene", "attribute": "scene_label",
                         "op": "in", "value": list(labels), "vocab": "places365",
                         "verified": True}],
            "method": "manual",
        },
        "target": {"video_id": video, "start_s": 10.0, "end_s": 12.0},
        "notes": "a note",
    }
    if gt:
        it["computed"] = {
            "target_keyframe_ids": ["kf_t"], "target_passes_filter": True,
            "filter_selectivity": [0.1],
            "geometric_gt_filtered": ["kf_t", "kf_a"],
            "geometric_gt_nofilter": ["kf_x"],
            "geometric_gt_raw_filtered": ["kf_t"],
            "geometric_gt_vec_nofilter": ["kf_t", "kf_p"],
            "scene_threshold": 0.10,
            "_pending": [],
        }
    return it


# ─── what the digest must and must not notice ───────────────────────────────

@pytest.mark.parametrize("mutate", [
    pytest.param(lambda i: i.update(raw_query_text="something else"), id="raw_text"),
    pytest.param(lambda i: i["decomposition"].update(vector_query="other"), id="vector_query"),
    pytest.param(lambda i: i["decomposition"]["filters"][0].update(value=["day"]), id="filter_value"),
    pytest.param(lambda i: i["decomposition"]["filters"][0].update(op="contains"), id="filter_op"),
    pytest.param(lambda i: i["decomposition"]["filters"].clear(), id="filter_removed"),
    pytest.param(lambda i: i["target"].update(video_id="09999"), id="target"),
])
def test_query_digest_changes_on_result_affecting_edits(mutate):
    a = _item()
    b = copy.deepcopy(a)
    mutate(b)
    assert query_digest(a) != query_digest(b)


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda i: i.update(notes="rewritten note"), id="notes"),
    pytest.param(lambda i: i.update(status="draft"), id="status"),
    pytest.param(lambda i: i.update(source={"team": "other"}), id="source"),
    pytest.param(lambda i: i["decomposition"]["filters"][0].update(verified=False), id="verified"),
    pytest.param(lambda i: i["decomposition"]["filters"][0].update(vocab="other"), id="vocab"),
    pytest.param(lambda i: i["decomposition"].update(method="assisted"), id="method"),
])
def test_query_digest_ignores_provenance_edits(mutate):
    """Provenance is not an input. Fixing a typo in a note must not invalidate
    somebody's completed run."""
    a = _item()
    b = copy.deepcopy(a)
    mutate(b)
    assert query_digest(a) == query_digest(b)


def test_item_digest_splits_query_from_gt():
    a = _item()
    b = copy.deepcopy(a)
    b["computed"]["geometric_gt_filtered"] = ["kf_z"]
    qa, ga = item_digest(a).split(":")
    qb, gb = item_digest(b).split(":")
    assert qa == qb          # query untouched
    assert ga != gb          # ground truth moved


def test_item_digest_gt_half_empty_without_gt():
    assert item_digest(_item(gt=False)).endswith(":")


def test_gt_digest_ignores_bookkeeping():
    a = _item()
    b = copy.deepcopy(a)
    b["computed"]["_pending"] = ["target_keyframe_ids"]
    b["computed"]["filter_diagnostics"] = {"anything": 1}
    assert item_digest(a) == item_digest(b)


# ─── BenchmarkVersion ───────────────────────────────────────────────────────

def _bv(items, version="1.0.0", corpus="v3c1", gt_params=None):
    return BenchmarkVersion.compute(version, corpus, items, gt_params or {})


def test_compute_counts_and_label():
    v = _bv([_item("q1"), _item("q2", gt=False)])
    assert v.n_items == 2
    assert v.n_with_gt == 1
    assert v.label == f"v3c1/1.0.0+{v.digest}"


def test_roundtrip_through_dict():
    v = _bv([_item("q1")], gt_params={"oracle_k": 100})
    assert BenchmarkVersion.from_dict(v.to_dict()) == v


# ─── compare() ──────────────────────────────────────────────────────────────

def test_identical_sets_are_identical():
    items = [_item("q1"), _item("q2")]
    c = compare(_bv(items), _bv(copy.deepcopy(items)))
    assert c.status == "identical"
    assert c.ok


def test_edited_query_is_incompatible_and_named():
    a = _bv([_item("q1"), _item("q2")])
    edited = [_item("q1"), _item("q2", text="totally different")]
    c = compare(a, _bv(edited))
    assert c.status == "incompatible"
    assert not c.ok
    assert "q2" in c.reason and "edited" in c.reason


def test_changed_gt_is_incompatible_and_distinguished_from_a_query_edit():
    a = _bv([_item("q1")])
    moved = _item("q1")
    moved["computed"]["geometric_gt_filtered"] = ["kf_different"]
    c = compare(a, _bv([moved]))
    assert c.status == "incompatible"
    assert "ground truth changed" in c.reason
    assert "edited" not in c.reason


def test_added_items_are_additive_and_scored_on_the_shared_set():
    """The yearly update: adding next year's queries must NOT throw away results
    for the queries that did not change."""
    a = _bv([_item("q1"), _item("q2")])
    b = _bv([_item("q1"), _item("q2"), _item("q3")])
    c = compare(a, b)
    assert c.status == "additive"
    assert c.ok
    assert c.shared == ["q1", "q2"]
    assert "added" in c.reason


def test_removed_items_are_additive_on_the_remainder():
    a = _bv([_item("q1"), _item("q2")])
    c = compare(a, _bv([_item("q1")]))
    assert c.status == "additive"
    assert c.shared == ["q1"]
    assert "gone" in c.reason


def test_different_corpus_is_incompatible():
    a = _bv([_item("q1")], corpus="v3c1")
    c = compare(a, _bv([_item("q1")], corpus="v3c2"))
    assert c.status == "incompatible"
    assert "corpus" in c.reason


def test_different_gt_parameters_are_incompatible():
    items = [_item("q1")]
    a = _bv(items, gt_params={"oracle_k": 100, "scene_threshold": 0.10})
    b = _bv(copy.deepcopy(items), gt_params={"oracle_k": 100, "scene_threshold": 0.15})
    c = compare(a, b)
    assert c.status == "incompatible"
    assert "0.1" in c.reason and "0.15" in c.reason


def test_missing_marker_is_incompatible():
    v = _bv([_item("q1")])
    assert compare(None, v).status == "incompatible"
    assert "predates versioning" in compare(None, v).reason
    assert compare(v, None).status == "incompatible"


def test_forgotten_version_bump_is_caught_by_the_digest():
    """The whole reason the digest exists: same semver, different contents."""
    a = _bv([_item("q1")], version="1.0.0")
    b = _bv([_item("q1", text="edited")], version="1.0.0")
    assert a.digest != b.digest
    assert compare(a, b).status == "incompatible"


def test_mis_set_version_on_identical_contents_is_reported_not_fatal():
    items = [_item("q1")]
    c = compare(_bv(items, version="1.0.0"),
                _bv(copy.deepcopy(items), version="1.1.0"))
    assert c.status == "identical"      # contents rule, not the label
    assert "mis-set" in c.reason


def test_harness_contract_is_an_int():
    assert isinstance(HARNESS_CONTRACT, int) and HARNESS_CONTRACT >= 1

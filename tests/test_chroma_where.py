"""Unit tests for ChromaAdapter's predicate -> (where, where_document) translation.

This is the system-under-test filter path for the denormalised layout, and — like
the pgvector COPY encoding — it is where a bug is silent: a mis-built `where`
does not raise, it just matches the wrong keyframes, and the recall number looks
plausible. The semantics pinned here mirror pgvector's `_filter_clauses`: AND
across predicates, OR (any-of) within one predicate's value list, and a
case-insensitive OCR substring (lower-cased on both sides).

Pure functions — no chromadb (the adapter imports it lazily), so these run in the
core CI with no extra installed.
"""

from __future__ import annotations

import pytest

from frame.adapters.chroma import _build_where, _build_where_document
from frame.core.schema import Predicate


def scene(*values) -> Predicate:
    return Predicate(filter_type="scene", attribute="scene_label", op="in",
                     value=list(values))


def obj(*values) -> Predicate:
    return Predicate(filter_type="object", attribute="object_label", op="in",
                     value=list(values))


def pattern(value) -> Predicate:
    return Predicate(filter_type="pattern-match", attribute="ocr_text",
                     op="contains", value=value)


class TestWhere:
    def test_no_filters_is_none(self):
        assert _build_where([]) is None

    def test_single_value_is_a_bare_contains(self):
        # One value -> no $or wrapper (Chroma wants a single top-level operator).
        assert _build_where([scene("night")]) == {"scene": {"$contains": "night"}}

    def test_multi_value_is_any_of(self):
        # label = ANY([...]) -> $or of $contains (the denormalised semi-join).
        assert _build_where([scene("mountain", "snowfield")]) == {
            "$or": [{"scene": {"$contains": "mountain"}},
                    {"scene": {"$contains": "snowfield"}}]
        }

    def test_two_predicates_are_anded(self):
        assert _build_where([scene("night"), obj("car")]) == {
            "$and": [{"scene": {"$contains": "night"}},
                     {"object": {"$contains": "car"}}]
        }

    def test_video_meta_maps_to_short_fields(self):
        cat = Predicate(filter_type="video-category", attribute="video_categories",
                        op="contains", value=["documentary"])
        tag = Predicate(filter_type="video-tag", attribute="video_tags",
                        op="contains", value=["news", "sport"])
        assert _build_where([cat]) == {"vcat": {"$contains": "documentary"}}
        assert _build_where([tag]) == {
            "$or": [{"vtag": {"$contains": "news"}},
                    {"vtag": {"$contains": "sport"}}]
        }

    def test_pattern_match_does_not_touch_metadata_where(self):
        assert _build_where([pattern("STOP")]) is None

    def test_scalar_value_is_accepted(self):
        # value may be a bare str, not only a list.
        assert _build_where([scene("night")]) == _build_where(
            [Predicate("scene", "scene_label", "in", "night")])

    def test_unsupported_filter_type_raises(self):
        bad = Predicate(filter_type="temporal", attribute="t", op="in", value=["x"])
        with pytest.raises(ValueError):
            _build_where([bad])


class TestWhereDocument:
    def test_no_pattern_is_none(self):
        assert _build_where_document([scene("night")]) is None

    def test_lowercases_to_match_stored_document(self):
        # $contains is case-sensitive; the document is stored lower-cased, so the
        # query substring must be too (reproduces pgvector's lower(text) LIKE).
        assert _build_where_document([pattern("STOP")]) == {"$contains": "stop"}

    def test_multi_value_is_any_of(self):
        assert _build_where_document([pattern(["Stop", "MIAMI"])]) == {
            "$or": [{"$contains": "stop"}, {"$contains": "miami"}]
        }

    def test_two_pattern_predicates_are_anded(self):
        assert _build_where_document([pattern("stop"), pattern("miami")]) == {
            "$and": [{"$contains": "stop"}, {"$contains": "miami"}]
        }

    def test_empty_values_are_dropped(self):
        # A stray empty string must not become a $contains "" that matches everything.
        assert _build_where_document([pattern(["", "stop"])]) == {"$contains": "stop"}


class TestCombined:
    def test_metadata_and_document_filters_coexist(self):
        # scene AND ocr: the scene clause lands in `where`, the OCR in
        # `where_document`; Chroma ANDs the two at query time.
        filters = [scene("night"), pattern("taxi")]
        assert _build_where(filters) == {"scene": {"$contains": "night"}}
        assert _build_where_document(filters) == {"$contains": "taxi"}

"""Unit tests for the COPY ... FROM STDIN (TEXT format) encoding used by
PgvectorAdapter.load_data().

This is the highest-risk code in the ingest path: a wrong escape does not raise,
it silently shifts columns or truncates a row, and you find out months later that
a filter matches the wrong keyframes. V3C really does contain video titles with
embedded newlines and OCR spans with tabs and backslashes (that is why the
original V3C1 load used TEXT rather than CSV), so those cases are pinned here.

Pure functions — no postgres, no psycopg2 (the adapter imports it lazily).
"""

from __future__ import annotations

import numpy as np

from frame.adapters.pgvector import (_copy_field, _copy_line, _LineReader,
                                     _pg_array_literal, _pg_escape,
                                     _vector_literal)


class TestEscaping:
    def test_backslash_is_escaped_first(self):
        # If \n were substituted before \\, the backslash of the escape itself
        # would be doubled and the field would decode as a literal "\n".
        assert _pg_escape("a\\b") == "a\\\\b"
        assert _pg_escape("a\\nb") == "a\\\\nb"      # literal backslash + 'n'
        assert _pg_escape("a\nb") == "a\\nb"         # real newline

    def test_row_delimiters_are_neutralised(self):
        """A raw tab or newline would end the field/row early and shift every
        column after it."""
        assert _pg_escape("a\tb") == "a\\tb"
        assert _pg_escape("a\r\nb") == "a\\r\\nb"

    def test_plain_text_is_untouched(self):
        assert _pg_escape("a red car, at night") == "a red car, at night"

    def test_quotes_need_no_escaping_in_text_format(self):
        # Unlike CSV — this is the whole reason TEXT format was chosen.
        assert _pg_escape('he said "hi"') == 'he said "hi"'


class TestArrayLiterals:
    def test_elements_are_quoted(self):
        assert _pg_array_literal(["news", "sport"]) == '{"news","sport"}'

    def test_empty_array(self):
        assert _pg_array_literal([]) == "{}"

    def test_comma_inside_element_stays_one_element(self):
        assert _pg_array_literal(["a,b"]) == '{"a,b"}'

    def test_quote_and_backslash_inside_element(self):
        assert _pg_array_literal(['a"b']) == '{"a\\"b"}'
        assert _pg_array_literal(["a\\b"]) == '{"a\\\\b"}'

    def test_null_element(self):
        assert _pg_array_literal(["a", None]) == '{"a",NULL}'


class TestFieldEncoding:
    def test_none_is_the_null_marker(self):
        assert _copy_field(None) == "\\N"

    def test_empty_string_is_not_null(self):
        assert _copy_field("") == ""

    def test_numbers(self):
        assert _copy_field(42) == "42"
        assert _copy_field(0.5) == "0.5"

    def test_bool(self):
        assert _copy_field(True) == "t"
        assert _copy_field(False) == "f"

    def test_array_field_is_escaped_after_being_built(self):
        """Two layers: array literal first, then COPY escaping over the result —
        otherwise a backslash in a tag corrupts the row."""
        assert _copy_field(["a\\b"]) == '{"a\\\\\\\\b"}'

    def test_line_is_tab_separated_and_newline_terminated(self):
        assert _copy_line(("k1", None, 3)) == "k1\t\\N\t3\n"

    def test_multiline_title_stays_one_row(self):
        line = _copy_line(("00123", "Title\nwith break"))
        assert line.count("\n") == 1 and line.endswith("\n")


class TestVectorLiteral:
    def test_format_matches_pgvector_text_input(self):
        v = np.array([0.5, -0.25], dtype=np.float32)
        assert _vector_literal(v) == "[0.50000000,-0.25000000]"

    def test_precision_matches_the_search_path(self):
        # search() formats the query vector with the same %.8f, so a stored and a
        # queried vector round-trip identically.
        v = np.array([1 / 3], dtype=np.float32)
        assert _vector_literal(v) == "[0.33333334]"


class TestLineReader:
    def test_read_spans_line_boundaries(self):
        r = _LineReader(iter(["abc\n", "de\n"]))
        assert r.read(2) == "ab"
        assert r.read(4) == "c\nde"
        assert r.read(4) == "\n"
        assert r.read(4) == ""

    def test_read_all(self):
        assert _LineReader(iter(["a\n", "b\n"])).read(-1) == "a\nb\n"

    def test_readline(self):
        r = _LineReader(iter(["a\n", "b\n"]))
        assert r.readline() == "a\n"
        assert r.readline() == "b\n"
        assert r.readline() == ""

    def test_is_lazy(self):
        """The generator must not be drained up front — that is what keeps a
        multi-GB COPY inside a few hundred KB of RAM."""
        pulled = []

        def gen():
            for i in range(1000):
                pulled.append(i)
                yield f"{i}\n"

        r = _LineReader(gen())
        r.read(4)
        assert len(pulled) < 10

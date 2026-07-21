"""
pgvector adapter — the first system under test (build this one first).

This is the SYSTEM-UNDER-TEST predicate translation. It is intentionally a
SEPARATE code path from oracle/build_gt.py's predicate_to_sql(): the oracle
computes exact truth (sequential scan, indexes off); this asks the real HNSW
index and lives with its approximation. The gap between them is a result we
measure — so the two must not share translation code.

Physical layout: pgvector holds the normalised V3C schema (see V3C Schema.md), so
`setup()` here is a connection + index sanity check + a pinned ANALYZE of every
filter relation (fairness invariant — see _ANALYZE_TABLES) — the "ingest the logical
schema" step is already satisfied by the existing V3C load. Filtering is done with
native JOIN/EXISTS against the side tables. (Chroma/Milvus adapters will instead
denormalise in their own setup() — that asymmetry is the research point.)

FAIRNESS: confidence thresholds are PINNED (scene 0.10, object 0.30 — the values
chosen in the query-set repo diagnostics) so every system filters over the same
passing universe. Pattern-match (OCR) has no threshold by design.

Requires the `pgvector` extra (psycopg2). Connection via libpq env vars
(PGHOST/PGUSER/…); on the HPC PGHOST is the unix socket dir (see HPC notes).
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from ..core.adapter import VectorDBAdapter
from ..core.schema import Predicate

# Pinned fairness thresholds (see query-set repo "Chosen confidence thresholds").
SCENE_THRESHOLD = 0.10
OBJECT_THRESHOLD = 0.30

# Side-table sources for label filters: filter_type -> (table, alias, threshold).
_LABEL_SOURCES = {
    "scene":  ("scene_labels", "sl", SCENE_THRESHOLD),
    "object": ("object_detections", "od", OBJECT_THRESHOLD),
}

# FAIRNESS INVARIANT — pinned statistics state. Every relation this adapter's
# filtered searches plan over is ANALYZEd in setup(), so the planner's selectivity
# estimates (and therefore the exact-seqscan ↔ approximate-HNSW plan choice, and
# thus recall) are deterministic instead of hostage to whenever autovacuum last ran.
# Array-overlap (`&&`) columns like videos.categories in particular have NO
# most-common-elements stats until ANALYZEd, so the planner falls back to a blind
# constant estimate and always picks exact — masking the very cutover we measure.
_ANALYZE_TABLES = ("keyframes", "scene_labels", "object_detections", "keyframe_ocr", "videos")


class PgvectorAdapter(VectorDBAdapter):
    name = "pgvector"

    def __init__(
        self,
        dsn: str | None = None,
        ef_search: int = 1000,
        iterative_scan: str = "relaxed_order",
    ):
        # dsn=None -> libpq reads PG* env vars (PGHOST socket dir on the HPC).
        # ef_search is capped at 1000 by pgvector and must be >= the run's k;
        # for a top-1000 run it is pinned at the ceiling (=k, no recall headroom).
        # iterative_scan lets a FILTERED search keep pulling candidates from the
        # HNSW index when the predicate prunes them, instead of the planner
        # falling back to a sequential scan (exact but slow). Modes: 'off',
        # 'relaxed_order', 'strict_order'.
        self.dsn = dsn
        self.ef_search = ef_search
        self.iterative_scan = iterative_scan
        self._conn = None

    def setup(self) -> None:
        import psycopg2  # optional dep (`pgvector` extra)

        self._conn = psycopg2.connect(self.dsn) if self.dsn else psycopg2.connect()
        self._conn.autocommit = True
        with self._conn.cursor() as cur:
            # HNSW search-time knobs; set per session.
            cur.execute("SET hnsw.ef_search = %s;", (self.ef_search,))
            cur.execute("SET hnsw.iterative_scan = %s;", (self.iterative_scan,))
            # Sanity: the vector index we rely on must exist.
            cur.execute("SELECT to_regclass('public.keyframes_embedding_hnsw_idx');")
            row = cur.fetchone()
            if row is None or row[0] is None:
                raise RuntimeError("keyframes HNSW index missing — check the V3C load")
            # Pin the statistics state (fairness invariant): refresh planner stats on
            # every filter relation so plan choice + recall are reproducible. Skip any
            # table absent from this deployment (e.g. before a schema is fully loaded).
            for table in _ANALYZE_TABLES:
                cur.execute("SELECT to_regclass(%s);", (f"public.{table}",))
                if _scalar(cur) is not None:
                    cur.execute(f"ANALYZE {table};")

    def teardown(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def search(
        self,
        query_vector: np.ndarray,
        filters: Sequence[Predicate],
        k: int,
    ) -> list[str]:
        assert self._conn is not None, "call setup() first"
        sql, params = self._search_sql(filters)
        vec_literal = "[" + ",".join(f"{x:.8f}" for x in query_vector.tolist()) + "]"
        with self._conn.cursor() as cur:
            cur.execute(sql, (*params, vec_literal, k))
            return [row[0] for row in cur.fetchall()]

    def _search_sql(self, filters: Sequence[Predicate]) -> tuple[str, list]:
        where_sql, params = self._build_filters(filters)
        sql = (
            "SELECT k.keyframe_id FROM keyframes k "
            f"{where_sql} ORDER BY k.embedding <=> %s::vector LIMIT %s"
        )
        return sql, params

    def explain(
        self,
        query_vector: np.ndarray,
        filters: Sequence[Predicate],
        k: int,
        iterative_scan: str | None = None,
    ) -> str:
        """Run EXPLAIN (ANALYZE, BUFFERS) on the exact SQL search() would run, under
        a chosen iterative_scan mode. Diagnostic — confirms whether a filtered query
        uses the HNSW index or falls back to a sequential scan."""
        assert self._conn is not None, "call setup() first"
        sql, params = self._search_sql(filters)
        vec_literal = "[" + ",".join(f"{x:.8f}" for x in query_vector.tolist()) + "]"
        mode = iterative_scan or self.iterative_scan
        with self._conn.cursor() as cur:
            cur.execute("SET hnsw.iterative_scan = %s;", (mode,))
            cur.execute("EXPLAIN (ANALYZE, BUFFERS) " + sql, (*params, vec_literal, k))
            plan = "\n".join(r[0] for r in cur.fetchall())
            # restore the adapter's configured mode
            cur.execute("SET hnsw.iterative_scan = %s;", (self.iterative_scan,))
        return plan

    # ── profiling diagnostics (selectivity + plan choice) ──
    # Thin read-only helpers for the query-set profiler (frame.core.profile). Not
    # part of the adapter run contract — a diagnostic, like explain(). They reuse
    # _build_filters so counts are taken over the SAME passing universe (pinned
    # thresholds) the real search filters on.
    def corpus_size(self) -> int:
        assert self._conn is not None, "call setup() first"
        with self._conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM keyframes")
            return _scalar(cur)

    def count_passing(self, filters: Sequence[Predicate]) -> int:
        """True global count of keyframes satisfying the FULL (AND-ed) predicate —
        the real selectivity, not the planner's estimate."""
        assert self._conn is not None, "call setup() first"
        where_sql, params = self._build_filters(filters)
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM keyframes k {where_sql}", params)
            return _scalar(cur)

    def count_members(self, ids: Sequence[str], filters: Sequence[Predicate]) -> int:
        """How many of `ids` satisfy the predicate — used to turn a query's exact
        unfiltered neighbours into a near-query pass-rate."""
        assert self._conn is not None, "call setup() first"
        if not ids:
            return 0
        clause_sql, params = self._filter_clauses(filters)
        cond = f"AND {clause_sql}" if clause_sql else ""
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT count(*) FROM keyframes k WHERE k.keyframe_id = ANY(%s) {cond}",
                [list(ids), *params],
            )
            return _scalar(cur)

    def plan_choice(
        self,
        query_vector: np.ndarray,
        filters: Sequence[Predicate],
        k: int,
        iterative_scan: str | None = None,
    ) -> tuple[str, int | None]:
        """Which strategy the planner picks for this filtered search, and its
        estimated passing set. Uses EXPLAIN (FORMAT JSON) WITHOUT ANALYZE — reads
        the chosen plan + row estimates without executing the query (cheap).

        Returns (plan, est_rows):
          * plan      — "hnsw" if the vector index is scanned (approximate post-filter),
                        "seqscan" if keyframes is sequentially scanned (exact filter-then-scan),
                        "other" otherwise.
          * est_rows  — the planner's estimated passing-set rows: the estimate of the
                        node feeding the top-N (Limit's child), which is what drives the
                        seqscan-vs-HNSW cost choice. A proxy — see the plan tree for detail.
        """
        assert self._conn is not None, "call setup() first"
        sql, params = self._search_sql(filters)
        vec_literal = "[" + ",".join(f"{x:.8f}" for x in query_vector.tolist()) + "]"
        mode = iterative_scan or self.iterative_scan
        with self._conn.cursor() as cur:
            cur.execute("SET hnsw.iterative_scan = %s;", (mode,))
            cur.execute("EXPLAIN (FORMAT JSON) " + sql, (*params, vec_literal, k))
            raw = _scalar(cur)
            cur.execute("SET hnsw.iterative_scan = %s;", (self.iterative_scan,))
        plan_tree = (json.loads(raw) if isinstance(raw, str) else raw)[0]["Plan"]
        return _classify_plan(plan_tree), _estimated_passing_rows(plan_tree)

    # ── predicate translation (system-under-test side) ──
    def _build_filters(self, filters: Sequence[Predicate]) -> tuple[str, list]:
        """WHERE fragment (or empty string) for the AND-ed predicate."""
        clause_sql, params = self._filter_clauses(filters)
        where = ("WHERE " + clause_sql) if clause_sql else ""
        return where, params

    def _filter_clauses(self, filters: Sequence[Predicate]) -> tuple[str, list]:
        """AND-ed EXISTS subqueries against the side tables, WITHOUT a leading
        WHERE (so callers can splice them into a larger predicate). Table/alias/
        column names come only from our own constants, never from item data, so
        they are safe to interpolate; all VALUES are bound parameters."""
        clauses: list[str] = []
        params: list = []
        for f in filters:
            if f.filter_type in _LABEL_SOURCES:
                table, alias, thresh = _LABEL_SOURCES[f.filter_type]
                clauses.append(
                    f"EXISTS (SELECT 1 FROM {table} {alias} "
                    f"WHERE {alias}.keyframe_id = k.keyframe_id "
                    f"AND {alias}.label = ANY(%s) AND {alias}.confidence >= %s)"
                )
                params.append(list(_as_list(f.value)))
                params.append(thresh)
            elif f.filter_type == "pattern-match":
                # case-insensitive substring over OCR spans; match ANY value.
                likes = ["%" + v.lower() + "%" for v in _as_list(f.value) if v]
                clauses.append(
                    "EXISTS (SELECT 1 FROM keyframe_ocr o "
                    "WHERE o.keyframe_id = k.keyframe_id "
                    "AND lower(o.text) LIKE ANY(%s))"
                )
                params.append(likes)
            else:
                raise ValueError(f"unsupported filter_type: {f.filter_type!r}")
        return " AND ".join(clauses), params


def _as_list(value) -> list:
    return [value] if isinstance(value, str) else list(value)


def _scalar(cur):
    """First column of the single row a scalar query returns (count / EXPLAIN json)."""
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("expected one row, got none")
    return row[0]


# ── EXPLAIN (FORMAT JSON) plan parsing (used by plan_choice) ──
HNSW_INDEX = "keyframes_embedding_hnsw_idx"


def _walk_plan(node: dict):
    """Yield this plan node and all its descendants."""
    yield node
    for child in node.get("Plans", []):
        yield from _walk_plan(child)


def _classify_plan(root: dict) -> str:
    """hnsw (vector index scanned → approximate post-filter) | seqscan (keyframes
    sequentially scanned → exact filter-then-scan) | other."""
    nodes = list(_walk_plan(root))
    if any(n.get("Index Name") == HNSW_INDEX for n in nodes):
        return "hnsw"
    if any(n.get("Node Type", "").endswith("Seq Scan")
           and n.get("Relation Name") == "keyframes" for n in nodes):
        return "seqscan"
    return "other"


def _estimated_passing_rows(root: dict) -> int | None:
    """Planner's estimated passing set: the row estimate of the node feeding the
    top-N. The root is the Limit (est rows = k, uninformative), so take its child;
    that node's estimate is the passing set the planner priced the choice on."""
    node = root
    if node.get("Node Type") == "Limit":
        children = node.get("Plans") or []
        if children:
            node = children[0]
    rows = node.get("Plan Rows")
    return int(rows) if rows is not None else None

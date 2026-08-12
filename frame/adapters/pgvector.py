"""
pgvector adapter — the first system under test (build this one first).

This is the SYSTEM-UNDER-TEST predicate translation. It is intentionally a
SEPARATE code path from oracle/build_gt.py's predicate_to_sql(): the oracle
computes exact truth (sequential scan, indexes off); this asks the real HNSW
index and lives with its approximation. The gap between them is a result we
measure — so the two must not share translation code.

Physical layout: pgvector holds the NORMALISED V3C schema (see V3C Schema.md).
`load_data()` materialises a Tier-2 canonical shard into exactly that layout —
one table per entity + the HNSW index — and `setup()` is then only a connection +
index sanity check + a pinned ANALYZE of every filter relation (fairness
invariant, see _ANALYZE_TABLES). Filtering is done with native JOIN/EXISTS against
the side tables. (Chroma/Milvus adapters will instead denormalise in their own
load_data() — that asymmetry is the research point.)

FAIRNESS: confidence thresholds are PINNED (scene 0.10, object 0.30 — the values
chosen in the query-set repo diagnostics) so every system filters over the same
passing universe. Pattern-match (OCR) has no threshold by design.

Requires the `pgvector` extra (psycopg2). Connection via libpq env vars
(PGHOST/PGUSER/…); on the HPC PGHOST is the unix socket dir (see HPC notes).
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np

from ..core.adapter import VectorDBAdapter
from ..core.dataset import EMBED_DIM, Dataset
from ..core.schema import Predicate

# Pinned fairness thresholds (see query-set repo "Chosen confidence thresholds").
SCENE_THRESHOLD = 0.10
OBJECT_THRESHOLD = 0.30

# Side-table sources for label filters: filter_type -> (table, alias, threshold).
_LABEL_SOURCES = {
    "scene":  ("scene_labels", "sl", SCENE_THRESHOLD),
    "object": ("object_detections", "od", OBJECT_THRESHOLD),
}

# Video-level metadata filters: filter_type -> videos.<column> (curated V3C ARRAY,
# set-overlap, NO confidence floor). Broad facets that exercise the planner's
# selectivity estimate — see the ANALYZE fairness invariant below.
_VIDEO_META_SOURCES = {
    "video-category": "categories",
    "video-tag": "tags",
}

# FAIRNESS INVARIANT — pinned statistics state. Every relation this adapter's
# filtered searches plan over is ANALYZEd in setup(), so the planner's selectivity
# estimates (and therefore the exact-seqscan ↔ approximate-HNSW plan choice, and
# thus recall) are deterministic instead of hostage to whenever autovacuum last ran.
# Array-overlap (`&&`) columns like videos.categories in particular have NO
# most-common-elements stats until ANALYZEd, so the planner falls back to a blind
# constant estimate and always picks exact — masking the very cutover we measure.
_ANALYZE_TABLES = ("keyframes", "scene_labels", "object_detections", "keyframe_ocr", "videos")

# ── Physical layout materialised by load_data() ────────────────────────────────
# DDL and index set reproduce V3C Schema.md EXACTLY. That fidelity is not
# cosmetic: which indexes exist decides what the planner can choose, and the
# seqscan↔HNSW cutover is precisely what this suite measures. Adding a helpful
# index here would silently change the results, so don't — change the schema doc
# first, deliberately.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64

_DDL = {
    "videos": """
        CREATE TABLE IF NOT EXISTS videos (
            video_id    TEXT PRIMARY KEY, vimeo_id TEXT, title TEXT,
            duration_s  DOUBLE PRECISION, width INTEGER, height INTEGER,
            channel     TEXT, upload_date TEXT, license TEXT,
            tags        TEXT[], categories TEXT[])""",
    "shots": """
        CREATE TABLE IF NOT EXISTS shots (
            shot_id      TEXT PRIMARY KEY, video_id TEXT, shot_index INTEGER,
            start_frame  INTEGER, end_frame INTEGER,
            start_time_s DOUBLE PRECISION, end_time_s DOUBLE PRECISION)""",
    "keyframes": f"""
        CREATE TABLE IF NOT EXISTS keyframes (
            keyframe_id TEXT PRIMARY KEY, shot_id TEXT, video_id TEXT,
            frame_number INTEGER, embedding vector({EMBED_DIM}))""",
    "keyframe_ocr": """
        CREATE TABLE IF NOT EXISTS keyframe_ocr (
            keyframe_id TEXT NOT NULL, span_index INTEGER NOT NULL,
            text TEXT NOT NULL, confidence REAL NOT NULL,
            PRIMARY KEY (keyframe_id, span_index))""",
    "scene_labels": """
        CREATE TABLE IF NOT EXISTS scene_labels (
            keyframe_id TEXT NOT NULL, label TEXT NOT NULL,
            confidence DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (keyframe_id, label))""",
    "object_detections": """
        CREATE TABLE IF NOT EXISTS object_detections (
            id BIGINT PRIMARY KEY, keyframe_id TEXT NOT NULL, label TEXT NOT NULL,
            confidence REAL NOT NULL, x1 REAL, y1 REAL, x2 REAL, y2 REAL)""",
    "object_detection_done": """
        CREATE TABLE IF NOT EXISTS object_detection_done (
            keyframe_id TEXT PRIMARY KEY)""",
}

# Built AFTER the COPY (index-then-load is far slower, and the HNSW build in
# particular must see the finished table).
_INDEXES = (
    "CREATE INDEX IF NOT EXISTS shots_video_id_idx ON shots USING btree (video_id)",
    "CREATE INDEX IF NOT EXISTS keyframes_video_id_idx ON keyframes USING btree (video_id)",
    "CREATE INDEX IF NOT EXISTS keyframe_ocr_keyframe_id ON keyframe_ocr USING btree (keyframe_id)",
    "CREATE INDEX IF NOT EXISTS keyframe_ocr_conf ON keyframe_ocr USING btree (confidence)",
    "CREATE INDEX IF NOT EXISTS keyframe_ocr_text_trgm "
    "ON keyframe_ocr USING gin (lower(text) gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS idx_scene_labels_keyframe_id ON scene_labels USING btree (keyframe_id)",
    "CREATE INDEX IF NOT EXISTS idx_scene_labels_label ON scene_labels USING btree (label)",
    "CREATE INDEX IF NOT EXISTS idx_object_detections_keyframe_id "
    "ON object_detections USING btree (keyframe_id)",
    "CREATE INDEX IF NOT EXISTS idx_object_detections_label ON object_detections USING btree (label)",
    f"CREATE INDEX IF NOT EXISTS {'keyframes_embedding_hnsw_idx'} ON keyframes "
    f"USING hnsw (embedding vector_cosine_ops) "
    f"WITH (m='{HNSW_M}', ef_construction='{HNSW_EF_CONSTRUCTION}')",
)

# Index names, for the union (re)load which must DROP them before appending several
# shards and rebuild ONCE at the end — streaming millions of rows into a live HNSW
# index is correct but pathologically slow. Kept in sync with _INDEXES by hand
# (there are ten of them); a mismatch only means a stale index is left to be
# rebuilt, which CREATE INDEX IF NOT EXISTS would then skip — so keep them aligned.
_INDEX_NAMES = (
    "shots_video_id_idx", "keyframes_video_id_idx", "keyframe_ocr_keyframe_id",
    "keyframe_ocr_conf", "keyframe_ocr_text_trgm", "idx_scene_labels_keyframe_id",
    "idx_scene_labels_label", "idx_object_detections_keyframe_id",
    "idx_object_detections_label", "keyframes_embedding_hnsw_idx",
)

# Columns COPYed per table, in parquet order. keyframes is absent on purpose: its
# rows are assembled from the metadata parquet + the embeddings h5 (see
# _load_keyframes), since the vector lives in a separate file.
_COPY_COLUMNS = {
    "videos": ("video_id", "vimeo_id", "title", "duration_s", "width", "height",
               "channel", "upload_date", "license", "tags", "categories"),
    "shots": ("shot_id", "video_id", "shot_index", "start_frame", "end_frame",
              "start_time_s", "end_time_s"),
    "keyframe_ocr": ("keyframe_id", "span_index", "text", "confidence"),
    "scene_labels": ("keyframe_id", "label", "confidence"),
    "object_detections": ("id", "keyframe_id", "label", "confidence", "x1", "y1", "x2", "y2"),
    "object_detection_done": ("keyframe_id",),
}

_KEYFRAME_COLUMNS = ("keyframe_id", "shot_id", "video_id", "frame_number", "embedding")


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

    # ── one-time ingest (Tier 2 -> pgvector's physical layout) ──
    def load_data(self, dataset: Dataset, force: bool = False) -> None:
        """COPY one canonical shard into the normalised V3C tables, then build the
        indexes. Idempotent: a table whose row count already matches the source
        parquet is skipped; a partially-loaded one is truncated and redone (the
        source is files, so a redo is always safe). `force=True` reloads
        everything. The ABC entry point — a thin wrapper over load_datasets()."""
        self.load_datasets([dataset], force=force)

    def load_datasets(self, datasets: Sequence[Dataset], force: bool = False) -> None:
        """Ingest one OR MORE canonical shards into a SINGLE pgvector instance — the
        union corpus (v3c1+2+3): one HNSW index over every shard's vectors, which is
        what a filtered-ANN benchmark at scale must search.

        One shard keeps the old idempotent, per-table, resume-friendly path. Several
        shards take the union path: it truncates, appends every shard, and builds the
        indexes ONCE at the end (see _load_union). The shards' primary keys are
        disjoint across shards (video/shot/keyframe ids are corpus-wide unique) with
        one exception — object_detections.id is assigned per shard at consolidate, so
        the union append reassigns it from a running counter to keep it unique.

        Streamed end to end — parquet row batches feed COPY ... FROM STDIN, so a
        multi-million-row shard loads in bounded memory. The one thing held whole is
        each shard's keyframe metadata dict (~4 small fields x N rows), needed
        because vectors arrive from a separate file and must be paired by id."""
        datasets = list(datasets)
        if not datasets:
            raise ValueError("load_datasets: no datasets given")
        for ds in datasets:
            ds.validate()

        own_conn = self._conn is None
        if own_conn:
            self._connect()
        conn = self._conn
        assert conn is not None

        try:
            names = ", ".join(d.name for d in datasets)
            print(f"[load] {names} -> pgvector", flush=True)
            with conn.cursor() as cur:
                self._create_schema(cur)

            if len(datasets) == 1:
                ds = datasets[0]
                self._load_keyframes(ds, force=force)
                for table in _COPY_COLUMNS:
                    if ds.has_table(table):
                        self._copy_table(ds, table, force=force)
                    else:
                        print(f"[load] skip {table}: not in this shard", flush=True)
                build = True
            else:
                build = self._load_union(datasets, force=force)

            if build:
                self._build_indexes()
            self._analyze()
            print("[load] done", flush=True)
        finally:
            if own_conn:
                self.teardown()

    def _create_schema(self, cur) -> None:
        import psycopg2
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # pg_trgm backs keyframe_ocr_text_trgm (the pattern-match filter).
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        except psycopg2.Error as e:
            print(f"[load] WARNING: pg_trgm unavailable ({e}); "
                  "the OCR trigram index will be skipped", flush=True)
        for ddl in _DDL.values():
            cur.execute(ddl)

    def _build_indexes(self) -> None:
        import psycopg2
        assert self._conn is not None
        print("[load] building indexes (this is the slow part) ...", flush=True)
        with self._conn.cursor() as cur:
            # HNSW build over the ~4.1M-vector union is the dominant cost. Two knobs
            # decide whether it finishes in hours or days, both session-scoped:
            #   * maintenance_work_mem — if the in-progress graph doesn't fit, pgvector
            #     spills to an on-disk build that is dramatically slower. Size it to
            #     hold the graph (comfortably under the job's --mem).
            #   * max_parallel_maintenance_workers — pgvector builds HNSW in parallel;
            #     the default (2) leaves most of an --cpus-per-task=N job idle.
            # Both are read from env so the SLURM script owns the numbers alongside its
            # --mem / --cpus request; the defaults keep a laptop run sane.
            mwm = os.environ.get("FRAME_MAINTENANCE_WORK_MEM", "2GB")
            workers = os.environ.get("FRAME_INDEX_PARALLEL_WORKERS", "0")
            cur.execute(f"SET maintenance_work_mem = '{mwm}';")
            cur.execute(f"SET max_parallel_maintenance_workers = {int(workers)};")
            print(f"[load]   maintenance_work_mem={mwm}, "
                  f"max_parallel_maintenance_workers={workers}", flush=True)
            for stmt in _INDEXES:
                try:
                    cur.execute(stmt)
                except psycopg2.Error as e:
                    # A missing extension (pg_trgm) must not sink the whole load.
                    print(f"[load] WARNING: index failed: {e}", flush=True)

    def _analyze(self) -> None:
        assert self._conn is not None
        with self._conn.cursor() as cur:
            for table in _ANALYZE_TABLES:
                if _table_exists(cur, table):
                    cur.execute(f"ANALYZE {table};")

    def _drop_indexes(self) -> None:
        assert self._conn is not None
        with self._conn.cursor() as cur:
            for name in _INDEX_NAMES:
                cur.execute(f"DROP INDEX IF EXISTS {name};")

    def _hnsw_exists(self) -> bool:
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.keyframes_embedding_hnsw_idx');")
            return _scalar(cur) is not None

    def _load_union(self, datasets: Sequence[Dataset], force: bool) -> bool:
        """(Re)load several shards into one instance; return whether indexes need
        building (False = the union was already loaded and nothing changed).

        Whole-union idempotency: if keyframes already holds exactly the summed
        vector count and the HNSW index is present, this is a no-op. Anything else
        (a partial load, a single-shard load, or --force) drops the indexes,
        truncates every table, and appends all shards fresh — then load_datasets
        rebuilds the indexes once. A cross-shard append is safe only after the
        indexes are gone; the alternative (inserting millions of rows into a live
        HNSW index) is correct but far too slow."""
        assert self._conn is not None
        total_kf = sum(ds.embedding_count() for ds in datasets)
        with self._conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM keyframes;")
            have = _scalar(cur)

        if not force and have == total_kf and total_kf > 0 and self._hnsw_exists():
            print(f"[load] union already loaded: {have:,} keyframes across "
                  f"{len(datasets)} shards; skipping (use force to rebuild)", flush=True)
            return False
        if not force and have == total_kf and total_kf > 0:
            # All rows are present but the HNSW index is not — the exact state a
            # load job left behind when it was killed mid-CREATE INDEX (the COPYs
            # committed under autocommit; the final HNSW build rolled back). Resume
            # by (re)building indexes ONLY: skip the multi-million-row reload, and
            # let CREATE INDEX IF NOT EXISTS no-op the btrees that already survived.
            print(f"[load] union rows present ({have:,}) but HNSW index missing; "
                  "resuming at index build (no reload)", flush=True)
            return True
        if have and not force:
            raise RuntimeError(
                f"keyframes holds {have:,} rows but the union expects {total_kf:,}. "
                "A partial or single-shard load is present. Re-run with force=True "
                "(scripts/load_dataset.py --force) to truncate and rebuild the union.")

        print(f"[load] union (re)load: dropping indexes and truncating "
              f"{len(datasets)} shards -> {total_kf:,} keyframes", flush=True)
        self._drop_indexes()
        with self._conn.cursor() as cur:
            # One TRUNCATE for all tables at once: truncating them separately fails
            # on FK constraints (e.g. shots references videos), and CASCADE keeps it
            # robust to FK edges not spelled out here. All targets are emptied anyway.
            targets = [t for t in ("keyframes",) + tuple(_COPY_COLUMNS)
                       if _table_exists(cur, t)]
            if targets:
                cur.execute(f"TRUNCATE {', '.join(targets)} CASCADE;")

        od_counter = [0]   # global object_detections id, unique across shards
        for ds in datasets:
            print(f"[load] shard {ds.name} ->", flush=True)
            self._copy_keyframes(ds, ds.embedding_count())
            for table in _COPY_COLUMNS:
                if not ds.has_table(table):
                    print(f"[load]   skip {table}: not in {ds.name}", flush=True)
                    continue
                counter = od_counter if table == "object_detections" else None
                self._append_table(ds, table, id_counter=counter)
        return True

    def _load_keyframes(self, dataset: Dataset, force: bool = False) -> None:
        """Guarded single-shard keyframe load — skip/append/raise per _prepare_table,
        then stream the rows (shared with the union path via _copy_keyframes)."""
        n_vectors = dataset.embedding_count()
        if not self._prepare_table("keyframes", n_vectors, force=force):
            return
        self._copy_keyframes(dataset, n_vectors)

    def _copy_keyframes(self, dataset: Dataset, n_vectors: int) -> None:
        """keyframes rows = metadata parquet JOINed by id to the embeddings h5.

        The h5 drives the loop because it is the big side and must stay streamed;
        a keyframe with no vector is skipped by design (it could never be a k-NN
        result, and its presence would only distort corpus_size). Assumes the target
        table is ready to receive rows (caller did the guard/truncate)."""
        assert self._conn is not None
        meta = {}
        for batch in dataset.iter_table(
                "keyframes", columns=["keyframe_id", "shot_id", "video_id", "frame_number"]):
            for row in batch:
                meta[row["keyframe_id"]] = (row["shot_id"], row["video_id"], row["frame_number"])
        print(f"[load] keyframes: {len(meta):,} metadata rows, {n_vectors:,} vectors", flush=True)

        missing = 0
        written = 0

        def rows():
            nonlocal missing, written
            for ids, vecs in dataset.iter_embeddings():
                for kid, vec in zip(ids, vecs):
                    m = meta.get(kid)
                    if m is None:
                        missing += 1
                        continue
                    shot_id, video_id, frame_number = m
                    yield _copy_line((kid, shot_id, video_id, frame_number,
                                      _vector_literal(vec)))
                    written += 1
                print(f"[load]   keyframes {written:,}/{n_vectors:,}", flush=True)

        self._copy_from(("keyframes", _KEYFRAME_COLUMNS), rows())
        if missing:
            print(f"[load] WARNING: {missing:,} vectors had no metadata row "
                  "(shard metadata and embeddings disagree)", flush=True)
        orphan = len(meta) - written
        if orphan > 0:
            print(f"[load] note: {orphan:,} keyframes have no vector — not loaded",
                  flush=True)

    def _copy_table(self, dataset: Dataset, table: str, force: bool = False) -> None:
        """Guarded single-shard table load (skip/append/raise), then stream rows."""
        expected = dataset.table_rows(table)
        if not self._prepare_table(table, expected, force=force):
            return
        self._append_table(dataset, table)

    def _append_table(self, dataset: Dataset, table: str,
                      id_counter: list[int] | None = None) -> None:
        """Stream one shard's table into the (already prepared) target with COPY.

        `id_counter` — when given (a one-element mutable [n]) — REPLACES each row's
        `id` with a running global value, so object_detections.id stays unique across
        shards despite each shard's consolidate assigning it 0-based. Only used on
        the union append path; single-shard loads keep the parquet id verbatim."""
        cols = _COPY_COLUMNS[table]
        expected = dataset.table_rows(table)
        n = 0

        def rows():
            nonlocal n
            for batch in dataset.iter_table(table, columns=list(cols)):
                for row in batch:
                    if id_counter is not None:
                        row = dict(row)
                        row["id"] = id_counter[0]
                        id_counter[0] += 1
                    yield _copy_line(tuple(row[c] for c in cols))
                n += len(batch)
                print(f"[load]   {table} {n:,}/{expected:,}", flush=True)

        self._copy_from((table, cols), rows())

    def _prepare_table(self, table: str, expected: int, force: bool) -> bool:
        """True if `table` should be loaded now.

        Three cases, and NONE of them destroys data implicitly:
          * count already == expected  -> skip (this is what makes load_data
            idempotent, and what a re-run after a preemption relies on)
          * empty                      -> load
          * non-empty but wrong count  -> raise
        The last case is deliberately not an automatic TRUNCATE. A row count can
        be "wrong" because the shard changed, because a previous load died
        halfway, or because this DB holds a *different* shard — and silently
        wiping millions of loaded rows (plus an hours-long HNSW rebuild) on that
        guess is not a call this code should make. `force=True` opts in."""
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {table};")
            have = _scalar(cur)
            if not force and have == expected and expected > 0:
                print(f"[load] skip {table}: already {have:,} rows", flush=True)
                return False
            if have and not force:
                raise RuntimeError(
                    f"{table} holds {have:,} rows but the shard has {expected:,}. "
                    "Refusing to overwrite: re-run with --force to truncate and "
                    "reload this system, or point --dataset at the matching shard.")
            if have:
                print(f"[load] truncating {table} ({have:,} rows) — force", flush=True)
                cur.execute(f"TRUNCATE {table};")
        return True

    def _copy_from(self, target: tuple[str, tuple[str, ...]], lines) -> None:
        table, cols = target
        assert self._conn is not None
        sql = f"COPY {table} ({', '.join(cols)}) FROM STDIN"
        with self._conn.cursor() as cur:
            # psycopg2 accepts any object with read()/readline() returning str
            # (the documented StringIO usage); the stubs only admit bytes/TextIO.
            cur.copy_expert(sql, _LineReader(lines))  # type: ignore[arg-type]

    def _connect(self) -> None:
        import psycopg2  # optional dep (`pgvector` extra)

        self._conn = psycopg2.connect(self.dsn) if self.dsn else psycopg2.connect()
        # Pin the session to UTF-8. Without this the client encoding follows the
        # server's (SQL_ASCII on this cluster's pg container), and copy_expert then
        # encodes the str COPY stream as ASCII — a single accented char in a V3C
        # title/OCR span (é, …) aborts the whole COPY. UTF-8 is safe either way:
        # transcoded on a UTF8 server, passed through untouched on SQL_ASCII.
        self._conn.set_client_encoding("UTF8")
        self._conn.autocommit = True

    def setup(self) -> None:
        self._connect()
        assert self._conn is not None
        with self._conn.cursor() as cur:
            # HNSW search-time knobs; set per session.
            cur.execute("SET hnsw.ef_search = %s;", (self.ef_search,))
            cur.execute("SET hnsw.iterative_scan = %s;", (self.iterative_scan,))
            # Sanity: the vector index we rely on must exist.
            cur.execute("SELECT to_regclass('public.keyframes_embedding_hnsw_idx');")
            row = cur.fetchone()
            if row is None or row[0] is None:
                raise RuntimeError(
                    "keyframes HNSW index missing — this system has not been loaded. "
                    "Run `python scripts/load_dataset.py --dataset <path>` (setup() "
                    "deliberately will not ingest: a run that silently loads is a run "
                    "whose timings mean nothing).")
            # Pin the statistics state (fairness invariant): refresh planner stats on
            # every filter relation so plan choice + recall are reproducible. Skip any
            # table absent from this deployment (e.g. before a schema is fully loaded).
            for table in _ANALYZE_TABLES:
                if _table_exists(cur, table):
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

    def set_session_knobs(
        self,
        ef_search: int | None = None,
        iterative_scan: str | None = None,
        enable_seqscan: bool | None = None,
    ) -> None:
        """Diagnostic — NOT part of the fair run contract. Re-apply search-time
        session GUCs on the LIVE connection without a fresh setup()/ANALYZE, so the
        cutover sweep (frame.core.sweep) can walk the ef_search × iterative_scan grid
        on ONE connection and pay the pinned ANALYZE state only once.

        `enable_seqscan=False` is the FORCE-HNSW counterfactual: it pushes filters
        the planner would otherwise run exactly (seqscan) onto the approximate HNSW
        index path, exposing the approximate recall penalty across the FULL
        selectivity range — not just the few broad filters the planner naturally
        routes to the index. (Session-wide, so side-table subplans are affected too;
        we only rely on the keyframes ORDER BY taking the HNSW index — confirm via
        plan_choice, which reflects this session state.) See the vault note
        'FRAME — pgvector planner split'. Updates the instance attrs so explain()/
        plan_choice()'s mode-restore stays consistent."""
        assert self._conn is not None, "call setup() first"
        with self._conn.cursor() as cur:
            if ef_search is not None:
                self.ef_search = ef_search
                cur.execute("SET hnsw.ef_search = %s;", (ef_search,))
            if iterative_scan is not None:
                self.iterative_scan = iterative_scan
                cur.execute("SET hnsw.iterative_scan = %s;", (iterative_scan,))
            if enable_seqscan is not None:
                cur.execute("SET enable_seqscan = %s;", ("on" if enable_seqscan else "off",))

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
            elif f.filter_type in _VIDEO_META_SOURCES:
                # set-overlap against the parent video's curated array; no threshold.
                col = _VIDEO_META_SOURCES[f.filter_type]
                clauses.append(
                    f"EXISTS (SELECT 1 FROM videos v WHERE v.video_id = k.video_id "
                    f"AND v.{col} && %s::text[])"
                )
                params.append(list(_as_list(f.value)))
            else:
                raise ValueError(f"unsupported filter_type: {f.filter_type!r}")
        return " AND ".join(clauses), params


def _as_list(value) -> list:
    return [value] if isinstance(value, str) else list(value)


# ── COPY ... FROM STDIN (TEXT format) encoding, used by load_data ──────────────
# TEXT, not CSV, deliberately: V3C titles and OCR spans contain embedded newlines
# and quotes, which CSV quoting gets wrong often enough to matter (this bit the
# original V3C1 load — see the pgvector HPC guide). TEXT uses tab separators and
# backslash escapes, with no quoting ambiguity at all.

def _pg_escape(s: str) -> str:
    """Escape one TEXT-format field. Backslash MUST be replaced first, or the
    escapes introduced below would themselves get escaped."""
    return (s.replace("\\", "\\\\")
             .replace("\n", "\\n")
             .replace("\r", "\\r")
             .replace("\t", "\\t"))


def _pg_array_literal(values) -> str:
    """Python list -> a pg array literal `{"a","b"}`. Every element is quoted, so
    commas/braces/spaces inside a tag are safe; NULL elements become unquoted NULL
    (the only way to express them)."""
    parts = []
    for v in values:
        if v is None:
            parts.append("NULL")
        else:
            inner = str(v).replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'"{inner}"')
    return "{" + ",".join(parts) + "}"


def _vector_literal(vec) -> str:
    """float32[768] -> pgvector's text input form `[a,b,...]`. 8 decimals matches
    what search() sends, so stored and queried vectors round-trip identically."""
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


def _copy_field(v) -> str:
    if v is None:
        return "\\N"                       # the TEXT-format NULL marker
    if isinstance(v, (list, tuple)):
        return _pg_escape(_pg_array_literal(v))
    if isinstance(v, bool):
        return "t" if v else "f"
    if isinstance(v, str):
        return _pg_escape(v)
    return _pg_escape(str(v))


def _copy_line(row) -> str:
    return "\t".join(_copy_field(v) for v in row) + "\n"


class _LineReader:
    """File-like adapter so psycopg2's copy_expert can pull from a generator of
    lines. copy_expert calls .read(size); we buffer just enough to answer each
    call, which keeps a multi-GB COPY at a few hundred KB of RAM."""

    def __init__(self, lines):
        self._lines = iter(lines)
        self._buf = ""

    def read(self, size: int = -1) -> str:
        if size is None or size < 0:
            return self._buf + "".join(self._lines)
        while len(self._buf) < size:
            try:
                self._buf += next(self._lines)
            except StopIteration:
                break
        chunk, self._buf = self._buf[:size], self._buf[size:]
        return chunk

    def readline(self, size: int = -1) -> str:
        if not self._buf:
            try:
                self._buf = next(self._lines)
            except StopIteration:
                return ""
        idx = self._buf.find("\n")
        cut = len(self._buf) if idx < 0 else idx + 1
        if 0 <= size < cut:
            cut = size
        chunk, self._buf = self._buf[:cut], self._buf[cut:]
        return chunk


def _table_exists(cur, table: str) -> bool:
    cur.execute("SELECT to_regclass(%s);", (f"public.{table}",))
    return _scalar(cur) is not None


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

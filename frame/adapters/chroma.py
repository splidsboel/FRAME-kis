"""
chroma adapter — the second system under test (single-collection, DENORMALISED).

Contrast with pgvector (the first adapter): pgvector holds the NORMALISED V3C
schema and answers a filtered query with a native JOIN/EXISTS across side tables.
Chroma has no joins and no cross-record predicates, so it CANNOT do that. Its only
move is the multi-table workaround this suite exists to measure: at ingest,
load_data() DENORMALISES every side relation onto the keyframe's own record — the
passing scene/object labels and the parent video's categories/tags become
array-valued metadata on the one collection, and the keyframe's OCR text becomes
that record's document. A filtered search is then a single-collection query with a
metadata `where` (+ a `where_document` for OCR), never a join. That asymmetry —
same logical schema, a different physical layout forced by the engine — IS the
research point, so it is kept visible here rather than hidden in a shared loader.

FAIRNESS: this is a SEPARATE predicate-translation path from the oracle's exact GT
and from pgvector's SQL (the gap between exact and approximate is a result we
measure, so the translators must not share code). But the PASSING UNIVERSE must be
identical across systems: confidence thresholds are PINNED at ingest (scene 0.10,
object 0.30 — the same values pgvector filters on), so the same keyframe passes
filter F in Chroma as in pgvector. Pattern-match (OCR) has no threshold by design;
Chroma's `$contains` is case-SENSITIVE, so both the stored document and the query
substring are lower-cased to reproduce pgvector's case-insensitive `lower(text)
LIKE`.

There is deliberately NO ANALYZE analog here: Chroma has no cost-based planner and
no exact↔approximate plan choice — every filtered query is the same HNSW search
with a metadata filter — so the statistics-freshness fairness invariant pgvector
needs simply does not arise. That absence is itself a contrast worth recording.

Requires the `chroma` extra (chromadb >= 1.5.0 — array-valued metadata and the
`$contains`/`$not_contains` operators over arrays landed in 1.5.0). Embedded client
(PersistentClient over a local directory): no server, no container. The persist
directory is FRAME_CHROMA_PATH (see chroma_load.sh); chromadb is imported lazily so
the harness core and the pure translation logic need no chromadb installed.
"""

from __future__ import annotations

import os
from typing import Iterator, Sequence

import numpy as np

from ..core.adapter import VectorDBAdapter
from ..core.dataset import Dataset
from ..core.schema import Predicate

# Pinned fairness thresholds — MUST match pgvector's (see the query-set repo
# "Chosen confidence thresholds"). Applied at INGEST here (a label only enters a
# record's metadata array if it passes), so Chroma's passing universe is identical
# to pgvector's, which applies the same floor in its EXISTS subquery at query time.
SCENE_THRESHOLD = 0.10
OBJECT_THRESHOLD = 0.30

# The single collection every keyframe record lives in.
COLLECTION = "keyframes"

# Metadata ARRAY field each label/video-meta filter is denormalised into, and (for
# the label filters) the source side table + confidence floor applied at ingest.
# Short keys keep the per-record metadata compact across millions of records.
#   filter_type -> (metadata field, source table, threshold)
_LABEL_INGEST = {
    "scene":  ("scene", "scene_labels", SCENE_THRESHOLD),
    "object": ("object", "object_detections", OBJECT_THRESHOLD),
}
# Video-level facets: denormalised from the parent video onto every one of its
# keyframes (Chroma cannot reach the video row at query time). No confidence floor.
#   filter_type -> (metadata field, videos column)
_VIDEO_META_INGEST = {
    "video-category": ("vcat", "categories"),
    "video-tag":      ("vtag", "tags"),
}

# Search-side: filter_type -> metadata array field the `$contains` is tested on.
# pattern-match is absent — it filters the record's DOCUMENT via where_document.
_FIELD_BY_FILTER = {
    "scene": "scene", "object": "object",
    "video-category": "vcat", "video-tag": "vtag",
}

# HNSW knobs, PINNED to pgvector's exactly (V3C Schema.md: m=16, ef_construction=64,
# cosine) so the two systems build comparable graphs over the same L2-normalised
# SigLIP vectors. ef_search is the search-beam width; pgvector pins hnsw.ef_search
# to k (=1000) with no recall headroom, so Chroma does the same. hnswlib requires
# ef_search >= n_results, so a k=1000 run needs ef_search >= 1000.
HNSW_SPACE = "cosine"
HNSW_MAX_NEIGHBORS = 16
HNSW_EF_CONSTRUCTION = 64

# chromadb caps a single add() batch (get_max_batch_size, ~5461). Stay under it and
# keep each commit's memory bounded; the ingest streams so this is the only buffer.
_ADD_BATCH = 4000


class ChromaAdapter(VectorDBAdapter):
    name = "chroma"

    def __init__(self, path: str | None = None, ef_search: int = 1000):
        # path=None -> FRAME_CHROMA_PATH env, else ~/chroma (mirrors pgvector's
        # PGDATA living outside the repo). The persist dir holds one collection.
        self.path = path or os.environ.get(
            "FRAME_CHROMA_PATH", os.path.expanduser("~/chroma"))
        self.ef_search = ef_search
        self._client = None
        self._collection = None

    # ── one-time ingest (Tier 2 -> Chroma's denormalised collection) ──
    def load_data(self, dataset: Dataset, force: bool = False) -> None:
        """Denormalise one canonical shard into the single Chroma collection. The
        ABC entry point — a thin wrapper over load_datasets()."""
        self.load_datasets([dataset], force=force)

    def load_datasets(self, datasets: Sequence[Dataset], force: bool = False) -> None:
        """Ingest one OR MORE canonical shards into ONE Chroma collection — the union
        corpus (v3c1+2+3), one HNSW index over every shard's vectors.

        Unlike pgvector there is no drop-indexes/rebuild dance: Chroma's hnswlib is
        an INCREMENTAL index, so every shard's denormalised records are simply
        add()-ed into the same collection. Each shard is pre-joined in turn (its side
        relations grouped per keyframe, then the video facets folded in) and streamed
        out in batches, so peak memory is one shard's grouping maps, not the corpus.

        Idempotent by count (mirrors pgvector): if the collection already holds
        exactly the summed vector total it is a no-op; a partial load raises unless
        `force`, which deletes the collection and rebuilds it from scratch. A
        cross-shard append into a live collection is correct (ids are corpus-wide
        unique), so a clean resume path is intentionally not attempted — a mismatch
        means --force."""
        datasets = list(datasets)
        if not datasets:
            raise ValueError("load_datasets: no datasets given")
        for ds in datasets:
            ds.validate()

        self._connect()
        expected = sum(ds.embedding_count() for ds in datasets)
        col = self._get_collection()

        if col is not None and not force:
            have = col.count()
            if have == expected and expected > 0:
                names = ", ".join(d.name for d in datasets)
                print(f"[load] chroma already loaded: {have:,} records across "
                      f"{len(datasets)} shard(s) [{names}]; skipping (use force to "
                      "rebuild)", flush=True)
                return
            if have:
                raise RuntimeError(
                    f"collection {COLLECTION!r} holds {have:,} records but the load "
                    f"expects {expected:,}. A partial or different load is present. "
                    "Re-run with force=True (scripts/load_dataset.py --force) to "
                    "delete and rebuild.")

        # Fresh build: drop any empty/partial collection and create it pinned.
        assert self._client is not None
        if col is not None:
            self._client.delete_collection(COLLECTION)
        col = self._create_collection()
        self._collection = col

        names = ", ".join(d.name for d in datasets)
        print(f"[load] {names} -> chroma @ {self.path} ({expected:,} records)",
              flush=True)
        for ds in datasets:
            self._load_shard(col, ds)
        print(f"[load] done: {col.count():,} records", flush=True)

    def _load_shard(self, col, ds: Dataset) -> None:
        """Pre-join ONE shard and stream its denormalised records into the collection.

        The side relations are grouped per keyframe first (labels above threshold,
        OCR text, video facets); then the embeddings h5 drives the record loop, so a
        vector with no keyframe metadata is skipped exactly as pgvector skips it (it
        could never be a valid k-NN answer). Records are add()-ed in bounded batches."""
        print(f"[load] shard {ds.name}: grouping side relations ...", flush=True)
        # One grouped map per label field (scene/object), driven by the pinned
        # ingest table + threshold so the passing universe matches pgvector's.
        label_maps = {field: _group_labels(ds, table, thr)
                      for field, table, thr in _LABEL_INGEST.values()}
        ocr = _group_ocr(ds)
        videos = _video_meta(ds)
        kf_video = _keyframe_video(ds)
        counts = " / ".join(f"{len(m):,} with {field}"
                            for field, m in label_maps.items())
        print(f"[load] shard {ds.name}: {len(kf_video):,} keyframes, "
              f"{counts} / {len(ocr):,} with OCR", flush=True)

        ids: list[str] = []
        embs: list[np.ndarray] = []
        metas: list[dict | None] = []
        docs: list[str] = []
        written = missing = 0

        def flush() -> None:
            nonlocal written
            if not ids:
                return
            col.add(ids=list(ids), embeddings=list(embs),
                    metadatas=list(metas), documents=list(docs))
            written += len(ids)
            print(f"[load]   {ds.name} {written:,} records", flush=True)
            ids.clear(); embs.clear(); metas.clear(); docs.clear()

        for kid, vec in _iter_vectors(ds):
            video_id = kf_video.get(kid)
            if video_id is None:
                missing += 1
                continue
            ids.append(kid)
            embs.append(vec)
            metas.append(_record_metadata(kid, video_id, label_maps, videos))
            docs.append(ocr.get(kid, ""))
            if len(ids) >= _ADD_BATCH:
                flush()
        flush()

        if missing:
            print(f"[load] WARNING: {missing:,} vectors had no keyframe metadata "
                  "(shard metadata and embeddings disagree)", flush=True)

    def _create_collection(self):
        assert self._client is not None
        return self._client.create_collection(
            name=COLLECTION,
            embedding_function=None,   # we supply vectors; skip the default (ONNX) EF
            configuration={"hnsw": {
                "space": HNSW_SPACE,
                "ef_construction": HNSW_EF_CONSTRUCTION,
                "max_neighbors": HNSW_MAX_NEIGHBORS,
                "ef_search": self.ef_search,
            }},
        )

    def _get_collection(self):
        """The collection handle, or None if it does not exist yet."""
        assert self._client is not None
        try:
            return self._client.get_collection(COLLECTION, embedding_function=None)
        except Exception:
            return None

    def _connect(self) -> None:
        import chromadb  # optional dep (`chroma` extra)
        if self._client is None:
            os.makedirs(self.path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.path)

    def setup(self) -> None:
        """PER-RUN: open the persisted collection, pin the search beam, verify it is
        loaded. Ingest does NOT happen here (a run that silently loads is a run whose
        timings mean nothing) — raise if the collection is missing/empty."""
        self._connect()
        col = self._get_collection()
        if col is None or col.count() == 0:
            raise RuntimeError(
                f"chroma collection {COLLECTION!r} missing or empty at {self.path} — "
                "this system has not been loaded. Run `python scripts/load_dataset.py "
                "--system chroma --dataset <path>` first (setup() deliberately will "
                "not ingest).")
        # Pin the search-beam width for this run (fairness: same beam as pgvector's
        # hnsw.ef_search). Search-time only; persisted config carries it too.
        col.modify(configuration={"hnsw": {"ef_search": self.ef_search}})
        self._collection = col

    def teardown(self) -> None:
        self._collection = None
        self._client = None

    def search(
        self,
        query_vector: np.ndarray,
        filters: Sequence[Predicate],
        k: int,
    ) -> list[str]:
        assert self._collection is not None, "call setup() first"
        where = _build_where(filters)
        where_document = _build_where_document(filters)
        res = self._collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=where,
            where_document=where_document,
            include=[],            # ids are always returned; skip docs/metadata/distances
        )
        return list(res["ids"][0])

    # ── diagnostic (parity with the load_dataset.py verify step) ──
    def corpus_size(self) -> int:
        assert self._collection is not None, "call setup() first"
        return self._collection.count()


# ─────────────────────────────────────────────────────────────────────────────
# Predicate translation (system-under-test side) — PURE functions, no chromadb.
# A metadata `where` for the label/video-meta filters + a `where_document` for the
# OCR pattern-match. AND across predicates; OR (any-of) within one predicate's value
# list. Kept module-level and dependency-free so the translation — the one place a
# filter bug is silent — is unit-testable without a live Chroma (see tests).
# ─────────────────────────────────────────────────────────────────────────────
def _build_where(filters: Sequence[Predicate]) -> dict | None:
    """Metadata `where` for the AND-ed predicate, or None if no metadata filter.
    Each label/video-meta filter becomes an any-of `$contains` over its array field;
    pattern-match is handled by _build_where_document instead."""
    clauses: list[dict] = []
    for f in filters:
        if f.filter_type in _FIELD_BY_FILTER:
            clauses.append(_any_contains(_FIELD_BY_FILTER[f.filter_type],
                                         _as_list(f.value)))
        elif f.filter_type == "pattern-match":
            continue
        else:
            raise ValueError(f"unsupported filter_type: {f.filter_type!r}")
    return _combine_and(clauses)


def _build_where_document(filters: Sequence[Predicate]) -> dict | None:
    """`where_document` for the OCR pattern-match filter(s), or None. `$contains` is
    case-sensitive, so the query substrings are lower-cased to match the lower-cased
    document stored at ingest (reproducing pgvector's case-insensitive `lower(text)
    LIKE ANY`). Multiple values within one filter are OR-ed (LIKE ANY); multiple
    pattern-match predicates are AND-ed."""
    clauses: list[dict] = []
    for f in filters:
        if f.filter_type != "pattern-match":
            continue
        likes = [v.lower() for v in _as_list(f.value) if v]
        conds = [{"$contains": s} for s in likes]
        if conds:
            clauses.append(conds[0] if len(conds) == 1 else {"$or": conds})
    return _combine_and(clauses)


def _any_contains(field: str, values: Sequence[str]) -> dict:
    """`field` array contains ANY of `values` — one `$contains` if a single value,
    else an `$or` over them (the denormalised analog of pgvector's `label = ANY(...)`)."""
    conds = [{field: {"$contains": v}} for v in values]
    return conds[0] if len(conds) == 1 else {"$or": conds}


def _combine_and(clauses: Sequence[dict]) -> dict | None:
    """AND a list of clauses: None if empty, the bare clause if one (Chroma wants a
    single operator at the top level, not a 1-element `$and`), else `$and`."""
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": list(clauses)}


def _as_list(value) -> list:
    return [value] if isinstance(value, str) else list(value)


# ─────────────────────────────────────────────────────────────────────────────
# Ingest-side denormalisation helpers — group each side relation per keyframe so a
# keyframe's whole label set can be written as ONE array on its single record.
# Streamed reads, but the grouping maps are materialised per shard (inherent to
# denormalisation: every label of a keyframe must be gathered before its record is
# written). Empty sets are omitted, never stored as `[]` (Chroma rejects empty
# metadata arrays, and a `$contains` over a missing field correctly does not match).
# ─────────────────────────────────────────────────────────────────────────────
def _group_labels(ds: Dataset, table: str, threshold: float) -> dict[str, list[str]]:
    """keyframe_id -> sorted DISTINCT labels passing `threshold`. object_detections
    has many rows per keyframe (one per detection), so labels are de-duplicated."""
    out: dict[str, set] = {}
    if not ds.has_table(table):
        return {}
    for batch in ds.iter_table(table, columns=["keyframe_id", "label", "confidence"]):
        for row in batch:
            conf = row["confidence"]
            if conf is None or conf < threshold:
                continue
            out.setdefault(row["keyframe_id"], set()).add(row["label"])
    return {k: sorted(v) for k, v in out.items()}


def _group_ocr(ds: Dataset) -> dict[str, str]:
    """keyframe_id -> LOWER-cased concatenation of its OCR spans (span order), joined
    by newline. No threshold — pgvector's pattern-match applies none. Stored as the
    record's document so where_document `$contains` can substring-match it."""
    spans: dict[str, list[tuple[int, str]]] = {}
    if not ds.has_table("keyframe_ocr"):
        return {}
    for batch in ds.iter_table("keyframe_ocr", columns=["keyframe_id", "span_index", "text"]):
        for row in batch:
            text = row["text"]
            if not text:
                continue
            spans.setdefault(row["keyframe_id"], []).append((row["span_index"], text))
    return {k: "\n".join(t for _, t in sorted(v)).lower() for k, v in spans.items()}


def _video_meta(ds: Dataset) -> dict[str, dict[str, list[str]]]:
    """video_id -> {vcat: [...], vtag: [...]} for the non-empty facets, ready to fold
    onto every one of the video's keyframes."""
    out: dict[str, dict[str, list[str]]] = {}
    if not ds.has_table("videos"):
        return {}
    cols = ["video_id"] + [c for _, c in _VIDEO_META_INGEST.values()]
    for batch in ds.iter_table("videos", columns=cols):
        for row in batch:
            facets: dict[str, list[str]] = {}
            for field, col in _VIDEO_META_INGEST.values():
                vals = row.get(col)
                if vals:
                    facets[field] = list(vals)
            if facets:
                out[row["video_id"]] = facets
    return out


def _keyframe_video(ds: Dataset) -> dict[str, str]:
    """keyframe_id -> video_id (to fold the video facets onto each keyframe and to
    mark which vectors have a metadata row at all)."""
    out: dict[str, str] = {}
    for batch in ds.iter_table("keyframes", columns=["keyframe_id", "video_id"]):
        for row in batch:
            out[row["keyframe_id"]] = row["video_id"]
    return out


def _record_metadata(
    kid: str,
    video_id: str,
    label_maps: dict[str, dict[str, list[str]]],
    videos: dict[str, dict[str, list[str]]],
) -> dict | None:
    """Assemble one keyframe's denormalised metadata dict, or None when it has no
    filterable metadata at all (Chroma rejects an empty `{}`; None is accepted, and
    the record is still reachable by pure vector search)."""
    md: dict[str, list[str]] = {}
    for field, grouped in label_maps.items():
        labels = grouped.get(kid)
        if labels:
            md[field] = labels
    md.update(videos.get(video_id, {}))
    return md or None


def _iter_vectors(ds: Dataset) -> Iterator[tuple[str, np.ndarray]]:
    """Stream (keyframe_id, vector) pairs from the embeddings h5 — the big side, kept
    streamed. Vectors are paired to metadata by id, never by position."""
    for ids, vecs in ds.iter_embeddings():
        for kid, vec in zip(ids, vecs):
            yield kid, vec

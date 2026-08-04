"""
Dataset — a read-only handle on one **Tier-2 canonical shard** (the
backend-neutral corpus produced by scripts/export_v3c.py or the scripts/prep_*
pipeline). See the vault: [[Data pipeline and adapter load refactor]].

This is the *logical* data contract of the two the suite rests on. It is the
input to `VectorDBAdapter.load_data()`, which materialises it into whatever
*physical* layout its system needs — pgvector into the normalised V3C tables,
Chroma/Milvus into one denormalised collection. That asymmetry is the research
point, so it lives in the adapters; this class only hands out neutral rows.

Layout of a canonical folder (all files optional except keyframes + embeddings):

    <path>/videos.parquet  shots.parquet  keyframes.parquet  keyframe_ocr.parquet
           scene_labels.parquet  object_detections.parquet  object_detection_done.parquet
           keyframes_embeddings.h5     /embedding float32[N,768], /keyframe_id str[N]
           MANIFEST.json               format contract + row counts

Reads are **streamed** (parquet row batches, h5 slices) so a 1.7M-keyframe shard
loads in bounded memory. pyarrow/h5py are imported lazily: the harness core
depends only on numpy, and a machine that merely *runs* the benchmark never needs
the ingest deps.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

# Load order is FK-safe: parents before children. object_detection_done is
# bookkeeping from the V3C1 detection pass; the prep pipeline does not emit it,
# hence every table here is treated as optional.
TABLES = (
    "videos",
    "shots",
    "keyframes",
    "keyframe_ocr",
    "scene_labels",
    "object_detections",
    "object_detection_done",
)

EMBEDDINGS_FILE = "keyframes_embeddings.h5"
MANIFEST_FILE = "MANIFEST.json"
EMBED_DIM = 768                      # SigLIP google/siglip-base-patch16-224, L2-normalised

DEFAULT_BATCH = 50_000


class Dataset:
    """One canonical shard on disk. `Dataset("data/canonical/v3c1")`."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path).expanduser()
        self._manifest: dict | None = None

    # ── identity / metadata ────────────────────────────────────────────────
    @property
    def name(self) -> str:
        """Shard name — the manifest's `dataset`, else the folder name."""
        return str(self.manifest.get("dataset") or self.path.name)

    @property
    def manifest(self) -> dict:
        """MANIFEST.json, or {} when absent (the prep pipeline writes one; a
        hand-assembled folder may not)."""
        if self._manifest is None:
            try:
                loaded = json.loads((self.path / MANIFEST_FILE).read_text())
            except (OSError, ValueError):
                loaded = {}
            self._manifest = loaded if isinstance(loaded, dict) else {}
        return self._manifest

    def expected_rows(self, table: str) -> int | None:
        """Row count the manifest claims for `table`, or None if unrecorded.
        Used by load_data() to decide whether a table is already fully ingested."""
        rows = self.manifest.get("tables", {}).get(table)
        return int(rows) if isinstance(rows, int) else None

    @property
    def expected_embeddings(self) -> int | None:
        emb = self.manifest.get("embeddings")
        return int(emb["count"]) if isinstance(emb, dict) and "count" in emb else None

    # ── table access ───────────────────────────────────────────────────────
    def table_path(self, table: str) -> Path:
        return self.path / f"{table}.parquet"

    def has_table(self, table: str) -> bool:
        return self.table_path(table).is_file()

    def tables(self) -> list[str]:
        """Canonical tables actually present, in FK-safe load order."""
        return [t for t in TABLES if self.has_table(t)]

    def table_rows(self, table: str) -> int:
        """Exact row count from the parquet footer (cheap — no data read)."""
        import pyarrow.parquet as pq
        return pq.read_metadata(self.table_path(table)).num_rows

    def iter_table(self, table: str, columns: list[str] | None = None,
                   batch_size: int = DEFAULT_BATCH) -> Iterator[list[dict]]:
        """Stream one table as batches of row dicts. Column order follows the
        parquet schema unless `columns` narrows it."""
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(self.table_path(table))
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            yield batch.to_pylist()

    # ── embeddings ─────────────────────────────────────────────────────────
    @property
    def embeddings_path(self) -> Path:
        return self.path / EMBEDDINGS_FILE

    def has_embeddings(self) -> bool:
        return self.embeddings_path.is_file()

    def embedding_count(self) -> int:
        import h5py
        with h5py.File(self.embeddings_path, "r") as h5:
            return int(h5["keyframe_id"].shape[0])

    def iter_embeddings(self, batch_size: int = DEFAULT_BATCH):
        """Stream (ids, vectors) slices — ids list[str], vectors float32[m, 768],
        aligned row-wise. Order is whatever the producer wrote (v3c1: ORDER BY
        keyframe_id; prep: video-sorted); callers must not assume either, so
        pair vectors with metadata by id, never by position."""
        import h5py
        import numpy as np
        with h5py.File(self.embeddings_path, "r") as h5:
            ids_ds, emb_ds = h5["keyframe_id"], h5["embedding"]
            n = int(ids_ds.shape[0])
            dim = int(emb_ds.shape[1])
            if dim != EMBED_DIM:
                raise ValueError(f"{self.embeddings_path}: dim {dim} != {EMBED_DIM}")
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                raw = ids_ds[start:stop]
                ids = [b.decode() if isinstance(b, bytes) else str(b) for b in raw]
                yield ids, np.asarray(emb_ds[start:stop], dtype=np.float32)

    # ── validation ─────────────────────────────────────────────────────────
    def validate(self) -> None:
        """Raise if this folder can't serve as a benchmark corpus. Cheap: file
        existence + parquet footers, no row scans."""
        if not self.path.is_dir():
            raise FileNotFoundError(f"canonical dataset not found: {self.path}")
        missing = [t for t in ("videos", "keyframes") if not self.has_table(t)]
        if missing:
            raise FileNotFoundError(
                f"{self.path}: missing required table(s) {missing} — "
                "run scripts/prep_consolidate.sh (or export_v3c.sh) first")
        if not self.has_embeddings():
            raise FileNotFoundError(
                f"{self.path}: missing {EMBEDDINGS_FILE} — the shard has no vectors")

    def describe(self) -> str:
        parts = [f"dataset {self.name} @ {self.path}"]
        for t in self.tables():
            parts.append(f"  {t:<24} {self.table_rows(t):>12,} rows")
        if self.has_embeddings():
            parts.append(f"  {'embeddings':<24} {self.embedding_count():>12,} vectors")
        return "\n".join(parts)

    def __repr__(self) -> str:
        return f"Dataset({str(self.path)!r})"

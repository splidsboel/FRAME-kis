"""
prep_consolidate.py — merge the per-video staging from the model passes into the
canonical single-file dataset, byte-compatible with the v3c1 export:

    _staging/detect/*.parquet  -> object_detections.parquet  (+ global int64 id)
                               -> object_detection_done.parquet (all keyframes of
                                  every completed detect video)
    _staging/scenes/*.parquet  -> scene_labels.parquet
    _staging/ocr/*.parquet     -> keyframe_ocr.parquet
    _staging/embed/*.npz       -> keyframes_embeddings.h5  (/embedding, /keyframe_id)
    videos/shots/keyframes.parquet are already written by prep_metadata.py.

Streams per video so memory stays bounded. Global order = sorted video_id then
keyframe_id (== ORDER BY keyframe_id), matching v3c1. Writes MANIFEST.json.

    python3 scripts/prep_consolidate.py --dataset v3c2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import sys
# scripts/ is one level below the repo root; put the root on sys.path for `import frame`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.prep.common import EMBED_DIM, SCHEMAS


def _staged(out_dir: str, pass_name: str) -> list:
    return sorted(glob.glob(os.path.join(out_dir, "_staging", pass_name, "*.parquet")))


def merge_parquet(out_dir: str, pass_name: str, table_name: str, add_id: bool) -> int:
    """Concatenate staged per-video parquets into the canonical <table_name>.parquet,
    streaming. If add_id, prepend a global sequential int64 `id` (object_detections)."""
    files = _staged(out_dir, pass_name)
    schema = SCHEMAS[table_name]
    writer = pq.ParquetWriter(os.path.join(out_dir, f"{table_name}.parquet"),
                              schema, compression="zstd")
    total = 0
    try:
        for f in files:
            t = pq.read_table(f)
            if t.num_rows == 0:
                continue
            if add_id:
                ids = pa.array(range(total, total + t.num_rows), type=pa.int64())
                t = t.add_column(0, pa.field("id", pa.int64()), ids)
            t = t.select([fld.name for fld in schema]).cast(schema)
            writer.write_table(t)
            total += t.num_rows
    finally:
        writer.close()
    print(f"[consolidate] {table_name}: {total:,} rows from {len(files)} videos", flush=True)
    return total


def build_object_detection_done(out_dir: str) -> int:
    """Every keyframe of every completed detect video (incl. zero-detection frames)."""
    done_videos = {Path(f).stem for f in _staged(out_dir, "detect")}
    kf = pq.read_table(os.path.join(out_dir, "keyframes.parquet"),
                       columns=["keyframe_id", "video_id"])
    mask = pa.compute.is_in(kf.column("video_id"), value_set=pa.array(sorted(done_videos)))
    ids = kf.filter(mask).column("keyframe_id")
    schema = SCHEMAS["object_detection_done"]
    pq.write_table(pa.table({"keyframe_id": ids}, schema=schema),
                   os.path.join(out_dir, "object_detection_done.parquet"), compression="zstd")
    print(f"[consolidate] object_detection_done: {len(ids):,} keyframes "
          f"({len(done_videos)} videos)", flush=True)
    return len(ids)


def build_embeddings_h5(out_dir: str) -> int:
    """Concatenate per-video embedding npz into one chunked HDF5, video-sorted."""
    files = sorted(glob.glob(os.path.join(out_dir, "_staging", "embed", "*.npz")))
    path = os.path.join(out_dir, "keyframes_embeddings.h5")
    str_dt = h5py.string_dtype(encoding="utf-8")
    n = 0
    with h5py.File(path, "w") as h5:
        h5.attrs["model"] = "google/siglip-base-patch16-224"
        h5.attrs["dim"] = EMBED_DIM
        h5.attrs["normalised"] = "L2"
        h5.attrs["order"] = "sorted video_id then keyframe_id; embedding[i] <-> keyframe_id[i]"
        emb = h5.create_dataset("embedding", shape=(0, EMBED_DIM), maxshape=(None, EMBED_DIM),
                                dtype="float32", chunks=(min(8192, 8192), EMBED_DIM),
                                compression="gzip", compression_opts=1)
        idset = h5.create_dataset("keyframe_id", shape=(0,), maxshape=(None,), dtype=str_dt)
        for f in files:
            with np.load(f, allow_pickle=True) as z:
                vecs = z["embedding"].astype(np.float32, copy=False)
                ids = z["keyframe_id"]
            m = vecs.shape[0]
            emb.resize(n + m, axis=0); emb[n:n + m] = vecs
            idset.resize(n + m, axis=0); idset[n:n + m] = ids.astype(str)
            n += m
    print(f"[consolidate] embeddings: {n:,} vectors from {len(files)} videos", flush=True)
    return n


def parquet_rows(out_dir: str, name: str) -> int:
    return pq.read_metadata(os.path.join(out_dir, f"{name}.parquet")).num_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="v3c2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard-root", default=None,
                    help="optional: report staged vs total video coverage per pass")
    args = ap.parse_args()

    import pyarrow.compute  # noqa: F401  (pa.compute used in build_object_detection_done)
    out_dir = args.out or os.path.join("data", "canonical", args.dataset)
    print(f"[consolidate] {out_dir}", flush=True)

    manifest = {
        "dataset": args.dataset,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "frame.prep model passes over the extracted shard (files, not pg)",
        "embedding_model": "google/siglip-base-patch16-224",
        "embedding_dim": EMBED_DIM,
        "embedding_normalised": "L2",
        "tables": {},
        "excluded": {
            "keyframes_flat": "pgvector-specific denormalisation (adapter load_data materialises this)",
            "query_embeddings": "query-side; part of the queryset artifact, not the corpus",
        },
    }

    # metadata tables already written by prep_metadata.py
    for name in ("videos", "shots", "keyframes"):
        p = os.path.join(out_dir, f"{name}.parquet")
        manifest["tables"][name] = parquet_rows(out_dir, name) if os.path.exists(p) else None

    manifest["tables"]["object_detections"] = merge_parquet(out_dir, "detect", "object_detections", add_id=True)
    manifest["tables"]["object_detection_done"] = build_object_detection_done(out_dir)
    manifest["tables"]["scene_labels"] = merge_parquet(out_dir, "scenes", "scene_labels", add_id=False)
    manifest["tables"]["keyframe_ocr"] = merge_parquet(out_dir, "ocr", "keyframe_ocr", add_id=False)

    n = build_embeddings_h5(out_dir)
    manifest["embeddings"] = {
        "count": n, "dim": EMBED_DIM, "dtype": "float32",
        "file": "keyframes_embeddings.h5",
        "datasets": {"vectors": "/embedding", "ids": "/keyframe_id"},
        "order": "sorted video_id then keyframe_id; embedding[i] <-> keyframe_id[i]",
    }

    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("[consolidate] wrote MANIFEST.json:", json.dumps(manifest["tables"]), flush=True)


if __name__ == "__main__":
    main()

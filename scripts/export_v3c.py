"""
Export the loaded V3C pgvector DB into FRAME's backend-neutral **canonical
dataset** (Tier 2 in [[Data pipeline and adapter load refactor]]).

This is step 1 of that plan: the processed V3C1 data currently lives ONLY as
rows inside the pgvector DB. Dump it to portable files so (a) a second adapter
(Milvus/Chroma) has a neutral source to load from, and (b) the same artifact
serves the oracle/query-set and Omar's public data export.

Output layout (--out, default data/canonical/<dataset>/):
    videos.parquet  shots.parquet  keyframes.parquet  keyframe_ocr.parquet
    object_detections.parquet  object_detection_done.parquet  scene_labels.parquet
    keyframes_embeddings.h5        /embedding float32 [N,768], /keyframe_id str [N]
    MANIFEST.json                  format contract + row counts

Deliberately EXCLUDED (physical-layout / query-side, not neutral corpus):
  - keyframes_flat : pgvector-specific denormalisation — that materialisation is
    exactly what each adapter's load_data() does, so it belongs in the adapter.
  - query_embeddings : query-side; part of the queryset artifact, not the corpus.

Metadata -> Parquet (typed, columnar; pg/Milvus/Chroma/DuckDB all ingest it).
Embeddings -> HDF5 (h5py; chunked, streamed, bounded RAM for 1.08M vectors).

Runs on the HPC host inside the `embeddings` conda env with an in-job postgres
(see export_v3c.sh). Connects via libpq PG* env vars (PGHOST = unix socket dir).
Needs pyarrow + h5py (pip-installed by the wrapper) + numpy + psycopg2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import psycopg2

try:
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pyarrow/h5py missing — the wrapper (export_v3c.sh) pip-installs them "
             "into the embeddings env; run this via sbatch export_v3c.sh, not bare.")

EMBED_DIM = 768          # SigLIP google/siglip-base-patch16-224, L2-normalised
FETCH = 50_000           # server-side cursor batch size

# ── Table export specs ──────────────────────────────────────────────────────
# Explicit arrow schema per table (the format contract; mirrors V3C Schema.md).
# pg type -> arrow: TEXT=string, INTEGER=int32, BIGINT=int64, DOUBLE=float64,
# REAL=float32, TEXT[]=list<string>. Column order = SELECT order = parquet order.
S = pa.string(); I32 = pa.int32(); I64 = pa.int64(); F64 = pa.float64(); F32 = pa.float32()
LSTR = pa.list_(pa.string())

TABLES = {
    "videos": pa.schema([
        ("video_id", S), ("vimeo_id", S), ("title", S), ("duration_s", F64),
        ("width", I32), ("height", I32), ("channel", S), ("upload_date", S),
        ("license", S), ("tags", LSTR), ("categories", LSTR),
    ]),
    "shots": pa.schema([
        ("shot_id", S), ("video_id", S), ("shot_index", I32), ("start_frame", I32),
        ("end_frame", I32), ("start_time_s", F64), ("end_time_s", F64),
    ]),
    # keyframes: metadata only — the vector goes to the .npy sidecar.
    "keyframes": pa.schema([
        ("keyframe_id", S), ("shot_id", S), ("video_id", S), ("frame_number", I32),
    ]),
    "keyframe_ocr": pa.schema([
        ("keyframe_id", S), ("span_index", I32), ("text", S), ("confidence", F32),
    ]),
    "object_detection_done": pa.schema([("keyframe_id", S)]),
    "object_detections": pa.schema([
        ("id", I64), ("keyframe_id", S), ("label", S), ("confidence", F32),
        ("x1", F32), ("y1", F32), ("x2", F32), ("y2", F32),
    ]),
    "scene_labels": pa.schema([
        ("keyframe_id", S), ("label", S), ("confidence", F64),
    ]),
}


def regclass_exists(cur, table: str) -> bool:
    cur.execute("SELECT to_regclass(%s);", (f"public.{table}",))
    return cur.fetchone()[0] is not None


def count(cur, table: str) -> int:
    cur.execute(f"SELECT count(*) FROM {table};")
    return cur.fetchone()[0]


def export_table(conn, table: str, schema: pa.Schema, out_dir: str) -> int:
    """Stream one table to parquet via a server-side cursor, bounded memory."""
    cols = [f.name for f in schema]
    path = os.path.join(out_dir, f"{table}.parquet")
    writer = pq.ParquetWriter(path, schema, compression="zstd")
    n = 0
    cur = conn.cursor(name=f"exp_{table}")   # named => server-side
    cur.itersize = FETCH
    cur.execute(f"SELECT {', '.join(cols)} FROM {table};")
    try:
        while True:
            rows = cur.fetchmany(FETCH)
            if not rows:
                break
            columns = list(zip(*rows))   # row-major -> column-major
            arrays = [pa.array(columns[i], type=schema.field(i).type)
                      for i in range(len(cols))]
            writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
            n += len(rows)
            print(f"    {table}: {n:,} rows", flush=True)
    finally:
        cur.close()
        writer.close()
    return n


def parse_vec(v) -> np.ndarray:
    """pgvector value -> float32 vector. np.ndarray if register_vector is active
    (binary, fast); otherwise the '[a,b,...]' text form."""
    if isinstance(v, np.ndarray):
        return v.astype(np.float32, copy=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # np.fromstring text mode is fast
        return np.fromstring(v[1:-1], dtype=np.float32, sep=",")


def export_embeddings(conn, out_dir: str) -> int:
    """Stream keyframes.embedding to a chunked HDF5 file (bounded RAM), ORDER BY
    keyframe_id so the metadata parquet and vectors align by id. Two aligned
    datasets: /embedding float32 [N,768] and /keyframe_id str [N]."""
    with conn.cursor() as cur:
        n = count(cur, "keyframes")
    path = os.path.join(out_dir, "keyframes_embeddings.h5")
    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(path, "w") as h5:
        h5.attrs["model"] = "google/siglip-base-patch16-224"
        h5.attrs["dim"] = EMBED_DIM
        h5.attrs["normalised"] = "L2"
        h5.attrs["order"] = "ORDER BY keyframe_id; embedding[i] <-> keyframe_id[i]"
        emb = h5.create_dataset(
            "embedding", shape=(n, EMBED_DIM), dtype="float32",
            chunks=(min(n, 8192), EMBED_DIM), compression="gzip", compression_opts=1)
        idset = h5.create_dataset("keyframe_id", shape=(n,), dtype=str_dt)

        cur = conn.cursor(name="exp_emb")
        cur.itersize = FETCH
        cur.execute("SELECT keyframe_id, embedding FROM keyframes ORDER BY keyframe_id;")
        i = 0
        try:
            while True:
                rows = cur.fetchmany(FETCH)
                if not rows:
                    break
                m = len(rows)
                block = np.empty((m, EMBED_DIM), dtype=np.float32)
                idblock = np.empty(m, dtype=object)
                for j, (kid, vec) in enumerate(rows):
                    arr = parse_vec(vec)
                    if arr.shape[0] != EMBED_DIM:
                        raise ValueError(f"{kid}: dim {arr.shape[0]} != {EMBED_DIM}")
                    block[j] = arr
                    idblock[j] = kid
                emb[i:i + m] = block
                idset[i:i + m] = idblock
                i += m
                print(f"    embeddings: {i:,}/{n:,}", flush=True)
        finally:
            cur.close()
    if i != n:
        raise RuntimeError(f"embedding count {i} != keyframes count {n}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="v3c1", help="name for the output subdir + manifest")
    ap.add_argument("--out", default=None, help="output dir (default data/canonical/<dataset>)")
    ap.add_argument("--skip-embeddings", action="store_true", help="metadata only (debug)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join("data", "canonical", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[export] dataset={args.dataset} -> {out_dir}", flush=True)

    conn = psycopg2.connect()   # PG* env vars
    conn.set_client_encoding("UTF8")   # some V3C metadata (titles) is UTF-8, not ASCII
    try:
        try:
            from pgvector.psycopg2 import register_vector
            register_vector(conn)
            print("[export] pgvector binary codec active (fast embedding read)", flush=True)
        except Exception:
            print("[export] pgvector codec unavailable — parsing vectors as text", flush=True)

        manifest = {
            "dataset": args.dataset,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "pgvector V3C DB (see V3C Schema.md)",
            "embedding_model": "google/siglip-base-patch16-224",
            "embedding_dim": EMBED_DIM,
            "embedding_normalised": "L2",
            "tables": {},
            "excluded": {
                "keyframes_flat": "pgvector-specific denormalisation (adapter load_data materialises this)",
                "query_embeddings": "query-side; part of the queryset artifact, not the corpus",
            },
        }

        with conn.cursor() as cur:
            present = {t: regclass_exists(cur, t) for t in TABLES}
        for table, schema in TABLES.items():
            if not present[table]:
                print(f"[export] SKIP {table} (absent from this deployment)", flush=True)
                manifest["tables"][table] = None
                continue
            print(f"[export] {table} ...", flush=True)
            manifest["tables"][table] = export_table(conn, table, schema, out_dir)

        if args.skip_embeddings:
            manifest["embeddings"] = None
        else:
            print("[export] keyframes embeddings ...", flush=True)
            n = export_embeddings(conn, out_dir)
            manifest["embeddings"] = {
                "count": n, "dim": EMBED_DIM, "dtype": "float32",
                "file": "keyframes_embeddings.h5",
                "datasets": {"vectors": "/embedding", "ids": "/keyframe_id"},
                "order": "ORDER BY keyframe_id; embedding[i] <-> keyframe_id[i]",
            }
    finally:
        conn.close()

    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("[export] wrote MANIFEST.json", flush=True)
    print("[export] done:", json.dumps(manifest["tables"]), flush=True)


if __name__ == "__main__":
    main()

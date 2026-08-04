#!/usr/bin/env python3
"""
check_load_roundtrip.py — integration check for an adapter's load_data().

Builds a TINY synthetic Tier-2 canonical shard in a temp dir, loads it into a
scratch database, and verifies the data survived: row counts, the awkward strings
V3C actually contains (embedded newlines/tabs/backslashes/quotes, array elements
with commas), vector ordering, every filter type, and idempotency. Then drops the
scratch database.

Runs in seconds and touches nothing real — the production V3C database is never
opened. This exists because the ingest path is the one place where a bug is
silent: a bad COPY escape shifts columns instead of raising, and you find out
months later that a filter matched the wrong keyframes.

    sbatch load_dataset.sh --check          # via the wrapper (starts postgres)
    python scripts/check_load_roundtrip.py  # with PG* already pointing somewhere

Unit tests cover the encoding functions in isolation (tests/test_pgvector_copy.py);
this covers them against a real postgres, which is where format bugs actually bite.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.adapters import PgvectorAdapter
from frame.core.dataset import EMBED_DIM, Dataset
from frame.core.schema import Predicate

SCRATCH_DB = "frame_loadtest"

# Strings chosen to break a naive COPY: a title spanning lines, OCR with a tab and
# a backslash, a tag containing a comma and a quote (which must stay ONE element).
NASTY_TITLE = 'Line one\nLine two\twith tab \\ backslash "quoted"'
NASTY_OCR = "STOP\tSIGN \\ 50%"
NASTY_TAG = 'news, politics'
NASTY_TAG2 = 'say "what"'


def build_shard(path: str) -> list[str]:
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq

    from frame.prep.common import SCHEMAS

    os.makedirs(path, exist_ok=True)
    kids = [f"90001_{i:05d}" for i in range(4)] + ["90002_00000"]

    tables = {
        "videos": [
            {"video_id": "90001", "vimeo_id": "v1", "title": NASTY_TITLE,
             "duration_s": 12.5, "width": 640, "height": 480, "channel": "chan",
             "upload_date": "2020-01-01", "license": "cc",
             "tags": [NASTY_TAG, NASTY_TAG2], "categories": ["documentary"]},
            {"video_id": "90002", "vimeo_id": None, "title": None,
             "duration_s": None, "width": None, "height": None, "channel": None,
             "upload_date": None, "license": None, "tags": [], "categories": None},
        ],
        "shots": [
            {"shot_id": k, "video_id": k.split("_")[0], "shot_index": i,
             "start_frame": i * 10, "end_frame": i * 10 + 9,
             "start_time_s": float(i), "end_time_s": float(i) + 1.0}
            for i, k in enumerate(kids)
        ],
        "keyframes": [
            {"keyframe_id": k, "shot_id": k, "video_id": k.split("_")[0],
             "frame_number": i} for i, k in enumerate(kids)
        ],
        "keyframe_ocr": [
            {"keyframe_id": kids[0], "span_index": 0, "text": NASTY_OCR, "confidence": 0.9},
            {"keyframe_id": kids[1], "span_index": 0, "text": "plain text", "confidence": 0.5},
        ],
        "scene_labels": [
            {"keyframe_id": kids[0], "label": "night", "confidence": 0.8},
            {"keyframe_id": kids[1], "label": "night", "confidence": 0.05},   # below threshold
            {"keyframe_id": kids[2], "label": "beach", "confidence": 0.7},
        ],
        "object_detections": [
            {"id": 1, "keyframe_id": kids[0], "label": "car", "confidence": 0.9,
             "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
            {"id": 2, "keyframe_id": kids[1], "label": "car", "confidence": 0.1,  # below threshold
             "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
        ],
        "object_detection_done": [{"keyframe_id": k} for k in kids],
    }

    for name, rows in tables.items():
        schema = SCHEMAS[name]
        cols = {f.name: [r.get(f.name) for r in rows] for f in schema}
        pq.write_table(
            pa.Table.from_pydict(cols, schema=schema),
            os.path.join(path, f"{name}.parquet"), compression="zstd")

    # Vectors: kids[0] is the unit e0 basis vector, the rest rotate away from it,
    # so an e0 query must rank them in list order.
    vecs = np.zeros((len(kids), EMBED_DIM), dtype=np.float32)
    for i in range(len(kids)):
        vecs[i, 0] = 1.0 - 0.1 * i
        vecs[i, 1] = np.sqrt(max(0.0, 1.0 - vecs[i, 0] ** 2))
    with h5py.File(os.path.join(path, "keyframes_embeddings.h5"), "w") as h5:
        h5.attrs["dim"] = EMBED_DIM
        h5.create_dataset("embedding", data=vecs)
        h5.create_dataset("keyframe_id", data=np.array(kids, dtype=object),
                          dtype=h5py.string_dtype(encoding="utf-8"))

    with open(os.path.join(path, "MANIFEST.json"), "w") as f:
        json.dump({"dataset": "loadtest",
                   "tables": {k: len(v) for k, v in tables.items()},
                   "embeddings": {"count": len(kids)}}, f)
    return kids


def scratch_db(drop_first: bool = True):
    import psycopg2
    admin = psycopg2.connect()          # PG* env vars -> maintenance db
    admin.autocommit = True
    with admin.cursor() as cur:
        if drop_first:
            cur.execute(f"DROP DATABASE IF EXISTS {SCRATCH_DB};")
        cur.execute(f"CREATE DATABASE {SCRATCH_DB};")
    admin.close()


def drop_scratch_db():
    import psycopg2
    admin = psycopg2.connect()
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {SCRATCH_DB};")
    admin.close()


def check(label: str, got, want) -> bool:
    ok = got == want
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {got!r}" + ("" if ok else f" != {want!r}"))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="don't drop the scratch DB")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="frame_loadtest_")
    failures = 0
    try:
        kids = build_shard(os.path.join(tmp, "loadtest"))
        dataset = Dataset(os.path.join(tmp, "loadtest"))
        print(dataset.describe())

        scratch_db()
        adapter = PgvectorAdapter(dsn=f"dbname={SCRATCH_DB}", ef_search=10)
        adapter.load_data(dataset)

        print("\n[check] contents")
        with adapter:
            conn = adapter._conn
            assert conn is not None
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM keyframes;")
                failures += not check("keyframes rows", cur.fetchone()[0], len(kids))
                cur.execute("SELECT count(*) FROM scene_labels;")
                failures += not check("scene_labels rows", cur.fetchone()[0], 3)
                cur.execute("SELECT count(*) FROM object_detections;")
                failures += not check("object_detections rows", cur.fetchone()[0], 2)

                # The whole point: awkward strings must come back byte-identical.
                cur.execute("SELECT title, tags, categories FROM videos WHERE video_id='90001';")
                title, tags, cats = cur.fetchone()
                failures += not check("title round-trip", title, NASTY_TITLE)
                failures += not check("tags round-trip", tags, [NASTY_TAG, NASTY_TAG2])
                failures += not check("categories round-trip", cats, ["documentary"])

                cur.execute("SELECT vimeo_id, tags, categories FROM videos WHERE video_id='90002';")
                vid, tags2, cats2 = cur.fetchone()
                failures += not check("NULL text preserved", vid, None)
                failures += not check("empty array preserved", tags2, [])
                failures += not check("NULL array preserved", cats2, None)

                cur.execute("SELECT text FROM keyframe_ocr WHERE keyframe_id=%s;", (kids[0],))
                failures += not check("ocr round-trip", cur.fetchone()[0], NASTY_OCR)

            print("\n[check] search + filters")
            q = np.zeros(EMBED_DIM, dtype=np.float32)
            q[0] = 1.0
            failures += not check("unfiltered ranking", adapter.search(q, [], k=3), kids[:3])

            scene = Predicate(filter_type="scene", attribute="scene_label",
                              op="in", value=["night"])
            # kids[1]'s 'night' is below the pinned 0.10 threshold -> excluded.
            failures += not check("scene filter (threshold pinned)",
                                 adapter.search(q, [scene], k=10), [kids[0]])

            obj = Predicate(filter_type="object", attribute="object_label",
                            op="in", value=["car"])
            failures += not check("object filter (threshold pinned)",
                                 adapter.search(q, [obj], k=10), [kids[0]])

            ocr = Predicate(filter_type="pattern-match", attribute="ocr_text",
                            op="contains", value=["stop"])
            failures += not check("pattern-match filter",
                                 adapter.search(q, [ocr], k=10), [kids[0]])

            cat = Predicate(filter_type="video-category", attribute="categories",
                            op="overlaps", value=["documentary"])
            failures += not check("video-category filter",
                                 adapter.search(q, [cat], k=10), kids[:4])

            failures += not check("corpus_size", adapter.corpus_size(), len(kids))
            failures += not check("count_passing(scene)", adapter.count_passing([scene]), 1)

        print("\n[check] idempotency — a second load must change nothing")
        adapter2 = PgvectorAdapter(dsn=f"dbname={SCRATCH_DB}")
        adapter2.load_data(dataset)
        with adapter2:
            failures += not check("corpus_size after reload", adapter2.corpus_size(), len(kids))

        print("\n[check] force reload")
        adapter3 = PgvectorAdapter(dsn=f"dbname={SCRATCH_DB}")
        adapter3.load_data(dataset, force=True)
        with adapter3:
            failures += not check("corpus_size after force", adapter3.corpus_size(), len(kids))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if not args.keep:
            try:
                drop_scratch_db()
            except Exception as e:
                print(f"[warn] could not drop {SCRATCH_DB}: {e}")

    print()
    if failures:
        sys.exit(f"FAILED: {failures} check(s)")
    print("all checks passed")


if __name__ == "__main__":
    main()

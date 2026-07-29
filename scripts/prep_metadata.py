"""
prep_metadata.py — build the non-model canonical tables for an extracted V3C
shard: videos.parquet (from info/<v>.json), shots.parquet (from msb/<v>.tsv) and
keyframes.parquet (by walking keyframes/<v>/*.png). CPU only, single pass.

Mirrors load_v3c1.py's parsing (v3cId zero-padded, shot_id = <video>_<idx:05d>,
tags/categories arrays) and embed_keyframes.py's keyframe convention
(keyframe_id = <video>_<N:05d>, shot_id = keyframe_id, frame_number = N).

    python3 scripts/prep_metadata.py --shard-root ~/datasets/V3C/V3C2 --dataset v3c2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import sys
# scripts/ is one level below the repo root; put the root on sys.path for `import frame`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.prep.common import SCHEMAS, iter_keyframes, video_dirs


def build_videos(shard_root: str, out_dir: str) -> int:
    info = Path(shard_root) / "info"
    rows = []
    for jf in sorted(info.glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
        rows.append({
            "video_id": str(data.get("v3cId", jf.stem)).zfill(5),
            "vimeo_id": _s(data.get("vimeoId")),
            "title": _s(data.get("title")),
            "duration_s": _f(data.get("duration")),
            "width": _i(data.get("width")),
            "height": _i(data.get("height")),
            "channel": _s(data.get("channel")),
            "upload_date": _s(data.get("uploadDate")),
            "license": _s(data.get("license")),
            "tags": data.get("tags") or [],
            "categories": data.get("categories") or [],
        })
    _write("videos", rows, out_dir)
    return len(rows)


def build_shots(shard_root: str, out_dir: str) -> int:
    msb = Path(shard_root) / "msb"
    rows = []
    for tf in sorted(msb.glob("*.tsv")):
        video_id = tf.stem
        with tf.open(encoding="utf-8", errors="replace") as fh:
            for idx, r in enumerate(csv.DictReader(fh, delimiter="\t")):
                rows.append({
                    "shot_id": f"{video_id}_{idx:05d}",
                    "video_id": video_id,
                    "shot_index": idx,
                    "start_frame": _i(r["startframe"]),
                    "end_frame": _i(r["endframe"]),
                    "start_time_s": _f(r["starttime"]),
                    "end_time_s": _f(r["endtime"]),
                })
    _write("shots", rows, out_dir)
    return len(rows)


def build_keyframes(shard_root: str, out_dir: str) -> int:
    rows = []
    for video_id, vdir in video_dirs(shard_root):
        for kid, vid, n, _path in iter_keyframes(vdir):
            rows.append({
                "keyframe_id": kid,
                "shot_id": kid,          # V3C1 convention: keyframes.shot_id == keyframe_id
                "video_id": vid,
                "frame_number": n,
            })
    _write("keyframes", rows, out_dir)
    return len(rows)


def _s(v): return None if v is None else str(v)
def _i(v): return None if v is None or v == "" else int(v)
def _f(v): return None if v is None or v == "" else float(v)


def _write(name: str, rows: list, out_dir: str) -> None:
    schema = SCHEMAS[name]
    cols = {f.name: [r.get(f.name) for r in rows] for f in schema}
    table = pa.table({f.name: pa.array(cols[f.name], type=f.type) for f in schema},
                     schema=schema)
    pq.write_table(table, os.path.join(out_dir, f"{name}.parquet"), compression="zstd")
    print(f"[metadata] {name}: {len(rows):,} rows", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-root", required=True, help="extracted shard dir, e.g. ~/datasets/V3C/V3C2")
    ap.add_argument("--dataset", default="v3c2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    shard_root = os.path.expanduser(args.shard_root)
    out_dir = args.out or os.path.join("data", "canonical", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[metadata] {shard_root} -> {out_dir}", flush=True)

    nv = build_videos(shard_root, out_dir)
    ns = build_shots(shard_root, out_dir)
    nk = build_keyframes(shard_root, out_dir)
    print(f"[metadata] done: videos={nv:,} shots={ns:,} keyframes={nk:,}", flush=True)


if __name__ == "__main__":
    main()

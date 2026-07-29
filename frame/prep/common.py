"""
Shared building blocks for the V3C processing passes (frame.prep).

SINGLE SOURCE OF TRUTH for the canonical Tier-2 format — the pyarrow schemas
here MUST match scripts/export_v3c.py (which produced the validated v3c1 set).
The V3C1 conventions are reproduced deliberately so shards stay comparable:
  * keyframe_id = f"{video_id}_{N:05d}"  (N = the number in shot<video>_<N>_RKF)
  * keyframes.shot_id = keyframe_id       (a self-reference, as embed_keyframes.py did)
  * OWLv2 score threshold 0.10, Places365 top-5, same external vocab/weights.

Input layout — EXTRACTED shards (V3C2/V3C3), unlike V3C1's tarballs:
    <shard_root>/keyframes/<video_id>/shot<video_id>_<N>_RKF.png
    <shard_root>/info/<video_id>.json
    <shard_root>/msb/<video_id>.tsv

Staging: each GPU pass writes ONE file per video into <out>/_staging/<pass>/, so
a preempted job resumes at video granularity (skip videos whose file exists) and
prep_consolidate.py merges the per-video files into the canonical single files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator, Tuple

import pyarrow as pa

# ── Constants: models / external assets (same as the V3C1 passes) ───────────────
EMBED_DIM = 768
SIGLIP_MODEL = "google/siglip-base-patch16-224"
OWLV2_MODEL = "google/owlv2-base-patch16-ensemble"
OWLV2_SCORE_THRESHOLD = 0.10          # store detections at/above this confidence
SCENE_TOP_K = 5                       # Places365 predictions per keyframe

VOCAB_PATH = os.path.expanduser("~/object_vocab.txt")
PLACES_WEIGHTS = os.path.expanduser("~/models/places365/resnet50_places365.pth.tar")
PLACES_LABELS = os.path.expanduser("~/models/places365/categories_places365.txt")
EASYOCR_MODEL_DIR = os.path.expanduser("~/models/easyocr")
OCR_LANGS = ["en"]

# shot<video(5 digits)>_<N>_RKF.png  ->  keyframe_id "<video>_<N:05d>"
FNAME_RE = re.compile(r"^shot(\d{5})_(\d+)_RKF\.(png|jpg)$")

# ── Canonical parquet schemas (mirror scripts/export_v3c.py) ────────────────────
_S = pa.string(); _I32 = pa.int32(); _I64 = pa.int64(); _F64 = pa.float64(); _F32 = pa.float32()
_LSTR = pa.list_(pa.string())

SCHEMAS = {
    "videos": pa.schema([
        ("video_id", _S), ("vimeo_id", _S), ("title", _S), ("duration_s", _F64),
        ("width", _I32), ("height", _I32), ("channel", _S), ("upload_date", _S),
        ("license", _S), ("tags", _LSTR), ("categories", _LSTR),
    ]),
    "shots": pa.schema([
        ("shot_id", _S), ("video_id", _S), ("shot_index", _I32), ("start_frame", _I32),
        ("end_frame", _I32), ("start_time_s", _F64), ("end_time_s", _F64),
    ]),
    "keyframes": pa.schema([
        ("keyframe_id", _S), ("shot_id", _S), ("video_id", _S), ("frame_number", _I32),
    ]),
    "keyframe_ocr": pa.schema([
        ("keyframe_id", _S), ("span_index", _I32), ("text", _S), ("confidence", _F32),
    ]),
    "object_detection_done": pa.schema([("keyframe_id", _S)]),
    "object_detections": pa.schema([
        ("id", _I64), ("keyframe_id", _S), ("label", _S), ("confidence", _F32),
        ("x1", _F32), ("y1", _F32), ("x2", _F32), ("y2", _F32),
    ]),
    "scene_labels": pa.schema([
        ("keyframe_id", _S), ("label", _S), ("confidence", _F64),
    ]),
}


def fname_to_keyframe(fname: str):
    """(keyframe_id, video_id, frame_number) or None for a keyframe file name."""
    m = FNAME_RE.match(fname)
    if not m:
        return None
    video_id, n = m.group(1), int(m.group(2))
    return f"{video_id}_{n:05d}", video_id, n


# ── Video / keyframe iteration over an extracted shard ──────────────────────────

def video_dirs(shard_root: str, shard: int = 0, num_shards: int = 1) -> list:
    """Sorted list of (video_id, Path) keyframe dirs owned by this shard
    (round-robin by index, like ocr_extract.py's array sharding)."""
    root = Path(shard_root) / "keyframes"
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    return [(p.name, p) for i, p in enumerate(dirs) if i % num_shards == shard]


def iter_keyframes(video_dir: Path) -> Iterator[Tuple[str, str, int, Path]]:
    """Yield (keyframe_id, video_id, frame_number, png_path) sorted by keyframe_id."""
    items = []
    for p in video_dir.iterdir():
        parsed = fname_to_keyframe(p.name)
        if parsed is not None:
            kid, vid, n = parsed
            items.append((kid, vid, n, p))
    items.sort(key=lambda t: t[0])
    return iter(items)


# ── Staging (per-video resumable output) ────────────────────────────────────────

def staging_dir(out_dir: str, pass_name: str) -> Path:
    d = Path(out_dir) / "_staging" / pass_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def staged_path(out_dir: str, pass_name: str, video_id: str, ext: str) -> Path:
    return staging_dir(out_dir, pass_name) / f"{video_id}.{ext}"


def is_done(out_dir: str, pass_name: str, video_id: str, ext: str) -> bool:
    return staged_path(out_dir, pass_name, video_id, ext).exists()


def atomic_write_parquet(table: "pa.Table", path: Path) -> None:
    """Write parquet to <path>.tmp then rename, so a killed job leaves no partial."""
    import pyarrow.parquet as pq
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)

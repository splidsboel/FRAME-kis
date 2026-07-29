"""
prep_ocr.py — EasyOCR in-frame text for an extracted V3C shard, one parquet per
video into <out>/_staging/ocr/ (keyframe_id, span_index, text, confidence).
Folds ocr_extract.py + ocr_load.py into a single file-emitting pass (no jsonl
intermediate, no pg). Full fidelity kept (every span, raw confidence, MIN_CONF=0)
— thresholding is a query-time concern. A keyframe with no text contributes no
rows (absence == no text; per-video file presence == done). Resumable, shardable.

    python3 scripts/prep_ocr.py --shard-root ~/datasets/V3C/V3C2 --dataset v3c2 \
        --shard $SLURM_ARRAY_TASK_ID --num-shards 8
"""

from __future__ import annotations

import argparse
import os

import easyocr
import numpy as np
import pyarrow as pa
from PIL import Image

import sys
# scripts/ is one level below the repo root; put the root on sys.path for `import frame`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.prep.common import (EASYOCR_MODEL_DIR, OCR_LANGS, SCHEMAS,
                               atomic_write_parquet, is_done, iter_keyframes,
                               staged_path, video_dirs)


def ocr_image(reader, path):
    """Return list of (text, confidence) spans for one keyframe."""
    img = np.asarray(Image.open(path).convert("RGB"))
    spans = []
    for _bbox, text, conf in reader.readtext(img, detail=1, paragraph=False):
        text = text.strip()
        if text:
            spans.append((text, round(float(conf), 4)))
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-root", required=True)
    ap.add_argument("--dataset", default="v3c2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    shard_root = os.path.expanduser(args.shard_root)
    out_dir = args.out or os.path.join("data", "canonical", args.dataset)

    gpu = bool(int(os.environ.get("OCR_GPU", "1")))
    print(f"[ocr] loading EasyOCR {OCR_LANGS} gpu={gpu}", flush=True)
    reader = easyocr.Reader(OCR_LANGS, gpu=gpu, model_storage_directory=EASYOCR_MODEL_DIR,
                            download_enabled=False)
    schema = SCHEMAS["keyframe_ocr"]

    vids = video_dirs(shard_root, args.shard, args.num_shards)
    print(f"[ocr] shard {args.shard}/{args.num_shards}: {len(vids)} videos", flush=True)

    for i, (video_id, vdir) in enumerate(vids):
        if is_done(out_dir, "ocr", video_id, "parquet"):
            continue
        rows = []
        for kid, _vid, _n, path in iter_keyframes(vdir):
            try:
                spans = ocr_image(reader, path)
            except Exception:
                spans = []
            for span_index, (text, conf) in enumerate(spans):
                rows.append({"keyframe_id": kid, "span_index": span_index,
                             "text": text, "confidence": conf})

        cols = {f.name: [r.get(f.name) for r in rows] for f in schema}
        table = pa.table({f.name: pa.array(cols[f.name], type=f.type) for f in schema},
                         schema=schema)
        atomic_write_parquet(table, staged_path(out_dir, "ocr", video_id, "parquet"))
        print(f"[ocr] [{i+1}/{len(vids)}] {video_id}: {len(rows)} spans", flush=True)

    print("[ocr] done", flush=True)


if __name__ == "__main__":
    main()

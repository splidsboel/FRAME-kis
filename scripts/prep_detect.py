"""
prep_detect.py — OWLv2 open-vocab object detection for an extracted V3C shard,
one parquet per video into <out>/_staging/detect/ (keyframe_id, label, confidence,
x1..y2). Ports owlv2_detect.py; output is files, not pg. A video with zero
detections still writes an (empty) parquet, so its presence marks it done AND
every processed keyframe is recoverable for object_detection_done at consolidate.
Resumable (skip videos with a staged parquet). Shardable for arrays.

Uses the SAME external vocabulary (~/object_vocab.txt) and threshold (0.10) as
the V3C1 run — a vocab change means a fresh run for ALL shards (comparability).

    python3 scripts/prep_detect.py --shard-root ~/datasets/V3C/V3C2 --dataset v3c2 \
        --shard $SLURM_ARRAY_TASK_ID --num-shards 8
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor
import pyarrow as pa   # AFTER torch: torch's newer libstdc++ must load before pyarrow's (GLIBCXX)

# scripts/ is one level below the repo root; put the root on sys.path for `import frame`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.prep.common import (OWLV2_MODEL, OWLV2_SCORE_THRESHOLD, VOCAB_PATH,
                               atomic_write_parquet, is_done, iter_keyframes,
                               staged_path, video_dirs)

BATCH_SIZE = 8

# staging schema = canonical object_detections minus the global `id` (assigned at consolidate)
_F32 = pa.float32(); _S = pa.string()
DET_SCHEMA = pa.schema([("keyframe_id", _S), ("label", _S), ("confidence", _F32),
                        ("x1", _F32), ("y1", _F32), ("x2", _F32), ("y2", _F32)])


def load_vocab(path: str) -> list:
    with open(path) as f:
        vocab = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not vocab:
        sys.exit(f"empty vocabulary at {path}")
    print(f"[detect] vocabulary: {len(vocab)} terms", flush=True)
    return vocab


def run_batch(model, processor, device, vocab, items):
    """items: list of (keyframe_id, PIL image) -> list of detection dict rows."""
    frame_ids = [x[0] for x in items]
    images = [x[1] for x in items]
    inputs = processor(text=[vocab] * len(images), images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([img.size[::-1] for img in images], device=device)
    if hasattr(processor, "post_process_object_detection"):
        results = processor.post_process_object_detection(
            outputs, threshold=OWLV2_SCORE_THRESHOLD, target_sizes=target_sizes)
    else:
        results = processor.post_process_grounded_object_detection(
            outputs, threshold=OWLV2_SCORE_THRESHOLD, target_sizes=target_sizes,
            text_labels=[vocab] * len(images))

    rows = []
    for kid, res in zip(frame_ids, results):
        boxes = res["boxes"].cpu().tolist()
        scores = res["scores"].cpu().tolist()
        labels = res.get("text_labels")
        if labels is None:
            labels = [vocab[i] for i in res["labels"].cpu().tolist()]
        for (x1, y1, x2, y2), score, lab in zip(boxes, scores, labels):
            rows.append({"keyframe_id": kid, "label": lab, "confidence": float(score),
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return rows


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab(VOCAB_PATH)
    print(f"[detect] device={device} loading {OWLV2_MODEL}", flush=True)
    processor = Owlv2Processor.from_pretrained(OWLV2_MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(OWLV2_MODEL).to(device).eval()

    vids = video_dirs(shard_root, args.shard, args.num_shards)
    print(f"[detect] shard {args.shard}/{args.num_shards}: {len(vids)} videos", flush=True)

    for i, (video_id, vdir) in enumerate(vids):
        if is_done(out_dir, "detect", video_id, "parquet"):
            continue
        rows, batch = [], []
        for kid, _vid, _n, path in iter_keyframes(vdir):
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            batch.append((kid, img))
            if len(batch) >= BATCH_SIZE:
                rows.extend(run_batch(model, processor, device, vocab, batch)); batch = []
        if batch:
            rows.extend(run_batch(model, processor, device, vocab, batch))

        cols = {f.name: [r.get(f.name) for r in rows] for f in DET_SCHEMA}
        table = pa.table({f.name: pa.array(cols[f.name], type=f.type) for f in DET_SCHEMA},
                         schema=DET_SCHEMA)
        atomic_write_parquet(table, staged_path(out_dir, "detect", video_id, "parquet"))
        print(f"[detect] [{i+1}/{len(vids)}] {video_id}: {len(rows)} detections", flush=True)

    print("[detect] done", flush=True)


if __name__ == "__main__":
    main()

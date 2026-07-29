"""
prep_scenes.py — Places365 (ResNet50) scene classification for an extracted V3C
shard, one parquet per video into <out>/_staging/scenes/ (keyframe_id, label,
confidence): top-5 predictions per keyframe. Ports places365_classify.py; output
is files, not pg. Resumable (skip staged videos). Shardable for arrays.

Labels kept in canonical Places365 form (grouping dir dropped, indoor/outdoor
sub-path KEPT). Same external weights/labels as the V3C1 run.

    python3 scripts/prep_scenes.py --shard-root ~/datasets/V3C/V3C2 --dataset v3c2 \
        --shard $SLURM_ARRAY_TASK_ID --num-shards 8
"""

from __future__ import annotations

import argparse
import os

import pyarrow as pa
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import sys
# scripts/ is one level below the repo root; put the root on sys.path for `import frame`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.prep.common import (PLACES_LABELS, PLACES_WEIGHTS, SCENE_TOP_K, SCHEMAS,
                               atomic_write_parquet, is_done, iter_keyframes,
                               staged_path, video_dirs)

BATCH_SIZE = 256
TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
_softmax = torch.nn.Softmax(dim=1)


def parse_label(line: str) -> str:
    raw = line.strip().split(" ")[0]                 # '/c/church/indoor'
    return "/".join(raw.lstrip("/").split("/")[1:])  # 'church/indoor'


def load_model(device):
    ckpt = torch.load(PLACES_WEIGHTS, map_location="cpu", weights_only=False)
    state = {(k[len("module."):] if k.startswith("module.") else k): v
             for k, v in ckpt["state_dict"].items()}
    model = models.resnet50(num_classes=365)
    model.load_state_dict(state)
    return model.to(device).eval()


def run_batch(model, device, labels, items):
    """items: list of (keyframe_id, tensor) -> list of (kid, label, conf) dict rows."""
    ids = [x[0] for x in items]
    tensors = torch.stack([x[1] for x in items]).to(device, non_blocking=True)
    with torch.no_grad():
        probs = _softmax(model(tensors))
    top_probs, top_idx = probs.topk(SCENE_TOP_K, dim=1)
    rows = []
    for i, kid in enumerate(ids):
        for rank in range(SCENE_TOP_K):
            rows.append({"keyframe_id": kid,
                         "label": labels[top_idx[i, rank].item()],
                         "confidence": float(top_probs[i, rank].item())})
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
    with open(PLACES_LABELS) as f:
        labels = [parse_label(ln) for ln in f if ln.strip()]
    assert len(labels) == 365, f"expected 365 labels, got {len(labels)}"
    print(f"[scenes] device={device} loading Places365 ResNet50", flush=True)
    model = load_model(device)
    schema = SCHEMAS["scene_labels"]

    vids = video_dirs(shard_root, args.shard, args.num_shards)
    print(f"[scenes] shard {args.shard}/{args.num_shards}: {len(vids)} videos", flush=True)

    for i, (video_id, vdir) in enumerate(vids):
        if is_done(out_dir, "scenes", video_id, "parquet"):
            continue
        rows, batch = [], []
        for kid, _vid, _n, path in iter_keyframes(vdir):
            try:
                batch.append((kid, TRANSFORM(Image.open(path).convert("RGB"))))
            except Exception:
                continue
            if len(batch) >= BATCH_SIZE:
                rows.extend(run_batch(model, device, labels, batch)); batch = []
        if batch:
            rows.extend(run_batch(model, device, labels, batch))

        cols = {f.name: [r.get(f.name) for r in rows] for f in schema}
        table = pa.table({f.name: pa.array(cols[f.name], type=f.type) for f in schema},
                         schema=schema)
        atomic_write_parquet(table, staged_path(out_dir, "scenes", video_id, "parquet"))
        print(f"[scenes] [{i+1}/{len(vids)}] {video_id}: {len(rows)} labels", flush=True)

    print("[scenes] done", flush=True)


if __name__ == "__main__":
    main()

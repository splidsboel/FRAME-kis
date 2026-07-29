"""
prep_embed.py — SigLIP-base image embeddings for an extracted V3C shard, written
one .npz per video into <out>/_staging/embed/ (arrays: keyframe_id, embedding
float32[m,768], L2-normalised). Ports embed_keyframes.py's inference; output is
files, not pg. Resumable (skip videos whose .npz exists). Shardable for arrays.

    python3 scripts/prep_embed.py --shard-root ~/datasets/V3C/V3C2 --dataset v3c2 \
        --shard $SLURM_ARRAY_TASK_ID --num-shards 8
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

import sys
# scripts/ is one level below the repo root; put the root on sys.path for `import frame`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.prep.common import (EMBED_DIM, SIGLIP_MODEL, is_done, iter_keyframes,
                               staged_path, video_dirs)

BATCH_SIZE = 256


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
    print(f"[embed] device={device} loading {SIGLIP_MODEL}", flush=True)
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model = AutoModel.from_pretrained(SIGLIP_MODEL).to(device).eval()

    vids = video_dirs(shard_root, args.shard, args.num_shards)
    print(f"[embed] shard {args.shard}/{args.num_shards}: {len(vids)} videos", flush=True)

    for i, (video_id, vdir) in enumerate(vids):
        if is_done(out_dir, "embed", video_id, "npz"):
            continue
        ids, images = [], []
        for kid, _vid, _n, path in iter_keyframes(vdir):
            try:
                images.append(Image.open(path).convert("RGB"))
                ids.append(kid)
            except Exception as e:
                print(f"[embed][warn] {path}: {e}", flush=True)
        if not ids:
            continue

        vecs = np.empty((len(ids), EMBED_DIM), dtype=np.float32)
        for b in range(0, len(images), BATCH_SIZE):
            batch = images[b:b + BATCH_SIZE]
            inputs = processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
                out = F.normalize(out, dim=-1)
            vecs[b:b + len(batch)] = out.cpu().numpy().astype(np.float32)

        final = staged_path(out_dir, "embed", video_id, "npz")
        tmp = final.with_name(final.name + ".tmp")
        with open(tmp, "wb") as fh:   # file object => np.savez writes here verbatim
            np.savez(fh, keyframe_id=np.array(ids, dtype=object), embedding=vecs)
        os.replace(tmp, final)        # atomic: only a complete .npz appears
        print(f"[embed] [{i+1}/{len(vids)}] {video_id}: {len(ids)} keyframes", flush=True)

    print("[embed] done", flush=True)


if __name__ == "__main__":
    main()

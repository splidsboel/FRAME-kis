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

    # Printed before touching torch/HF: tasks 6+7 of job 100871 burned 24h on cn12
    # without emitting a single line, so this pins down whether a future hang is in
    # imports, CUDA init, or the model load.
    print(f"[embed] start shard={args.shard}/{args.num_shards} out={out_dir}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[embed] device={device} loading {SIGLIP_MODEL}", flush=True)
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
    model = AutoModel.from_pretrained(SIGLIP_MODEL).to(device).eval()

    vids = video_dirs(shard_root, args.shard, args.num_shards)
    print(f"[embed] shard {args.shard}/{args.num_shards}: {len(vids)} videos", flush=True)

    for i, (video_id, vdir) in enumerate(vids):
        if is_done(out_dir, "embed", video_id, "npz"):
            continue
        # Only the paths are held for the whole video — decoding every keyframe up
        # front peaked at tens of GB on long videos and OOM-killed the job (32 GB
        # cgroup). Images are now opened and released one batch at a time, so peak
        # RSS is bounded by BATCH_SIZE, not by video length.
        frames = [(kid, path) for kid, _vid, _n, path in iter_keyframes(vdir)]
        if not frames:
            continue

        ids, chunks = [], []
        for b in range(0, len(frames), BATCH_SIZE):
            batch_ids, images = [], []
            for kid, path in frames[b:b + BATCH_SIZE]:
                try:
                    with Image.open(path) as im:
                        images.append(im.convert("RGB"))
                    batch_ids.append(kid)
                except Exception as e:
                    print(f"[embed][warn] {path}: {e}", flush=True)
            if not batch_ids:
                continue

            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
                out = F.normalize(out, dim=-1)
            chunks.append(out.cpu().numpy().astype(np.float32))
            ids.extend(batch_ids)
            for im in images:
                im.close()
            del images, inputs, out

        if not ids:
            continue
        vecs = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        assert vecs.shape == (len(ids), EMBED_DIM), (vecs.shape, len(ids))

        final = staged_path(out_dir, "embed", video_id, "npz")
        tmp = final.with_name(final.name + ".tmp")
        with open(tmp, "wb") as fh:   # file object => np.savez writes here verbatim
            np.savez(fh, keyframe_id=np.array(ids, dtype=object), embedding=vecs)
        os.replace(tmp, final)        # atomic: only a complete .npz appears
        print(f"[embed] [{i+1}/{len(vids)}] {video_id}: {len(ids)} keyframes", flush=True)

    print("[embed] done", flush=True)


if __name__ == "__main__":
    main()

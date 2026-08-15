#!/usr/bin/env python3
"""
check_chroma_roundtrip.py — integration check for ChromaAdapter.load_data().

The Chroma analog of check_load_roundtrip.py (pgvector). It reuses that script's
tiny synthetic Tier-2 shard — the same awkward strings, the same below-threshold
labels, the same vectors — denormalises it into a scratch Chroma persist dir, and
verifies the data survived the JOIN-into-one-record: every filter type resolves to
the SAME passing keyframes pgvector's normalised JOIN returns, the pinned
thresholds drop the same rows, and load is idempotent + force-reloadable. Then the
scratch dir is removed.

This matters because the denormalisation is the one place a Chroma bug is silent: a
label folded onto the wrong record, a threshold applied on the wrong side, or a
case-folding slip in the OCR document does not raise — it just quietly matches the
wrong keyframes, and the recall number still looks plausible.

    sbatch chroma_load.sh --check           # via the wrapper (pip installs deps)
    python scripts/check_chroma_roundtrip.py

Needs the `chroma` + ingest deps (chromadb, pyarrow, h5py). Touches nothing real.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.adapters import ChromaAdapter
from frame.core.dataset import EMBED_DIM, Dataset
from frame.core.schema import Predicate

# Reuse the exact synthetic shard + assertion helper the pgvector check uses, so the
# two adapters are validated against identical data and any divergence is a real
# adapter difference, not a fixture difference.
from scripts.check_load_roundtrip import build_shard, check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="don't delete the scratch dir")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="frame_chromatest_")
    persist = os.path.join(tmp, "chroma")
    failures = 0
    try:
        kids = build_shard(os.path.join(tmp, "loadtest"))
        dataset = Dataset(os.path.join(tmp, "loadtest"))
        print(dataset.describe())

        adapter = ChromaAdapter(path=persist, ef_search=10)
        adapter.load_data(dataset)

        print("\n[check] search + filters (denormalised)")
        q = np.zeros(EMBED_DIM, dtype=np.float32)
        q[0] = 1.0
        with adapter:
            failures += not check("unfiltered ranking", adapter.search(q, [], k=3), kids[:3])

            scene = Predicate("scene", "scene_label", "in", ["night"])
            # kids[1]'s 'night' is below the pinned 0.10 threshold -> excluded, exactly
            # as in pgvector: the passing universe must be identical across systems.
            failures += not check("scene filter (threshold pinned)",
                                  adapter.search(q, [scene], k=10), [kids[0]])

            obj = Predicate("object", "object_label", "in", ["car"])
            # kids[1]'s 'car' is below the pinned 0.30 object threshold -> excluded.
            failures += not check("object filter (threshold pinned)",
                                  adapter.search(q, [obj], k=10), [kids[0]])

            ocr = Predicate("pattern-match", "ocr_text", "contains", ["stop"])
            # OCR document is stored lower-cased; 'STOP...' matches 'stop' (pgvector's
            # case-insensitive LIKE reproduced via lower-cased $contains).
            failures += not check("pattern-match (case-insensitive)",
                                  adapter.search(q, [ocr], k=10), [kids[0]])

            cat = Predicate("video-category", "categories", "overlaps", ["documentary"])
            # Video-level facet denormalised onto every keyframe of video 90001.
            failures += not check("video-category filter",
                                  adapter.search(q, [cat], k=10), kids[:4])

            tag = Predicate("video-tag", "tags", "overlaps", ["news, politics"])
            # A tag with a comma must survive as ONE array element, not split.
            failures += not check("video-tag filter (comma in element)",
                                  adapter.search(q, [tag], k=10), kids[:4])

            # Conjunction: scene AND object, both on kids[0] -> AND-ed across filters.
            failures += not check("scene AND object (conjunction)",
                                  adapter.search(q, [scene, obj], k=10), [kids[0]])

            # A miss: nothing carries this scene label.
            miss = Predicate("scene", "scene_label", "in", ["kitchen"])
            failures += not check("empty result set", adapter.search(q, [miss], k=10), [])

            failures += not check("corpus_size", adapter.corpus_size(), len(kids))

        print("\n[check] idempotency — a second load must change nothing")
        adapter2 = ChromaAdapter(path=persist)
        adapter2.load_data(dataset)
        with adapter2:
            failures += not check("corpus_size after reload", adapter2.corpus_size(), len(kids))

        print("\n[check] force reload")
        adapter3 = ChromaAdapter(path=persist)
        adapter3.load_data(dataset, force=True)
        with adapter3:
            failures += not check("corpus_size after force", adapter3.corpus_size(), len(kids))
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        sys.exit(f"FAILED: {failures} check(s)")
    print("all checks passed")


if __name__ == "__main__":
    main()

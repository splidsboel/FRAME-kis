#!/usr/bin/env python3
"""
load_dataset.py — ingest a Tier-2 canonical shard into one system under test.

This is the out-of-band, one-time step between the shared data pipeline and a
benchmark run (see [[Data pipeline and adapter load refactor]]):

    Tier 1 raw shard  --prep_*.py-->  Tier 2 canonical files
                                            |
                                            |  THIS SCRIPT: adapter.load_data()
                                            v
                                      Tier 3 the system's physical layout

Deliberately NOT part of run_benchmark.py. A benchmark run must never pay (or
hide) a multi-million-row ingest; `setup()` raises if the data isn't there rather
than quietly loading it.

    python scripts/load_dataset.py --dataset data/canonical/v3c1
    python scripts/load_dataset.py --dataset data/canonical/v3c2 --system pgvector
    python scripts/load_dataset.py --dataset data/canonical/v3c1 --describe

    # union corpus (v3c1+2+3) — one HNSW index over every shard:
    python scripts/load_dataset.py \
        --dataset data/canonical/v3c1 data/canonical/v3c2 data/canonical/v3c3
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.adapters import PgvectorAdapter
from frame.core.dataset import Dataset

ADAPTERS = {
    "pgvector": PgvectorAdapter,
    # "chroma": ChromaAdapter,   # later — same call, different physical layout
    # "milvus": MilvusAdapter,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, nargs="+",
                    help="path(s) to canonical shard(s), e.g. data/canonical/v3c1. "
                         "Several paths load the UNION corpus into one instance "
                         "(one HNSW index over every shard).")
    ap.add_argument("--system", default="pgvector", choices=sorted(ADAPTERS))
    ap.add_argument("--force", action="store_true",
                    help="reload even if the row counts already match (for the "
                         "union: truncate and rebuild from scratch)")
    ap.add_argument("--describe", action="store_true",
                    help="print what the shard(s) contain and exit (no DB needed)")
    args = ap.parse_args()

    datasets = [Dataset(p) for p in args.dataset]
    for ds in datasets:
        ds.validate()

    if args.describe:
        for ds in datasets:
            print(ds.describe())
        return

    adapter = ADAPTERS[args.system]()
    adapter.load_datasets(datasets, force=args.force)

    # Prove the result is queryable through the same path a run will use.
    adapter.setup()
    try:
        print(f"[load] verified: {adapter.corpus_size():,} keyframes queryable "
              f"in {args.system}")
    finally:
        adapter.teardown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_benchmark.py — wire one adapter end-to-end: load query set -> run -> analyze.

    uv run python run_benchmark.py --system pgvector

Needs the enriched data/benchmark.jsonl (queryset/build.py + oracle/build_gt.py),
a reachable V3C postgres (PG* env vars / PGHOST socket), and the encoder deps
(`uv sync --extra encode`). Adding Chroma/Milvus later = a new adapter here.
"""

from __future__ import annotations

import argparse
import os

from frame import Analyzer, Runner, load_query_set
from frame.adapters import PgvectorAdapter
from frame.core.encode import CachingEncoder, SiglipEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

ADAPTERS = {
    "pgvector": PgvectorAdapter,
    # "chroma": ChromaAdapter,   # later
    # "milvus": MilvusAdapter,   # later
}


def explain_one(adapter, encoder, items, qid, k):
    """Confirm whether a filtered search uses the HNSW index or a seqscan, and
    whether iterative_scan flips it. Only pgvector supports .explain()."""
    item = next((it for it in items if it.query_id == qid), None)
    if item is None:
        raise SystemExit(f"no such query_id: {qid}")
    if not item.filters:
        raise SystemExit(f"{qid} has no filters — nothing to explain")
    if not hasattr(adapter, "explain"):
        raise SystemExit(f"{adapter.name} adapter has no explain()")

    vec = encoder.encode(item.vector_query)
    with adapter:
        for mode in ("off", "relaxed_order"):
            print(f"\n{'='*70}\n{qid}  filters={len(item.filters)}  "
                  f"k={k}  hnsw.iterative_scan={mode}\n{'='*70}")
            print(adapter.explain(vec, item.filters, k, iterative_scan=mode))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="pgvector", choices=sorted(ADAPTERS))
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--bench", default=os.path.join(DATA, "benchmark.jsonl"))
    ap.add_argument("--explain", metavar="QID",
                    help="diagnostic: EXPLAIN the filtered search for one query "
                         "under iterative_scan off vs relaxed_order, then exit")
    args = ap.parse_args()

    items = load_query_set(args.bench)
    print(f"loaded {len(items)} items from {os.path.relpath(args.bench, HERE)}")

    encoder = CachingEncoder(SiglipEncoder())
    adapter = ADAPTERS[args.system]()

    if args.explain:
        explain_one(adapter, encoder, items, args.explain, args.k)
        return

    with adapter:
        raw = Runner(adapter, encoder).run(items, k=args.k)

    raw_path = os.path.join(DATA, f"raw_results.{args.system}.jsonl")
    raw.write_jsonl(raw_path)

    metrics = Analyzer().analyze(raw, items)
    metrics_path = os.path.join(DATA, f"metrics.{args.system}.jsonl")
    metrics.write_jsonl(metrics_path)

    print()
    print(Analyzer().summary(metrics))
    print()
    print(f"raw     -> {os.path.relpath(raw_path, HERE)}")
    print(f"metrics -> {os.path.relpath(metrics_path, HERE)}")


if __name__ == "__main__":
    main()

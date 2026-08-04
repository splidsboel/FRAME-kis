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


# pgvector HNSW iterative_scan modes, swept as an experiment axis (the planner
# takes the HNSW path on selective filters, where this knob governs the recall /
# latency / truncation tradeoff — see "pgvector planner split" finding).
PG_ITERATIVE_MODES = ("off", "relaxed_order", "strict_order")


def run_and_score(system, encoder, items, k, warmup, repeat, adapter_kwargs, label):
    """One full run for one adapter configuration; writes tagged artifacts, returns Metrics."""
    adapter = ADAPTERS[system](**adapter_kwargs)
    with adapter:
        raw = Runner(adapter, encoder, warmup=warmup, repeat=repeat).run(items, k=k)

    suffix = f".{label}" if label else ""
    raw_path = os.path.join(DATA, f"raw_results.{system}{suffix}.jsonl")
    raw.write_jsonl(raw_path)
    metrics = Analyzer().analyze(raw, items)
    metrics_path = os.path.join(DATA, f"metrics.{system}{suffix}.jsonl")
    metrics.write_jsonl(metrics_path)

    print()
    print(Analyzer().summary(metrics))
    print()
    print(f"raw     -> {os.path.relpath(raw_path, HERE)}")
    print(f"metrics -> {os.path.relpath(metrics_path, HERE)}")
    print(f"figures -> python scripts/plot_metrics.py --in "
          f"{os.path.relpath(metrics_path, HERE)}")
    return metrics


def print_sweep_comparison(results_by_mode, k):
    print("\n" + "=" * 60)
    print(f"iterative_scan sweep — filtered condition (k={k})")
    print("=" * 60)
    print(f"{'mode':>14} | {'recall@'+str(k):>11} | {'MRR':>6} | {'med lat ms':>11}")
    print("-" * 60)
    for mode, m in results_by_mode.items():
        print(f"{mode:>14} | {m.mean_recall_filtered(k):>11.3f} | "
              f"{m.mrr_filtered():>6.3f} | {m.median_latency_filtered():>11.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="pgvector", choices=sorted(ADAPTERS))
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=1,
                    help="untimed passes per search to prime plan cache + buffers")
    ap.add_argument("--repeat", type=int, default=5,
                    help="timed passes per search; latency reported as their median")
    ap.add_argument("--bench", default=os.path.join(DATA, "benchmark.jsonl"))
    ap.add_argument("--iterative-scan", default="relaxed_order",
                    choices=[*PG_ITERATIVE_MODES, "sweep"],
                    help="pgvector HNSW iterative_scan mode; 'sweep' runs all three "
                         "and tags outputs per mode (pgvector only)")
    ap.add_argument("--explain", metavar="QID",
                    help="diagnostic: EXPLAIN the filtered search for one query "
                         "under iterative_scan off vs relaxed_order, then exit")
    args = ap.parse_args()

    items = load_query_set(args.bench)
    print(f"loaded {len(items)} items from {os.path.relpath(args.bench, HERE)}")

    encoder = CachingEncoder(SiglipEncoder())

    if args.explain:
        explain_one(ADAPTERS[args.system](), encoder, items, args.explain, args.k)
        return

    is_pg = args.system == "pgvector"
    if args.iterative_scan == "sweep":
        if not is_pg:
            raise SystemExit("--iterative-scan sweep is pgvector-only")
        modes = list(PG_ITERATIVE_MODES)
    else:
        modes = [args.iterative_scan]

    results_by_mode = {}
    for mode in modes:
        # iterative_scan is a pgvector knob; other adapters take no such kwarg.
        adapter_kwargs = {"iterative_scan": mode} if is_pg else {}
        label = mode if len(modes) > 1 else None
        if len(modes) > 1:
            print(f"\n{'#'*60}\n# iterative_scan = {mode}\n{'#'*60}")
        results_by_mode[mode] = run_and_score(
            args.system, encoder, items, args.k,
            args.warmup, args.repeat, adapter_kwargs, label)

    if len(modes) > 1:
        print_sweep_comparison(results_by_mode, args.k)


if __name__ == "__main__":
    main()

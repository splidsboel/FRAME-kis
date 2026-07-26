#!/usr/bin/env python3
"""
sweep_cutover.py — dense recall-vs-selectivity sweep across the exact↔approximate
cutover.

    uv run python scripts/sweep_cutover.py --system pgvector

Turns the bracketed cutover claim (~5–10% selectivity) into a measured curve: for
each filtered item, sweeps an ef_search × iterative_scan grid at a moderate run
depth k (default 100) and records the plan, rows returned, and recall@k vs the
oracle's EXACT filtered k-NN (gt_filtered). Two plan modes:

  * auto       — the planner's real seqscan/HNSW choice (WHERE the cutover flips)
  * force-hnsw — SET enable_seqscan=off (WHAT the approximate path costs at each
                 selectivity, densely, even where the planner stays exact)

Scores ALL filtered items with GT, including the harm exemplars whose filter
excludes their own target (recall vs gt_filtered is geometric, independent of KIS
task success). See frame/core/sweep.py and the vault note
"FRAME — pgvector planner split".

Needs the enriched data/benchmark.jsonl and a reachable V3C postgres (PG* env vars
/ PGHOST socket) — so it runs on the HPC, like run_benchmark / profile_queryset.
"""

from __future__ import annotations

import argparse
import os
import sys

# scripts/ lives one level below the repo root; put the root on sys.path so the
# `frame` package imports whether run via uv, conda, or a bare python3.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from frame import Sweeper, load_query_set
from frame.adapters import PgvectorAdapter
from frame.core.encode import CachingEncoder, SiglipEncoder
from frame.core.sweep import (
    DEFAULT_EF_GRID,
    DEFAULT_K,
    DEFAULT_MODES,
    DEFAULT_PLAN_MODES,
    DEFAULT_SCORE_KS,
    write_jsonl,
)

DATA = os.path.join(REPO, "data")

ADAPTERS = {
    "pgvector": PgvectorAdapter,
    # "chroma": ChromaAdapter,   # later — needs its own plan/selectivity diagnostics
    # "milvus": MilvusAdapter,   # later
}


def _ints(csv: str) -> list[int]:
    return [int(x) for x in csv.split(",") if x.strip()]


def _strs(csv: str) -> list[str]:
    return [x.strip() for x in csv.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="pgvector", choices=sorted(ADAPTERS))
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help="run depth (LIMIT); ef_search points below it are dropped "
                         "(pgvector requires ef_search >= k)")
    ap.add_argument("--ef-search", default=",".join(map(str, DEFAULT_EF_GRID)),
                    help="comma-separated ef_search grid (values <k or >1000 dropped)")
    ap.add_argument("--iterative-scan", default=",".join(DEFAULT_MODES),
                    help="comma-separated pgvector iterative_scan modes")
    ap.add_argument("--plan", default=",".join(DEFAULT_PLAN_MODES),
                    help="comma-separated plan modes: auto and/or force-hnsw")
    ap.add_argument("--score-ks", default=",".join(map(str, DEFAULT_SCORE_KS)),
                    help="comma-separated recall@k depths to score (each <= k)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="untimed passes per search to prime plan cache + buffers")
    ap.add_argument("--repeat", type=int, default=5,
                    help="timed passes per search; latency reported as their median")
    ap.add_argument("--bench", default=os.path.join(DATA, "benchmark.jsonl"))
    args = ap.parse_args()

    plan_modes = _strs(args.plan)
    bad = [p for p in plan_modes if p not in ("auto", "force-hnsw")]
    if bad:
        raise SystemExit(f"unknown --plan mode(s): {bad} (choose auto / force-hnsw)")
    if args.system != "pgvector" and "force-hnsw" in plan_modes:
        raise SystemExit("force-hnsw is pgvector-only (enable_seqscan GUC)")

    items = load_query_set(args.bench)
    print(f"loaded {len(items)} items from {os.path.relpath(args.bench, REPO)}")

    encoder = CachingEncoder(SiglipEncoder())
    sweeper = Sweeper(
        ADAPTERS[args.system](),
        encoder,
        k=args.k,
        ef_grid=_ints(args.ef_search),
        modes=_strs(args.iterative_scan),
        plan_modes=plan_modes,
        score_ks=_ints(args.score_ks),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    print(f"grid: k={sweeper.k}  ef_search={sweeper.ef_grid}  "
          f"iterative_scan={list(sweeper.modes)}  plan={list(sweeper.plan_modes)}  "
          f"score@{list(sweeper.score_ks)}")

    cells = sweeper.run(items)

    print()
    print(sweeper.summary(cells))

    out = os.path.join(DATA, f"sweep.{args.system}.jsonl")
    write_jsonl(cells, out, args.system, sweeper.k)
    print()
    print(f"{len(cells)} cells -> {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()

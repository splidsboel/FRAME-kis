#!/usr/bin/env python3
"""
profile_queryset.py — selectivity / plan profile of the authored query set.

    uv run python scripts/profile_queryset.py --system pgvector

For each filtered item, reports where its REAL VBS filter lands on pgvector's
exact↔approximate cutover: true global selectivity, the plan the planner picks
(HNSW post-filter vs exact seqscan) + its estimated passing set, the near-query
pass-rate (local pass-rate the HNSW walk sees), and their divergence. See
frame/core/profile.py and the vault note "FRAME — pgvector planner split".

Needs the enriched data/benchmark.jsonl and a reachable V3C postgres (PG* env
vars / PGHOST socket) — so it runs on the HPC, like run_benchmark. Near-query
pass-rate additionally needs geometric_gt_vec_nofilter in the GT (a build_gt
baseline); it reports n/a where absent, the rest still profiles.
"""

from __future__ import annotations

import argparse
import os
import sys

# scripts/ lives one level below the repo root; put the root on sys.path so the
# `frame` package imports whether run via uv, conda, or a bare python3.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from frame import Profiler, load_query_set
from frame.adapters import PgvectorAdapter
from frame.core.encode import CachingEncoder, SiglipEncoder
from frame.core.profile import write_jsonl

DATA = os.path.join(REPO, "data")

ADAPTERS = {
    "pgvector": PgvectorAdapter,
    # "chroma": ChromaAdapter,   # later — needs its own plan/selectivity diagnostics
    # "milvus": MilvusAdapter,   # later
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="pgvector", choices=sorted(ADAPTERS))
    ap.add_argument("--k", type=int, default=1000,
                    help="retrieval depth the plan is priced at (match the run's k "
                         "so plan choice agrees with the benchmark)")
    ap.add_argument("--near-query-n", type=int, default=100,
                    help="how many of each query's exact unfiltered neighbours to "
                         "test for the near-query (local) pass-rate")
    ap.add_argument("--bench", default=os.path.join(DATA, "benchmark.jsonl"))
    args = ap.parse_args()

    items = load_query_set(args.bench)
    print(f"loaded {len(items)} items from {os.path.relpath(args.bench, REPO)}")

    encoder = CachingEncoder(SiglipEncoder())
    profiler = Profiler(ADAPTERS[args.system](), encoder,
                        k=args.k, near_query_n=args.near_query_n)
    profiles = profiler.profile(items)

    print()
    print(profiler.summary(profiles))

    out = os.path.join(DATA, f"profile.{args.system}.jsonl")
    write_jsonl(profiles, out, args.system)
    print()
    print(f"profile -> {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()

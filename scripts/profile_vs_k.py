#!/usr/bin/env python3
"""
profile_vs_k.py — plan choice as a function of retrieval depth k.

    uv run python scripts/profile_vs_k.py --system pgvector

The controlled confirmation of the cutover's k-dependence found by the step-2 sweep
(see the vault: "FRAME — cutover sweep"). For each filtered item it prices the plan
the planner would pick — exact seqscan (filter-then-scan) vs approximate HNSW
post-filter — at several k values, on ONE connection with one pinned ANALYZE.

Mechanism under test: an HNSW post-filter walks ≈ k / pass-rate candidates before
LIMIT k stops it, so the seqscan↔HNSW flip is NOT a fixed selectivity threshold —
it moves toward broader coverage as k shrinks. This script turns that into a
plan-flip-vs-depth curve over the real query set.

Cheap: plan_choice uses EXPLAIN (FORMAT JSON) WITHOUT ANALYZE (no query executed),
so this only encodes each item once and issues one EXPLAIN per (item, k). Emits a
tidy artifact (one row per (query_id, k)) plus a plan matrix. Plan is priced at the
adapter's default iterative_scan (relaxed_order), matching the sweep.

Needs the enriched data/benchmark.jsonl and a reachable V3C postgres (PG* env vars /
PGHOST socket) — runs on the HPC like the other DB scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from frame import load_query_set
from frame.adapters import PgvectorAdapter
from frame.core.encode import CachingEncoder, SiglipEncoder
from frame.core.profile import _summarize_filters

DATA = os.path.join(REPO, "data")
DEFAULT_K_GRID = (50, 100, 250, 1000)

ADAPTERS = {"pgvector": PgvectorAdapter}


def _ints(csv: str) -> list[int]:
    return [int(x) for x in csv.split(",") if x.strip()]


_PLAN_ABBR = {"hnsw": "hnsw", "seqscan": "seq", "other": "oth"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="pgvector", choices=sorted(ADAPTERS))
    ap.add_argument("--k-grid", default=",".join(map(str, DEFAULT_K_GRID)),
                    help="comma-separated retrieval depths to price the plan at")
    ap.add_argument("--bench", default=os.path.join(DATA, "benchmark.jsonl"))
    args = ap.parse_args()

    k_grid = _ints(args.k_grid)
    items = load_query_set(args.bench)
    filtered = [it for it in items if it.filters]
    print(f"loaded {len(items)} items ({len(filtered)} filtered) from "
          f"{os.path.relpath(args.bench, REPO)}")
    print(f"k grid: {k_grid}")

    encoder = CachingEncoder(SiglipEncoder())
    rows: list[dict] = []
    with ADAPTERS[args.system]() as adapter:      # one connection, one pinned ANALYZE
        corpus = adapter.corpus_size()
        for it in filtered:
            vec = encoder.encode(it.vector_query)
            global_count = adapter.count_passing(it.filters)
            sel = (global_count / corpus) if corpus else 0.0
            for k in k_grid:
                plan, est = adapter.plan_choice(vec, it.filters, k)
                rows.append({
                    "query_id": it.query_id,
                    "filter_summary": _summarize_filters(it.filters),
                    "n_filters": len(it.filters),
                    "global_selectivity": sel,
                    "k": k,
                    "plan": plan,
                    "planner_est_rows": est,
                })

    # ── plan matrix: rows = items by selectivity, cols = k ──
    by_qid: dict[str, dict[int, dict]] = {}
    for r in rows:
        by_qid.setdefault(r["query_id"], {})[r["k"]] = r
    order = sorted(by_qid, key=lambda q: next(iter(by_qid[q].values()))["global_selectivity"])
    header = f"{'qid':>6} | {'sel':>6} | " + " | ".join(f"k={k:<5}" for k in k_grid)
    print("\nplan by retrieval depth (seq = exact seqscan, hnsw = approximate, oth = other)")
    print(header)
    print("-" * len(header))
    for q in order:
        r0 = next(iter(by_qid[q].values()))
        cells = [f"{_PLAN_ABBR.get(by_qid[q].get(k, {}).get('plan',''),'-'):<7}" for k in k_grid]
        print(f"{q:>6} | {r0['global_selectivity']:6.3f} | " + " | ".join(cells))

    # per-k HNSW count — the headline curve
    print("\nHNSW-path items by k (of %d filtered):" % len(filtered))
    for k in k_grid:
        n = sum(1 for r in rows if r["k"] == k and r["plan"] == "hnsw")
        print(f"  k={k:>5}: {n}")

    out = os.path.join(DATA, f"profile_vs_k.{args.system}.jsonl")
    with open(out, "w") as f:
        f.write(json.dumps({"system": args.system, "kind": "plan_vs_k",
                            "k_grid": k_grid}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n{len(rows)} rows -> {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()

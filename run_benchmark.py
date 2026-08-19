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

from frame import Analyzer, Runner
from frame.adapters import ChromaAdapter, PgvectorAdapter
from frame.core.encode import CachingEncoder, SiglipEncoder
from frame.core.schema import BY_NAME, PRIMARY_FILTERED, load_benchmark

# k×ef sweep (see vault "FRAME k×ef sweep — plan and code spec"). The recall/MRR
# cutoffs and MRR caps swept per cell — capped to <= the cell's k, since no rank
# beyond the retrieval depth exists. 10 is added over the default ks to read the
# KIS-relevant top-10.
SWEEP_KS = (5, 10, 25, 50, 100)
SWEEP_MRR_CAPS = (10, 50, 100)
# Lean condition subset for the sweep: ef is a recall dial only on the HNSW path, so
# the two semantic cells carry the signal; the raw cells only add filtered-search
# cost. The full 2x2 is run at the chosen headline (k, ef) cell instead.
LEAN_CONDITIONS = (BY_NAME["semantic+filter"], BY_NAME["semantic+nofilter"])

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

ADAPTERS = {
    "pgvector": PgvectorAdapter,
    "chroma": ChromaAdapter,   # denormalised single-collection (no JOINs)
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


def run_and_score(system, encoder, qs, k, warmup, repeat, adapter_kwargs, label,
                  allow_mismatch=False, exemplars="isolate", grade_variants=False,
                  ks=None, mrr_caps=None, conditions=None):
    """One full run for one adapter configuration; writes tagged artifacts, returns Metrics.

    `exemplars` controls how the filter-harm exemplars (target excluded by its own
    filter) are treated: 'isolate' (default) keeps them out of the headline but
    reports them separately; 'exclude' keeps them out with only a count note;
    'include' folds them into the headline aggregate too.

    `grade_variants` additionally runs every real human phrasing of each item as its
    own query (task-success MRR over real wordings — Omar, 2026-08-18).

    `ks` / `mrr_caps` override the Analyzer's recall / MRR cutoffs (the k×ef sweep
    caps them to the cell's depth); `conditions` restricts the 2x2 cells the Runner
    executes (the sweep runs a lean subset). All three default to the full-run
    behaviour when None.
    """
    analyzer_kwargs = {"score_harm_exemplars": exemplars == "include"}
    if ks is not None:
        analyzer_kwargs["ks"] = ks
    if mrr_caps is not None:
        analyzer_kwargs["mrr_caps"] = mrr_caps
    analyzer = Analyzer(**analyzer_kwargs)
    adapter = ADAPTERS[system](**adapter_kwargs)

    # Stream results to raw_path AS the run proceeds, so a wall-clock kill (SLURM
    # time limit) on a long run keeps every query already finished. The final
    # write_jsonl below is the canonical complete write; on success it just rewrites
    # the same bytes, and on a kill the streamed partial file is left analysable.
    suffix = f".{label}" if label else ""
    raw_path = os.path.join(DATA, f"raw_results.{system}{suffix}.jsonl")
    with adapter:
        raw = Runner(adapter, encoder, warmup=warmup, repeat=repeat,
                     grade_variants=grade_variants, conditions=conditions).run(
            qs.items, k=k, benchmark=qs.version, progress_path=raw_path)

    raw.write_jsonl(raw_path)
    metrics = analyzer.analyze(raw, qs.items, benchmark=qs.version,
                               allow_mismatch=allow_mismatch)
    metrics_path = os.path.join(DATA, f"metrics.{system}{suffix}.jsonl")
    metrics.write_jsonl(metrics_path)

    print()
    print(analyzer.summary(metrics))
    # The filtered workload split by conjunction selectivity (Omar, 2026-08-10).
    sel = analyzer.summary_by_selectivity(metrics)
    if sel:
        print(sel)
    # The filter-harm exemplars, on their own terms — unless suppressed.
    n_exemplars = sum(1 for q in metrics.per_query if q.harm_exemplar)
    if n_exemplars and exemplars != "exclude":
        print(analyzer.harm_exemplar_report(metrics))
    elif n_exemplars:
        print(f"\n[note] {n_exemplars} filter-harm exemplar(s) excluded from the "
              f"headline (target excluded by its own filter); --exemplars isolate to "
              f"see them")
    # Per-phrasing task success (only when --grade-variants ran).
    variants = analyzer.variant_summary(metrics)
    if variants:
        print(variants)
    print()
    print(f"raw     -> {os.path.relpath(raw_path, HERE)}")
    print(f"metrics -> {os.path.relpath(metrics_path, HERE)}")
    print(f"figures -> python scripts/plot_metrics.py --in "
          f"{os.path.relpath(metrics_path, HERE)}")
    return metrics


def print_sweep_comparison(results_by_mode, k):
    # The knob only affects the FILTER cells (it governs how the planner walks the
    # index under a predicate), so the sweep is reported on the primary filtered
    # condition rather than averaged across the 2x2.
    cond = PRIMARY_FILTERED
    print("\n" + "=" * 66)
    print(f"iterative_scan sweep — {cond} condition (k={k})")
    print("=" * 66)
    print(f"{'mode':>14} | {'recall@'+str(k):>11} | {'MRR':>6} | {'med lat ms':>11} | {'p95':>8}")
    print("-" * 66)
    for mode, m in results_by_mode.items():
        print(f"{mode:>14} | {m.mean_recall(cond, k):>11.3f} | "
              f"{m.mrr(cond):>6.3f} | {m.median_latency(cond):>11.1f} | "
              f"{m.latency_percentile(cond, 95):>8.1f}")


def _ints(csv: str) -> list[int]:
    return [int(x) for x in csv.split(",") if x.strip()]


def run_kef_sweep(args, encoder, qs):
    """Sweep k × ef_search, running the pipeline natively at each legal cell.

    Legal cell: k <= ef <= 1000 (both pgvector and Chroma/hnswlib require the beam
    to be at least the fetch depth). Each cell writes artifacts tagged
    `.k{k}.ef{ef}` and scores recall/MRR only at cutoffs <= k. iterative_scan is
    pinned to relaxed_order on pgvector (sweeping it is the separate cutover
    diagnostic, frame/core/sweep.py)."""
    is_pg = args.system == "pgvector"
    k_grid = _ints(args.k_grid) if args.k_grid else [args.k]
    ef_grid = _ints(args.ef_grid) if args.ef_grid else [args.k]
    cells = [(k, ef) for k in k_grid for ef in ef_grid if k <= ef <= 1000]
    if not cells:
        raise SystemExit(f"empty k×ef grid: no cell satisfies k <= ef <= 1000 "
                         f"(k_grid={k_grid}, ef_grid={ef_grid})")
    conditions = LEAN_CONDITIONS if args.conditions == "lean" else None

    print(f"k×ef sweep: {len(cells)} cell(s) on {args.system} — "
          f"{args.conditions} conditions")
    for k, ef in cells:
        adapter_kwargs = {"ef_search": ef}
        if is_pg:
            adapter_kwargs["iterative_scan"] = "relaxed_order"
        ks = [x for x in SWEEP_KS if x <= k]
        caps = [x for x in SWEEP_MRR_CAPS if x <= k]
        print(f"\n{'#'*60}\n# k={k}  ef_search={ef}\n{'#'*60}")
        run_and_score(
            args.system, encoder, qs, k, args.warmup, args.repeat,
            adapter_kwargs, label=f"k{k}.ef{ef}", allow_mismatch=args.allow_mismatch,
            exemplars=args.exemplars, grade_variants=args.grade_variants,
            ks=ks, mrr_caps=caps, conditions=conditions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="pgvector", choices=sorted(ADAPTERS))
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--k-grid", default=None,
                    help="comma-separated retrieval depths to sweep NATIVELY (each a "
                         "separate run at that LIMIT); triggers the k×ef sweep. "
                         "Default: just --k")
    ap.add_argument("--ef-grid", default=None,
                    help="comma-separated hnsw ef_search values to sweep; cells with "
                         "ef < k or ef > 1000 are dropped (both engines need ef>=k). "
                         "Triggers the k×ef sweep. Default: the adapter's ef_search")
    ap.add_argument("--conditions", default="full", choices=["full", "lean"],
                    help="which 2x2 cells to run in the sweep: 'lean' = the two "
                         "semantic cells (semantic+filter, semantic+nofilter); 'full' "
                         "= all four. Only affects the k×ef sweep")
    ap.add_argument("--warmup", type=int, default=1,
                    help="untimed passes per search to prime plan cache + buffers")
    ap.add_argument("--repeat", type=int, default=5,
                    help="timed passes per search; latency reported as their median")
    ap.add_argument("--bench", default=os.path.join(DATA, "benchmark.jsonl"))
    ap.add_argument("--iterative-scan", default="relaxed_order",
                    choices=[*PG_ITERATIVE_MODES, "sweep"],
                    help="pgvector HNSW iterative_scan mode; 'sweep' runs all three "
                         "and tags outputs per mode (pgvector only)")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="score even if the results were not produced against this "
                         "query set / harness version (numbers will NOT be "
                         "comparable — see frame/core/version.py)")
    ap.add_argument("--exemplars", default="isolate",
                    choices=["isolate", "exclude", "include"],
                    help="how to treat the filter-harm exemplars (target excluded by "
                         "its own filter): 'isolate' keeps them out of the headline "
                         "but reports them separately (default); 'exclude' just notes "
                         "the count; 'include' also folds them into the headline")
    ap.add_argument("--grade-variants", action="store_true",
                    help="also run every real human phrasing of each item as its own "
                         "query (task-success MRR over real wordings, all vs "
                         "succeeding-only). Adds searches — a filtered phrasing costs "
                         "as much as a 2x2 filter cell, so this is off by default")
    ap.add_argument("--explain", metavar="QID",
                    help="diagnostic: EXPLAIN the filtered search for one query "
                         "under iterative_scan off vs relaxed_order, then exit")
    args = ap.parse_args()

    qs = load_benchmark(args.bench)
    items = qs.items
    print(f"loaded {len(items)} items from {os.path.relpath(args.bench, HERE)}")
    if qs.version is None:
        print("[WARN] this benchmark.jsonl has no version marker — rebuild it with "
              "`uv run python queryset/build.py` so runs can be shown comparable")
    else:
        print(f"query set: {qs.version.label}  "
              f"({qs.version.n_with_gt}/{qs.version.n_items} with GT)")

    encoder = CachingEncoder(SiglipEncoder())

    if args.explain:
        explain_one(ADAPTERS[args.system](), encoder, items, args.explain, args.k)
        return

    # k×ef sweep: run the full pipeline NATIVELY at each (k, ef) cell, tagging each
    # cell's artifacts. Triggered by --k-grid and/or --ef-grid. See the vault note
    # "FRAME k×ef sweep — plan and code spec".
    if args.k_grid or args.ef_grid:
        if args.iterative_scan == "sweep":
            raise SystemExit("--iterative-scan sweep and the k×ef sweep "
                             "(--k-grid/--ef-grid) are separate experiments; pick one")
        run_kef_sweep(args, encoder, qs)
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
            args.system, encoder, qs, args.k,
            args.warmup, args.repeat, adapter_kwargs, label, args.allow_mismatch,
            exemplars=args.exemplars, grade_variants=args.grade_variants)

    if len(modes) > 1:
        print_sweep_comparison(results_by_mode, args.k)


if __name__ == "__main__":
    main()

# FRAME — a Filtered-ANN benchmark suite for Known-Item Search

FRAME benchmarks how vector-database systems handle **filtered approximate
nearest-neighbor search** under realistic Known-Item-Search (KIS) workloads over
the [V3C](https://videobrowsershowdown.org/) video collection.

Instead of synthetic predicates at swept selectivities, its queries come from real
interactive-retrieval sessions. Each query is decomposed into a **semantic part**
(sent to an embedding model) and **structured filter predicates** (scene, object,
in-frame text), each with exact ground truth. Running every query both *with* and
*without* its filters lets the suite measure the **delta** from pushing an
attribute into a filter versus leaving it in the embedding query — under two
lenses:

- **Geometric correctness** — Recall@k against exact k-NN.
- **Task success (KIS)** — rank of the known target item, summarised as MRR.

## How it works

A benchmark run has two halves that meet at scoring:

```
OFFLINE / ORACLE (system-agnostic, exact)      ONLINE / SYSTEM UNDER TEST
queryset/queries/*.json                        data/benchmark.jsonl
   │ queryset/build.py                             │
   ▼                                               ▼   adapter.setup()      (per run)
data/benchmark.jsonl ──┐                     Runner(adapter, encoder).run()
   │ oracle/build_gt.py │                          │   adapter.search(vec, filters, k)
   ▼ (exact kNN; ground │                          ▼
     truth written into │                     data/raw_results.<sys>.jsonl
     each item's block)  │                         │
   └──────────┬─────────┘                          │
              ▼                                     │
        Analyzer().analyze(raw, items)  ◄──────────┘
              ▼
        data/metrics.<sys>.jsonl   (Recall@k, MRR, filtered-vs-unfiltered Δ)
```

The **oracle** computes what the correct answer *is* (exact search, independent of
any system). Each **adapter** answers the same queries through a real index. Their
predicate translations are deliberately separate code paths — the gap between the
oracle's exact result and a system's approximate one is precisely what the
benchmark reports.

Two contracts hold the design together:

- a shared **logical schema** every system must be able to answer queries against, and
- a shared **API** (`VectorDBAdapter`) every system implements.

How a system *physically* stores the data is up to its adapter — a system with
native joins can keep the schema normalized; one without can denormalize into a
single collection. That mapping lives in the adapter, on purpose, because it is
part of what the benchmark compares.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                    # core harness only
uv sync --extra pgvector --extra encode    # to run against pgvector

# 1. compile the authored query set -> data/benchmark.jsonl
uv run python queryset/build.py

# 2. enrich it with exact ground truth (needs the V3C DB + a GPU; see oracle/)
sbatch build_gt.sh --scene-threshold 0.10 --object-threshold 0.30

# 3. run a system and score it
uv run python run_benchmark.py --system pgvector
```

Optional dependency groups keep the heavy toolchains out of a light install:
`pgvector` (DB driver), `encode` (embedding model), `oracle` (everything the
ground-truth build needs), `viz` (plots).

## Adding your own system

Implement three methods on `VectorDBAdapter` (`frame/core/adapter.py`):

```python
class MyAdapter(VectorDBAdapter):
    name = "mysystem"

    def load_data(self, dataset: Dataset) -> None:
        # ONE-TIME ingest. Materialize the shared logical schema (a Tier-2
        # canonical shard: parquet tables + an embeddings h5) into your system's
        # own physical layout, then build the vector index. Must be idempotent.
        ...

    def setup(self) -> None:
        # PER-RUN. Connect, verify the index exists, apply search-time knobs.
        # Do NOT ingest here — raise if the data is missing.
        ...

    def search(self, query_vector, filters, k) -> list[str]:
        # translate `filters` (AND-ed abstract predicates; empty == no filter)
        # into your native query, run filtered k-NN, return k ids ranked best-first.
        ...
```

`load_data()` is where the multi-table workaround lives: pgvector loads the
normalized tables and JOINs; a system without joins must denormalize the same
neutral files into one flat collection. Keeping it separate from `setup()` means
a benchmark run never pays (or hides) a multi-million-row ingest.

Ingest once, then run:

```bash
sbatch load_dataset.sh --dataset data/canonical/v3c1   # one-time, per system
sbatch load_dataset.sh --check                         # verify your load_data()
sbatch run_benchmark.sh --system mysystem              # per run
```

Register it in `run_benchmark.py` + `scripts/load_dataset.py` and run. The shared `Runner` (drives the queries,
all four conditions, timings) and `Analyzer` (scores against the oracle ground truth)
are reused unchanged, so every system is measured the same way.

## Repository layout

```
frame/                     harness package
  core/    schema · dataset (Tier-2 handle) · adapter (ABC) · runner · analyzer · encode
  adapters/  pgvector · (add your own)
queryset/                  authored query set (source of truth) + build.py
  queries/*.json
oracle/                    build_gt.py — exact ground-truth computation
scripts/                   one-off analysis probes (profile_queryset, author_probe) + SLURM wrappers
tests/                     unit tests for the harness core (run with `uv run pytest`)
data/                      generated artifacts (gitignored)
logs/                      SLURM job output *.out (gitignored)
build_gt.sh                batch job for the oracle
run_benchmark.py / .sh     run + score a system end-to-end
```

All SLURM wrappers write their `.out` to `logs/` and are submitted from the repo
root (e.g. `sbatch build_gt.sh …`, `sbatch scripts/author_probe.sh …`).

## Design notes

- **Query set is the source of truth.** `queryset/queries/*.json` are authored by
  hand; `queryset/build.py` compiles and validates them into `data/benchmark.jsonl`.
  Ground truth is filled in place by the oracle — it lives in each item's `computed`
  block, so there is a single artifact rather than a separate ground-truth file.
- **Fairness.** All systems share one embedding encoder and one set of filter
  thresholds, so every system searches the same query vectors over the same filtered
  universe; only the retrieval/filtering under test varies.
- **One deep run.** Each query retrieves a large `k` once; the analyzer derives
  Recall@k at smaller cutoffs from that single ranked list.
- **The 2×2 condition matrix.** Every query runs four cells — `{raw_query_text,
  vector_query} × {predicate, no predicate}` — so the two deltas are separable: what
  pushing an attribute into a filter costs or buys, and what isolating the semantic
  remainder does. Each cell is scored against **its own** exact oracle answer. Items
  with no predicate run only the two no-filter cells (the filter cells would be the
  same search). Conditions are defined once in `frame/core/schema.py:CONDITIONS`.
- **Aggregates use one common subset.** Cross-condition numbers cover only items
  scorable in *every* condition, because the no-filter cells are scorable for items
  the filter cells are not — averaging each over its own subset would compare
  different query sets and read the difference as a condition effect.
- Adapters return **ranked ids only** — sufficient for Recall@k and MRR.

## Tests

The pure-logic core (schema, analyzer, runner, profiler, adapter contract, caching
encoder, and the query-set validator) is covered by a fast unit suite:

```bash
uv sync            # core harness + pytest (dev group); no heavy extras needed
uv run pytest
```

The suite needs no live database, GPU, or model download — the DB adapter
(`pgvector` → Postgres), the real `SiglipEncoder` (torch), and the oracle GT build
(HPC/GPU) are integration concerns and are out of scope here. CI runs the same
command on every push to `main` and every PR (`.github/workflows/tests.yml`).

## Status

Early. The shared harness and a pgvector adapter are in place; the query set and
ground-truth pipeline run against V3C. Additional systems and a larger query set
are in progress.

## License

TBD.

# FRAME

FRAME is a benchmark for filtered approximate nearest-neighbor (ANN) search. It
measures how vector databases behave when a similarity query is combined with
structured filters, using real Known-Item-Search (KIS) queries over the
[V3C](https://videobrowsershowdown.org/) video collection instead of synthetic
predicates at swept selectivities.

The queries come from interactive-retrieval sessions. Each one is split into a
semantic part that goes to an embedding model and a set of filter predicates
(scene, object, in-frame text), and every part has exact ground truth. Running a
query with and without its filters shows the effect of pushing an attribute into
a filter rather than leaving it in the embedding query. FRAME reports two things
about that effect:

- **Recall@k** against exact k-NN (is the approximate index still finding the
  right neighbors?).
- **Task success**, the rank of the known target item, summarized as MRR (does
  the filter help or hurt the actual search task?).

The suite currently ships adapters for **pgvector** and **ChromaDB**, and a
data-prep pipeline that builds the V3C dataset from raw shards.

## Install

FRAME uses [uv](https://docs.astral.sh/uv/). The core harness has a light
dependency footprint; the heavy toolchains live behind optional extras so you
only pull what a given task needs.

```bash
uv sync                                    # core harness + tests
uv sync --extra pgvector --extra encode    # run against pgvector
uv sync --extra chroma   --extra encode    # run against ChromaDB
```

| extra | pulls in | needed for |
|---|---|---|
| `pgvector` | psycopg2 | the pgvector adapter |
| `chroma` | chromadb (>=1.5) | the Chroma adapter |
| `encode` | torch, transformers | the shared SigLIP encoder |
| `oracle` | torch, transformers, psycopg2 | the ground-truth build |
| `viz` | matplotlib | the plotting scripts |

## Quickstart

```bash
# 1. compile the authored query set into data/benchmark.jsonl
uv run python queryset/build.py

# 2. fill in exact ground truth (needs the V3C DB and a GPU; see oracle/)
sbatch build_gt.sh --scene-threshold 0.10 --object-threshold 0.30

# 3. run a system and score it
uv run python run_benchmark.py --system pgvector
```

`run_benchmark.py` runs every query in all four conditions (see below), scores the
ranked ids against the oracle, and writes `data/metrics.<system>.jsonl`. Retrieval
depth is a swept axis rather than a single deep run: the suite sweeps `k` and
`ef_search` directly (`--k-grid`, `--ef-grid`, or `--iterative-scan sweep`) so the
approximate path is measured at the depth it actually runs at, avoiding the plan
bias a cost-based planner would show if one deep run were truncated after the
fact. The default operating point is a moderate `k`, with `k=1000` kept as a
recall ceiling. Runs record latencies with warmup and repeats, and refuse to score
results produced against a different query set or harness contract unless you pass
`--allow-mismatch`. See `uv run python run_benchmark.py --help` for the full set of
knobs.

On the cluster, submit `run_benchmark.sh` and `build_gt.sh` from the repo root
with `sbatch`; both write their output to `logs/`.

## How it works

A run has an offline half that computes what the correct answer *is*, and an
online half that asks each system the same questions through a real index. They
meet at scoring.

The **offline half** is the oracle, and it is system-agnostic. `queryset/build.py`
compiles the authored `queryset/queries/*.json` into `data/benchmark.jsonl`, and
`oracle/build_gt.py` then does exact k-NN over the corpus and writes the ground
truth back into each item's own block. Because this search is exact and independent
of any system under test, its answer is the reference every system is measured
against.

The **online half** is the system under test. `Runner(adapter, encoder).run()`
reads the same `data/benchmark.jsonl`, calls the adapter's `setup()` once per run,
and for every query calls `adapter.search(vec, filters, k)` through the system's
real index and filter translation, writing the ranked ids to
`data/raw_results.<sys>.jsonl`.

The two halves **meet at scoring**: `Analyzer().analyze(raw, items)` reads the raw
results and the oracle's ground truth together and emits
`data/metrics.<sys>.jsonl` (Recall@k, MRR, and the filtered-vs-unfiltered Δ). The
gap between the exact reference and what each system's index actually returned is
what the benchmark reports.

Two contracts hold this together: a shared logical schema every system answers
queries against, and a shared `VectorDBAdapter` API every system implements. How a
system physically stores the data is left to its adapter. A system with native
joins can keep the schema normalized; one without joins can denormalize into a
single collection. That mapping is part of what FRAME compares, so it belongs in
the adapter rather than in the harness.

### The four conditions

Every query runs a 2×2 matrix: `{raw query text, semantic-only text} ×
{filter, no filter}`. This separates two effects. The filter/no-filter axis shows
what pushing an attribute into a filter costs or buys; the text axis shows what
isolating the semantic remainder does. Each cell is scored against its own exact
oracle answer. Queries with no predicate only run the two no-filter cells.
Cross-condition aggregates are computed over the items scorable in every
condition, so the numbers compare the same set of queries rather than reading a
difference in coverage as a condition effect.

Some fairness invariants are baked in so systems are measured the same way: one
shared embedding encoder, and one set of extraction thresholds pinned at ingest
(scene probability 0.10, object confidence 0.30) so every system searches the same
filtered universe. The conditions are defined once in `frame/core/schema.py`.

## Building the dataset

Adapters ingest a backend-neutral **canonical dataset** rather than raw video:
per-shard parquet tables plus a `keyframes_embeddings.h5`, under
`data/canonical/<shard>/`. Each keyframe carries one 768-dim SigLIP embedding; the
filterable metadata comes from content-extraction passes (Places365 scene labels,
OWLv2 object detections, EasyOCR text). The query set targets V3C1
(~1.08M keyframes), but the retrieval corpus is the union of all three V3C shards
(~4.14M keyframes), so every query is scored against the full distractor pool.
There are two ways to produce the canonical form.

V3C1 already lives in a Postgres instance, so it is dumped directly:

```bash
sbatch export_v3c.sh
```

V3C2 and V3C3 are only available as extracted keyframe shards, so the model
passes run over the files. The four passes are independent and resumable per
video, so they run in parallel:

```bash
sbatch scripts/prep_metadata.sh    # CPU:  videos / shots / keyframes parquet
sbatch scripts/prep_embed.sh       # GPU:  SigLIP embeddings
sbatch scripts/prep_detect.sh      # GPU:  OWLv2 object detections
sbatch scripts/prep_scenes.sh      # GPU:  Places365 scene labels
sbatch scripts/prep_ocr.sh         # GPU:  EasyOCR in-frame text
# once metadata and the four passes finish:
sbatch scripts/prep_consolidate.sh # CPU:  merge staging into canonical files
```

Pass a shard root and name to target another shard, e.g.
`sbatch scripts/prep_embed.sh ~/datasets/V3C/V3C3 v3c3`. The pass conventions
(id format, thresholds, vocab, weights) match the validated V3C1 set so shards
stay comparable; they live in `frame/prep/common.py`. See `scripts/README.md` for
the details, including the `NUM_SHARDS` pinning that keeps a resubmitted array job
from re-mapping videos.

## Adding a system

Implement three methods on `VectorDBAdapter` (`frame/core/adapter.py`):

```python
class MyAdapter(VectorDBAdapter):
    name = "mysystem"

    def load_data(self, dataset: Dataset) -> None:
        # One-time ingest. Read the canonical dataset (parquet + embeddings h5),
        # map it into your system's own layout, and build the vector index.
        # Must be idempotent.
        ...

    def setup(self) -> None:
        # Per run. Connect, check the index is there, apply search-time settings.
        # Do not ingest here; raise if the data is missing.
        ...

    def search(self, query_vector, filters, k) -> list[str]:
        # Translate `filters` (AND-ed predicates; empty means no filter) into your
        # native query, run filtered k-NN, and return k ids best-first.
        ...
```

`load_data()` is where a multi-table workaround lives. pgvector loads the
normalized tables and joins; a system without joins denormalizes the same files
into one collection. Keeping it separate from `setup()` means a benchmark run
never pays for, or hides, a multi-million-row ingest.

```bash
sbatch load_dataset.sh --dataset data/canonical/v3c1   # one-time, per system
sbatch load_dataset.sh --check                         # verify your load_data()
sbatch run_benchmark.sh --system mysystem              # per run
```

Register the adapter in `run_benchmark.py` and `scripts/load_dataset.py`. The
shared `Runner` and `Analyzer` are reused unchanged, so every system runs the
same queries and is scored the same way.

## Repository layout

```
frame/                     harness package
  core/    schema · dataset · adapter (ABC) · runner · analyzer · encode · sweep
  adapters/  pgvector (normalized, joins) · chroma (denormalized, one collection)
  prep/    shared building blocks for the V3C model passes
queryset/                  authored query set (source of truth) + build.py
  queries/*.json
oracle/                    build_gt.py — exact ground-truth computation
scripts/                   prep pipeline, analysis probes, plots, SLURM wrappers
tests/                     unit tests for the harness core
data/                      generated artifacts (gitignored)
logs/                      SLURM job output (gitignored)
build_gt.sh                oracle batch job
run_benchmark.py / .sh     run and score a system end-to-end
```

`scripts/` holds three kinds of thing: the `prep_*` data pipeline, one-off
diagnostic probes (query-set profiling, per-label selectivity), and the `plot_*`
figure scripts. The plots need only JSON and matplotlib, so they run locally
after pulling the artifacts. See `scripts/README.md`.

## Versioning

`queryset/queryset.json` holds a hand-set semver, bumped when the query set
changes:

| bump | when | effect on existing results |
|---|---|---|
| MAJOR | a query's meaning changed, items removed, or ground truth recomputed under different parameters | void |
| MINOR | items added, nothing else | still valid; scoring restricts to the shared items |
| PATCH | nothing result-affecting (notes, status, flags) | unaffected |

`queryset/build.py` also digests the result-affecting contents and warns if they
changed while the semver did not, since a hand-set version eventually gets
forgotten. Provenance fields (notes, source, status, a filter's vocab or verified
flags) sit outside the digest, so fixing a typo does not invalidate a finished
run. Because the digest separates the authored half from the ground-truth half, a
rebuild carries ground truth forward for every query that did not change and drops
it only for those that did.

Each results file records what it was measured against: the benchmark version
(the query set and its ground truth, e.g. `v3c1/1.0.0+d9ac09a6e5c6`) and the
harness contract. The Analyzer refuses to score a mismatch unless you pass
`--allow-mismatch`. See `frame/core/version.py`.

## Tests

```bash
uv sync
uv run pytest
```

The unit suite covers the pure-logic core: schema, analyzer, runner, profiler,
the adapter contract, the caching encoder, and the query-set validator. It needs
no live database, GPU, or model download; the real adapters, the torch encoder,
and the oracle build are integration concerns and are out of scope. CI runs the
same command on every push and PR to `main` (`.github/workflows/tests.yml`).

## About

FRAME was built for a master's thesis on filtered-ANN search in vector databases,
using real Video Browser Showdown workloads to study how systems handle complex
filtered queries. The query set is a single-annotator draft of around 40 KIS tasks
derived from VBS logs (scene, object, and unfiltered workloads), and a query set
spanning V3C2/V3C3 targets is future work, so expect rough edges.

## License

MIT. See [LICENSE](LICENSE).

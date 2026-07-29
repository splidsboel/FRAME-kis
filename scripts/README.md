# scripts/ — one-off analysis probes

Ad-hoc probes and their SLURM wrappers, kept out of the repo root so the core
pipeline (`run_benchmark.py`, `build_gt.sh`, `frame/`, `queryset/`, `oracle/`)
stays uncluttered. These are analysis tools, **not** part of the benchmark
pipeline — they read the DB / the enriched `data/benchmark.jsonl` and print
diagnostics.

Submit from the **repo root** (so `logs/` and `data/` resolve), e.g.:

```bash
sbatch scripts/profile_queryset.sh                       # cutover plan/selectivity profile
sbatch scripts/author_probe.sh                           # per-label selectivity + per-target labels
sbatch scripts/author_probe.sh --unions candidates.json  # verify chosen disjunction selectivities
```

| script | what it does |
|---|---|
| `profile_queryset.py` / `.sh` | where each real VBS filter lands on pgvector's exact↔approximate cutover (plan, estimated vs true selectivity, near-query pass-rate). See the vault note *FRAME — pgvector planner split*. |
| `author_probe.py` / `.sh` | DB inputs for authoring disjunctive KIS queries: global per-label selectivity, each target's carried labels, and a `--unions` mode that verifies a candidate hedge's union selectivity + target-pass. |

Both resolve the repo root as the parent of `scripts/`, so `data/` paths and the
`frame` import work regardless of where they physically sit. SLURM `.out` files
land in `logs/` (gitignored).

---

## V3C data-prep pipeline (`prep_*` + `export_v3c`)

These **are** pipeline (not probes): they build the backend-neutral **canonical
dataset** (`data/canonical/<shard>/`: parquet + `keyframes_embeddings.h5`) that
each adapter's `load_data()` ingests. Design: vault note *Data pipeline and
adapter load refactor*; shared code in `frame/prep/common.py` (single source of
truth for the schemas). No postgres needed except `export_v3c`.

- **V3C1 (already in pg)** → dump the DB: `sbatch export_v3c.sh` (repo root).
- **V3C2 / V3C3 (files only)** → run the model passes over the extracted shard:

```bash
sbatch scripts/prep_metadata.sh                 # CPU: videos/shots/keyframes.parquet
sbatch scripts/prep_embed.sh                    # GPU array: SigLIP -> _staging/embed/*.npz
sbatch scripts/prep_detect.sh                   # GPU array: OWLv2  -> _staging/detect/*.parquet
sbatch scripts/prep_scenes.sh                   # GPU array: Places365 -> _staging/scenes/*.parquet
sbatch scripts/prep_ocr.sh                      # GPU array: EasyOCR -> _staging/ocr/*.parquet
# ...after metadata + the four passes finish:
sbatch scripts/prep_consolidate.sh              # CPU: _staging/* -> canonical single files + MANIFEST
# V3C3: pass the shard, e.g.  sbatch scripts/prep_embed.sh ~/datasets/V3C/V3C3 v3c3
```

The four model passes are independent (run in parallel) and resumable per video
(a preempted array task skips videos whose staging file exists). `--num-shards`
is taken from the array size — keep `#SBATCH --array` in sync.

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

#!/usr/bin/env bash
# sweep_kef.sh — launch the k×ef sweep on BOTH systems (LEAN conditions).
#
# Thin submit-time wrapper, NOT itself a SLURM job: it sbatch-es run_benchmark.sh
# once per system, so each cell reuses that script's node-local staging + in-job
# postgres harness unchanged (no duplication). Run from the repo root on `ssh hpc3`:
#
#     ./sweep_kef.sh                    # both systems, default grid
#     SYSTEMS=pgvector ./sweep_kef.sh   # one system
#     EF_GRID=100,200,500,1000 SYSTEMS=chroma ./sweep_kef.sh   # coarser ef for chroma
#
# Needs the GT-enriched data/benchmark.jsonl (build_gt.sh) and, for chroma, its
# loaded persist dir (chroma_load.sh) — same prerequisites as run_benchmark.sh.
#
# Grid: k <= ef <= 1000 (both engines require ef >= k). At k in {10,25,50,100} and
# ef in {10,25,50,100,200,500,1000} that is 22 legal cells per system. All cutoffs
# stay <= 100 = the oracle GT depth, so NO GT rebuild is needed.
#
# NOTE on chroma cost: chroma's filtered latency is the predicate/allow-list cost,
# which ef does NOT change (ef only moves the cheap HNSW walk). Its ef axis is
# therefore a recall curve at ~constant latency — a COARSER EF_GRID is usually
# enough and much cheaper. Set EF_GRID=100,200,1000 SYSTEMS=chroma for that.
#
# AFTER both jobs finish: pull data/metrics.<system>.k*.ef*.jsonl, then
#     python3 scripts/plot_kef_sweep.py --data data --out results/figures
# read the printed headline (+ data/headline_point.json), and run the headline
# FULL-2x2 cell per system at its ef@0.9-recall:
#     sbatch run_benchmark.sh --system <sys> --k-grid 50 --ef-grid <ef> --conditions full
set -euo pipefail

K_GRID="${K_GRID:-10,25,50,100}"
EF_GRID="${EF_GRID:-10,25,50,100,200,500,1000}"
SYSTEMS="${SYSTEMS:-pgvector chroma}"

for sys in $SYSTEMS; do
    echo "[sweep_kef] submitting $sys  k={$K_GRID}  ef={$EF_GRID}  (lean)"
    sbatch --job-name="frame_kef_${sys}" run_benchmark.sh \
        --system "$sys" \
        --k-grid "$K_GRID" --ef-grid "$EF_GRID" --conditions lean
done
echo "[sweep_kef] submitted. Watch: squeue -u \$USER ; logs/frame_run_*.out"

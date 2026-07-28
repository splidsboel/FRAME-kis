#!/usr/bin/env bash
#SBATCH --job-name=frame_profile
#SBATCH --partition=cores_any
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/frame_profile_%j.out

# Selectivity / plan profile of the authored query set. Same in-job postgres +
# conda `embeddings` setup as run_benchmark.sh (read that for details). Reads the
# GT-enriched data/benchmark.jsonl produced by build_gt.sh, so run that FIRST —
# the near-query pass-rate needs geometric_gt_vec_nofilter from the GT.
#
# Submit from the repo root:
#     sbatch scripts/profile_queryset.sh                          # defaults: --system pgvector
#     sbatch scripts/profile_queryset.sh --k 1000 --near-query-n 100
# Extra args pass straight through to profile_queryset.py.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"   # run `sbatch` from the FRAME-kis root
SIF="$HOME/containers/pgvector-pg16.sif"
PGDATA="$HOME/pgdata"
PGSOCKET="/tmp/pg_${SLURM_JOB_ID:-$$}"
mkdir -p "$PGSOCKET"

echo "[$(date)] Starting postgres..."
apptainer exec --bind /dev/shm --bind /tmp --bind "$PGSOCKET:$PGSOCKET" "$SIF" \
    postgres -D "$PGDATA" -k "$PGSOCKET" -c listen_addresses='' -c logging_collector=off &
PG_PID=$!

for i in $(seq 1 60); do
    if apptainer exec --bind /dev/shm --bind /tmp "$SIF" \
            pg_isready -h "$PGSOCKET" -U postgres -q 2>/dev/null; then
        echo "[$(date)] postgres ready (${i}s)"; break
    fi
    sleep 1
done

module load Anaconda3
set +u
source activate embeddings
set -u

export PGHOST="$PGSOCKET"
export PGUSER="postgres"
export PGDATABASE="postgres"

cd "$PROJECT_DIR"   # so `import frame` resolves and data/ paths line up
echo "[$(date)] Running profile_queryset.py $* ..."
python3 -u scripts/profile_queryset.py "$@"

echo "[$(date)] Plotting query-set selectivity (best-effort; needs matplotlib) ..."
python3 -u scripts/plot_query_selectivity.py \
    --in data/profile.pgvector.jsonl --out results/figures \
    || echo "[warn] plotting skipped — run scripts/plot_query_selectivity.py locally with the viz extra."

echo "[$(date)] Stopping postgres..."
apptainer exec --bind /dev/shm --bind /tmp "$SIF" pg_ctl stop -D "$PGDATA" -m fast
wait "$PG_PID" 2>/dev/null || true
echo "[$(date)] Done."

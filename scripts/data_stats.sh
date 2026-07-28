#!/usr/bin/env bash
#SBATCH --job-name=frame_datastats
#SBATCH --partition=cores_any
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=logs/frame_datastats_%j.out

# Corpus / metadata characterization of the V3C oracle DB (statistik over data og
# metadata — Omar, meeting 2026-07-20). Same in-job postgres pattern as
# author_probe.sh / profile_queryset.sh — read those for details. Pure SQL for the
# stats; the plotting step needs matplotlib (in the `embeddings` conda env, or run
# scripts/plot_data_stats.py locally after pulling data/data_stats.json).
#
# Submit from the repo root (so logs/, data/, results/ resolve):
#     sbatch scripts/data_stats.sh
#     sbatch scripts/data_stats.sh --top-n 50   # widen the co-occurrence label grid

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
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

cd "$PROJECT_DIR"   # so queryset/, data/, results/ paths line up
echo "[$(date)] Running scripts/data_stats.py $* ..."
python3 -u scripts/data_stats.py "$@"

echo "[$(date)] Plotting (best-effort; needs matplotlib) ..."
python3 -u scripts/plot_data_stats.py --in data/data_stats.json --out results/figures \
    || echo "[warn] plotting skipped — run scripts/plot_data_stats.py locally with the viz extra."

echo "[$(date)] Stopping postgres..."
apptainer exec --bind /dev/shm --bind /tmp "$SIF" pg_ctl stop -D "$PGDATA" -m fast
wait "$PG_PID" 2>/dev/null || true
echo "[$(date)] Done."

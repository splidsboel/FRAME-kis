#!/usr/bin/env bash
#SBATCH --job-name=frame_authorprobe
#SBATCH --partition=cores_any
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/frame_authorprobe_%j.out

# DB inputs for authoring disjunctive KIS queries (step 1 of the cutover work).
# Same in-job postgres pattern as profile_queryset.sh — read that for details.
# No GPU / conda needed: this probe is pure SQL (psycopg2 only).
#
# Submit from the repo root (so logs/ and data/ resolve):
#     sbatch scripts/author_probe.sh                          # survey: global sel + per-target labels
#     sbatch scripts/author_probe.sh --unions candidates.json # verify chosen union selectivities

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

cd "$PROJECT_DIR"   # so queryset/ and data/ paths line up
echo "[$(date)] Running scripts/author_probe.py $* ..."
python3 -u scripts/author_probe.py "$@"

echo "[$(date)] Stopping postgres..."
apptainer exec --bind /dev/shm --bind /tmp "$SIF" pg_ctl stop -D "$PGDATA" -m fast
wait "$PG_PID" 2>/dev/null || true
echo "[$(date)] Done."

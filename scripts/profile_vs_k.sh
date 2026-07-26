#!/usr/bin/env bash
#SBATCH --job-name=frame_pvk
#SBATCH --partition=cores_any
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/frame_pvk_%j.out

# Plan choice vs retrieval depth k — the controlled confirmation of the cutover's
# k-dependence (see the vault: "FRAME — cutover sweep"). Same in-job postgres +
# conda `embeddings` setup as profile_queryset.sh (read that for details). Reads the
# GT-enriched data/benchmark.jsonl produced by build_gt.sh. Cheap (EXPLAIN only, no
# query execution), so 30 min is plenty.
#
# Submit from the repo root:
#     sbatch scripts/profile_vs_k.sh                     # defaults: k=50,100,250,1000
#     sbatch scripts/profile_vs_k.sh --k-grid 25,50,100,250,500,1000
# Extra args pass straight through to profile_vs_k.py.

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
echo "[$(date)] Running profile_vs_k.py $* ..."
python3 -u scripts/profile_vs_k.py "$@"

echo "[$(date)] Stopping postgres..."
apptainer exec --bind /dev/shm --bind /tmp "$SIF" pg_ctl stop -D "$PGDATA" -m fast
wait "$PG_PID" 2>/dev/null || true
echo "[$(date)] Done."

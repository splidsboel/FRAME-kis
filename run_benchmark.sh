#!/usr/bin/env bash
#SBATCH --job-name=frame_run
#SBATCH --partition=cores_any
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=frame_run_%j.out

# Run a system end-to-end and score it. Like build_gt.sh it needs the V3C postgres
# (started in-job via Apptainer) and the conda `embeddings` env (torch/transformers/
# psycopg2 are already there — no uv on the HPC). Encoding ~56 short texts runs fine
# on CPU, so this uses cores_any, not a GPU node. Reads the GT-enriched
# data/benchmark.jsonl produced by build_gt.sh, so run that FIRST.
#
# Submit from the repo root:
#     sbatch run_benchmark.sh                 # defaults: --system pgvector
#     sbatch run_benchmark.sh --system pgvector --k 1000
# Extra args pass straight through to run_benchmark.py.

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

# libpq env so the PgvectorAdapter's psycopg2.connect() finds the in-job socket
# as the postgres superuser (auth=trust over the unix socket).
export PGHOST="$PGSOCKET"
export PGUSER="postgres"
export PGDATABASE="postgres"

cd "$PROJECT_DIR"   # so `import frame` resolves and data/ paths line up
echo "[$(date)] Running run_benchmark.py $* ..."
python3 -u run_benchmark.py "$@"

echo "[$(date)] Stopping postgres..."
apptainer exec --bind /dev/shm --bind /tmp "$SIF" pg_ctl stop -D "$PGDATA" -m fast
wait "$PG_PID" 2>/dev/null || true
echo "[$(date)] Done."

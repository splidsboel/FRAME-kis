#!/usr/bin/env bash
#SBATCH --job-name=frame_load
#SBATCH --partition=cores_any
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/frame_load_%j.out

# Step 3 of [[Data pipeline and adapter load refactor]]: ingest a Tier-2 canonical
# shard into a system under test via adapter.load_data(). Same in-job postgres +
# conda `embeddings` pattern as run_benchmark.sh / export_v3c.sh.
#
# Submit from the FRAME-kis repo root, from `ssh hpc3`:
#     sbatch load_dataset.sh --dataset data/canonical/v3c1
#     sbatch load_dataset.sh --dataset data/canonical/v3c2 --force
# Extra args pass straight through to scripts/load_dataset.py.
#
# Wall time is dominated by the HNSW build. For the UNION load (v3c1+2+3 =
# ~4.1M 768-d vectors) this is severe: with the postgres-default maintenance_work_mem
# (64MB) pgvector builds the graph on disk in tiny passes and does NOT finish in 12h
# (job 102410 timed out there). We fix it two ways below:
#   * maintenance_work_mem = 24GB  -> the whole graph (~13GB of vectors + links) fits
#     in memory, so it's an in-memory build, not the on-disk crawl.
#   * max_parallel_maintenance_workers = 8 -> pgvector builds HNSW in parallel.
# Both are passed to the adapter via FRAME_* env vars (see _build_indexes); postgres
# is started with matching server ceilings so the session can actually get workers.
# The job is safe to re-run: load_datasets() skips the reload when the rows are
# already present and resumes straight at the index build (only the HNSW was missing).

set -euo pipefail

# HNSW build tuning. maintenance_work_mem is a single shared budget for the build
# (not per-worker); keep it well under --mem. Parallel workers <= cpus-per-task.
export FRAME_MAINTENANCE_WORK_MEM="${FRAME_MAINTENANCE_WORK_MEM:-24GB}"
export FRAME_INDEX_PARALLEL_WORKERS="${FRAME_INDEX_PARALLEL_WORKERS:-8}"

# Admin-mandated in every job script; the postgres socket stays on node-local
# /tmp because $HOME/tmp is NFS.
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SIF="$HOME/containers/pgvector-pg16.sif"
PGDATA="$HOME/pgdata"
PGSOCKET="/tmp/pg_${SLURM_JOB_ID:-$$}"
mkdir -p "$PGSOCKET"

echo "[$(date)] Starting postgres..."
# Raise the parallel-worker ceilings at the server level so the session's
# max_parallel_maintenance_workers (set in _build_indexes) is actually honoured --
# the defaults (max_worker_processes/max_parallel_workers = 8, maintenance = 2)
# would silently cap the HNSW build back to 2 workers.
apptainer exec --bind /dev/shm --bind /tmp --bind "$PGSOCKET:$PGSOCKET" "$SIF" \
    postgres -D "$PGDATA" -k "$PGSOCKET" -c listen_addresses='' -c logging_collector=off \
    -c max_worker_processes=20 -c max_parallel_workers=16 \
    -c max_parallel_maintenance_workers=8 &
PG_PID=$!

# Shut postgres down cleanly however this job ends -- scancel, an error under
# `set -e`, or success. A killed postgres leaves $PGDATA dirty, and the *next*
# job silently pays for it in crash recovery.
stop_pg() {
    apptainer exec --bind /dev/shm --bind /tmp "$SIF" \
        pg_ctl stop -D "$PGDATA" -m fast 2>/dev/null || true
    wait "$PG_PID" 2>/dev/null || true
}
trap stop_pg EXIT
trap 'exit 143' INT TERM   # let the EXIT trap do the shutdown, then die

# Wait generously: if a previous job was killed (scancel, or an error under
# `set -e`), postgres starts by running crash recovery, which fsyncs the whole
# 20G data directory -- minutes, not the ~4s of a clean start. And fail LOUDLY on
# timeout: falling through a silent timeout only defers the error to the next
# psql/python call, where it reads as an unrelated connection failure.
pg_ready=0
for i in $(seq 1 300); do
    if apptainer exec --bind /dev/shm --bind /tmp "$SIF" \
            pg_isready -h "$PGSOCKET" -U postgres -q 2>/dev/null; then
        echo "[$(date)] postgres ready (${i} attempts)"; pg_ready=1; break
    fi
    sleep 1
done
if [ "$pg_ready" -ne 1 ]; then
    echo "[$(date)] FATAL: postgres never became ready -- see its log above." >&2
    echo "  'database system was interrupted' means crash recovery was still running." >&2
    exit 1
fi

module load Anaconda3
set +u
source activate embeddings
set -u

# pyarrow + h5py aren't in the embeddings env; add them (wheels, no build). No-op if present.
echo "[$(date)] Ensuring pyarrow + h5py are installed..."
python3 -c "import pyarrow, h5py" 2>/dev/null || pip install --quiet pyarrow h5py

export PGHOST="$PGSOCKET"
export PGUSER="postgres"
export PGDATABASE="postgres"

cd "$PROJECT_DIR"
if [ "${1:-}" = "--check" ]; then
    # Integration check for load_data(): synthetic shard -> scratch DB -> verify.
    # Touches nothing real; see scripts/check_load_roundtrip.py.
    shift
    echo "[$(date)] Running scripts/check_load_roundtrip.py $* ..."
    python3 -u scripts/check_load_roundtrip.py "$@"
else
    echo "[$(date)] Running scripts/load_dataset.py $* ..."
    python3 -u scripts/load_dataset.py "$@"
fi

echo "[$(date)] Stopping postgres..."
stop_pg
echo "[$(date)] Done."

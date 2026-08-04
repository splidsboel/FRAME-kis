#!/usr/bin/env bash
#SBATCH --job-name=frame_load
#SBATCH --partition=cores_any
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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
# Wall time is dominated by the HNSW build over ~1M vectors, not the COPY. 12h is
# generous; the job is safe to re-run because load_data() skips tables whose row
# counts already match.

set -euo pipefail

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
apptainer exec --bind /dev/shm --bind /tmp --bind "$PGSOCKET:$PGSOCKET" "$SIF" \
    postgres -D "$PGDATA" -k "$PGSOCKET" -c listen_addresses='' -c logging_collector=off &
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

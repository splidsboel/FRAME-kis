#!/usr/bin/env bash
#SBATCH --job-name=frame_sweep
#SBATCH --partition=cores_any
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/frame_sweep_%j.out

# Dense recall-vs-selectivity sweep across the exact↔approximate cutover. Same
# in-job postgres + conda `embeddings` setup as run_benchmark.sh / profile_queryset.sh
# (read those for details). Reads the GT-enriched data/benchmark.jsonl produced by
# build_gt.sh, so run that FIRST — recall is scored against gt_filtered.
#
# Submit from the repo root:
#     sbatch scripts/sweep_cutover.sh                         # defaults (k=100, full grid)
#     sbatch scripts/sweep_cutover.sh --plan auto             # planner's real choice only
#     sbatch scripts/sweep_cutover.sh --k 100 --ef-search 100,200,500,1000
# Extra args pass straight through to sweep_cutover.py.

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

export PGHOST="$PGSOCKET"
export PGUSER="postgres"
export PGDATABASE="postgres"

cd "$PROJECT_DIR"   # so `import frame` resolves and data/ paths line up
echo "[$(date)] Running sweep_cutover.py $* ..."
python3 -u scripts/sweep_cutover.py "$@"

echo "[$(date)] Stopping postgres..."
stop_pg
echo "[$(date)] Done."

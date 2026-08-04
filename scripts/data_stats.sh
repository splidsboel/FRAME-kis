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

cd "$PROJECT_DIR"   # so queryset/, data/, results/ paths line up
echo "[$(date)] Running scripts/data_stats.py $* ..."
python3 -u scripts/data_stats.py "$@"

echo "[$(date)] Plotting (best-effort; needs matplotlib) ..."
python3 -u scripts/plot_data_stats.py --in data/data_stats.json --out results/figures \
    || echo "[warn] plotting skipped — run scripts/plot_data_stats.py locally with the viz extra."

echo "[$(date)] Stopping postgres..."
stop_pg
echo "[$(date)] Done."

#!/usr/bin/env bash
#SBATCH --job-name=build_gt
#SBATCH --partition=acltr
#SBATCH --exclude=cn12   # cn12 (acltr) has NO apptainer -> postgres can't start (job 104538, 2026-08-18)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/build_gt_%j.out

# Enrich data/benchmark.jsonl with DB-computed ground truth. Needs BOTH a GPU
# (SigLIP text encoder) and the V3C postgres, so it starts the in-job postgres,
# then runs the oracle. Submit from the repo root:  sbatch build_gt.sh [args]
#
# Args after the script name pass straight to oracle/build_gt.py, e.g.:
#     sbatch build_gt.sh                                          # diagnostics pass
#     sbatch build_gt.sh --scene-threshold 0.10 --object-threshold 0.30
#
# EXCEPT --corpus <label>, which is peeled off and passed to queryset/build.py
# instead (it labels the benchmark identity, so union GT must be built with it):
#     sbatch build_gt.sh --corpus v3c1+2+3 --scene-threshold 0.10 --object-threshold 0.30
#
# Submit from `ssh hpc3` (never hpc.itu.dk). Stays on `acltr` on purpose: the
# `scavenge` desktop* nodes have a driver too old for torch cu121, so
# torch.cuda.is_available() silently comes back False there.

set -euo pipefail

# Admin-mandated in every job script (keeps temp off the shared system /tmp).
# The postgres socket below deliberately stays on node-local /tmp — $HOME/tmp is
# NFS, and a unix socket there would be slow and fragile.
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"   # run `sbatch` from the FRAME-kis root
SIF="$HOME/containers/pgvector-pg16.sif"
PGDATA="$HOME/pgdata"
PGSOCKET="/tmp/pg_${SLURM_JOB_ID:-$$}"
mkdir -p "$PGSOCKET"

# Fail FAST if this node lacks apptainer (some acltr nodes do -- cn12). Without this
# the missing-apptainer postgres start dies silently in the background and we only
# find out after the 300s pg_isready loop times out. A clear message in 1s beats that.
command -v apptainer >/dev/null 2>&1 || {
    echo "[$(date)] FATAL: apptainer not found on $(hostname) -- exclude this node (see #SBATCH --exclude)." >&2
    exit 1
}

echo "[$(date)] Starting postgres..."
# GT is exact -> every brute_knn/target_rank is an index-off SEQUENTIAL scan of
# the whole keyframes heap. On the full v3c1+2+3 corpus (~4.1M keyframes) the
# embedding column alone is ~13G, so with default shared_buffers (128M) each of
# the dozens of per-query scans re-reads the heap from NFS and a single query
# blows past the wall clock. Give postgres enough cache to keep the hot heap
# resident after the first scan (fits under the 32G job memory above); the rest
# then run in-memory. effective_cache_size just informs the planner.
apptainer exec --bind /dev/shm --bind /tmp --bind "$PGSOCKET:$PGSOCKET" "$SIF" \
    postgres -D "$PGDATA" -k "$PGSOCKET" -c listen_addresses='' -c logging_collector=off \
    -c shared_buffers=16GB -c effective_cache_size=24GB -c work_mem=256MB &
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

apptainer exec --bind /dev/shm --bind /tmp "$SIF" \
    psql -h "$PGSOCKET" -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;" postgres

module load Anaconda3
set +u
source activate embeddings
set -u

export PGHOST="$PGSOCKET"

# Peel --corpus off the args (it belongs to build.py, not build_gt.py); everything
# else passes straight through to the oracle.
BUILD_ARGS=()
GT_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --corpus)   BUILD_ARGS+=(--corpus "$2"); shift 2;;
        --corpus=*) BUILD_ARGS+=(--corpus "${1#*=}"); shift;;
        *)          GT_ARGS+=("$1"); shift;;
    esac
done

# Always recompile benchmark.jsonl from the authored queries/*.json first, so a
# stale/hand-edited benchmark.jsonl can never feed the GT run (queries/*.json is
# the single source of truth).
echo "[$(date)] Rebuilding data/benchmark.jsonl from queryset/queries/ ..."
python3 -u "$PROJECT_DIR/queryset/build.py" "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}"

echo "[$(date)] Running oracle/build_gt.py ${GT_ARGS[@]+"${GT_ARGS[@]}"} ..."
python3 -u "$PROJECT_DIR/oracle/build_gt.py" "${GT_ARGS[@]+"${GT_ARGS[@]}"}"

echo "[$(date)] Stopping postgres..."
stop_pg
echo "[$(date)] Done."

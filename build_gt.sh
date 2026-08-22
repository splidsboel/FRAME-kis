#!/usr/bin/env bash
#SBATCH --job-name=build_gt
# NODE TARGETING (2026-08-20). GT is exact -> every brute_knn/target_rank is an
# index-off SEQUENTIAL scan of the whole keyframes heap. On the full v3c1+2+3
# corpus (~4.1M keyframes, ~67 GB PGDATA) those scans off NFS crawl at ~1-2 MB/s;
# the previous acltr/no-stage version got through only 18/41 items in a 12 h wall
# (job 104679, timed out). So this now STAGES PGDATA onto node-local /scratch first,
# exactly like run_benchmark.sh, and pins to the big-/scratch IB nodes cn3/cn6 via
# cores_any + an exclude list. The SigLIP encoder only embeds ~80 short query texts
# once; that runs fine on CPU (torch auto-falls-back -- oracle/build_gt.py:424), so
# we do NOT request a GPU. Bonus: encoding GT's query vectors on CPU matches how
# run_benchmark.sh encodes them (also CPU/cores_any), so the GT and the sweep search
# with a byte-identical query embedding -- no device drift at the recall boundary.
#SBATCH --partition=cores_any
#SBATCH --exclude=cn4,cn5,cn7,cn12,cn16,cn17,cn18   # within cores_any -> leaves cn3,cn6
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/build_gt_%j.out

# Enrich data/benchmark.jsonl with DB-computed ground truth. Needs the V3C postgres
# (and the SigLIP text encoder, on CPU), so it stages + starts an in-job postgres,
# then runs the oracle. Submit from the repo root, from `ssh hpc3`:
#     sbatch build_gt.sh                                          # diagnostics pass
#     sbatch build_gt.sh --scene-threshold 0.10 --object-threshold 0.30
#
# EXCEPT --corpus <label>, which is peeled off and passed to queryset/build.py
# instead (it labels the benchmark identity, so union GT must be built with it):
#     sbatch build_gt.sh --corpus v3c1+2+3 --scene-threshold 0.10 --object-threshold 0.30
#
# Opt out of staging with FRAME_STAGE_PGDATA=0 (then it runs straight off NFS and
# the wall clock is NOT quotable -- the reason staging exists).

set -euo pipefail

# Admin-mandated in every job script (keeps temp off the shared system /tmp).
# The postgres socket below deliberately stays on node-local /tmp — $HOME/tmp is
# NFS, and a unix socket there would be slow and fragile.
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"   # run `sbatch` from the FRAME-kis root
SIF="$HOME/containers/pgvector-pg16.sif"
CANON_PGDATA="$HOME/pgdata"        # canonical DB (NFS, source of truth; loaded once)
PGSOCKET="/tmp/pg_${SLURM_JOB_ID:-$$}"
mkdir -p "$PGSOCKET"

# ── Node-local staging (the latency fix, shared with run_benchmark.sh) ──────────────
# Copy PGDATA to node-local /scratch so every brute scan does node-local I/O instead of
# ~1-2 MB/s random NFS reads. The copy is a ONE-TIME SEQUENTIAL read (fast over IB); the
# canonical NFS source stays pristine (any write lands in the disposable local copy).
# Everything staged is torn down on exit -- success, error under `set -e`, or scancel/
# timeout (the INT/TERM trap converts SIGTERM into a normal exit so the EXIT trap runs).
# A hard SIGKILL cannot run the trap; the orphan reaper in pick_stage_base cleans such
# leftovers on the next job on that node.
STAGED_DIRS=()            # node-local copies to remove on exit
STAGE_BASE=""
STAGE_BIND=()             # apptainer --bind for /scratch (set once PGDATA is staged)
pick_stage_base() {
    [ -n "$STAGE_BASE" ] && return 0
    local base
    for base in "/scratch/$USER" "/tmp/$USER"; do
        if mkdir -p "$base" 2>/dev/null && [ -w "$base" ]; then STAGE_BASE="$base"; break; fi
    done
    : "${STAGE_BASE:=/tmp/$USER}"
    # Reap OUR OWN orphaned stagings from jobs that were HARD-killed (a SIGKILL can't run
    # the cleanup trap). A staging dir whose trailing job id is no longer in the queue is
    # safe to delete -- this is what keeps node-local disk from filling up.
    local d jid
    for d in "$STAGE_BASE"/frame_*_*; do
        [ -d "$d" ] || continue
        jid="${d##*_}"
        case "$jid" in ''|*[!0-9]*) continue ;; esac       # skip non-numeric suffixes
        if [ -z "$(squeue -h -j "$jid" -o %i 2>/dev/null)" ]; then
            echo "[$(date)] reaping orphan staging $d (job $jid not in queue)" >&2; rm -rf "$d"
        fi
    done
}
# stage_local <src-dir> <tag>  ->  prints the node-local copy path on stdout
stage_local() {
    local src="$1" tag="$2" dst need_kb free_kb
    pick_stage_base
    dst="$STAGE_BASE/frame_${tag}_${SLURM_JOB_ID:-$$}"
    need_kb=$(du -sk "$src" | awk '{print $1}')
    free_kb=$(df -Pk "$STAGE_BASE" | awk 'NR==2{print $4}')
    echo "[$(date)] staging $src -> $dst (need $((need_kb/1048576))G, free $((free_kb/1048576))G on $STAGE_BASE)" >&2
    if [ "$free_kb" -lt "$((need_kb + need_kb/10))" ]; then
        echo "[$(date)] FATAL: not enough node-local space to stage $src." >&2; exit 1
    fi
    rm -rf "$dst"
    cp -a "$src" "$dst"
    STAGED_DIRS+=("$dst")
    printf '%s\n' "$dst"
}

# Single cleanup: stop postgres if we started it, then remove every node-local staging
# copy. Set the trap up-front so a cancel mid-copy is also handled.
started_pg=0
PG_PID=""
stop_pg() {
    apptainer exec --bind /dev/shm --bind /tmp ${STAGE_BIND[@]+"${STAGE_BIND[@]}"} "$SIF" \
        pg_ctl stop -D "$PGDATA" -m fast 2>/dev/null || true
    [ -n "$PG_PID" ] && { wait "$PG_PID" 2>/dev/null || true; }
}
cleanup() {
    [ "$started_pg" -eq 1 ] && stop_pg
    local d
    for d in ${STAGED_DIRS[@]+"${STAGED_DIRS[@]}"}; do
        [ -d "$d" ] || continue
        echo "[$(date)] removing node-local staging $d"; rm -rf "$d"
    done
}
trap cleanup EXIT
trap 'exit 143' INT TERM   # let the EXIT trap do the shutdown + cleanup, then die

# Fail FAST if apptainer can't actually RUN a container on this node -- BEFORE the ~67 GB
# staging copy, so we never copy to a node that can't start postgres anyway. A bare
# `command -v` misses user-namespace exhaustion, so RUNTIME-test with a throwaway exec.
if ! apptainer exec "$SIF" true >/dev/null 2>&1; then
    echo "[$(date)] FATAL: apptainer cannot run a container on $(hostname) " \
         "(missing, or user namespaces exhausted) -- exclude this node and resubmit." >&2
    exit 1
fi

# Stage PGDATA onto node-local disk (opt out with FRAME_STAGE_PGDATA=0 -> straight off NFS).
if [ "${FRAME_STAGE_PGDATA:-1}" = "1" ]; then
    PGDATA="$(stage_local "$CANON_PGDATA" pgdata)"
    rm -f "$PGDATA/postmaster.pid"   # never carry a stale lock into the copy
    chmod 700 "$PGDATA"
    echo "[$(date)] staged PGDATA to node-local disk: $PGDATA"
else
    PGDATA="$CANON_PGDATA"
    echo "[$(date)] FRAME_STAGE_PGDATA=0 -> running straight off NFS $PGDATA"
fi

# Apptainer only auto-binds $HOME, /tmp, /dev/shm, /proc, /sys and cwd. If PGDATA now
# lives on node-local /scratch, the container CANNOT SEE it unless we bind it in.
case "$PGDATA" in
    /scratch/*) STAGE_BIND=(--bind "$(dirname "$PGDATA")") ;;
esac

echo "[$(date)] Starting postgres..."
# GT is exact -> every brute_knn/target_rank is an index-off SEQUENTIAL scan of the
# whole keyframes heap. Staged on node-local disk the scans are already fast; the big
# shared_buffers then keeps the hot heap resident after the first scan so the dozens of
# per-query scans run in-memory. effective_cache_size just informs the planner.
apptainer exec --bind /dev/shm --bind /tmp ${STAGE_BIND[@]+"${STAGE_BIND[@]}"} --bind "$PGSOCKET:$PGSOCKET" "$SIF" \
    postgres -D "$PGDATA" -k "$PGSOCKET" -c listen_addresses='' -c logging_collector=off \
    -c shared_buffers=32GB -c effective_cache_size=48GB -c work_mem=256MB &
PG_PID=$!
started_pg=1

# Wait generously: if a previous job was killed (scancel, or an error under `set -e`),
# postgres starts by running crash recovery, which fsyncs the whole data directory --
# minutes, not the ~4s of a clean start. Fail LOUDLY on timeout: a silent fall-through
# only defers the error to the next psql/python call as a confusing connection failure.
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

apptainer exec --bind /dev/shm --bind /tmp ${STAGE_BIND[@]+"${STAGE_BIND[@]}"} "$SIF" \
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
# the single source of truth). GT already present for an UNCHANGED query is carried
# forward; only queries whose content changed lose it and get recomputed below.
echo "[$(date)] Rebuilding data/benchmark.jsonl from queryset/queries/ ..."
python3 -u "$PROJECT_DIR/queryset/build.py" "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}"

echo "[$(date)] Running oracle/build_gt.py ${GT_ARGS[@]+"${GT_ARGS[@]}"} ..."
python3 -u "$PROJECT_DIR/oracle/build_gt.py" "${GT_ARGS[@]+"${GT_ARGS[@]}"}"

echo "[$(date)] Stopping postgres + removing node-local staging..."
cleanup            # EXIT trap re-runs it as an idempotent no-op
echo "[$(date)] Done."

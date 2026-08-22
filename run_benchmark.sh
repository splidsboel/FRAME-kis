#!/usr/bin/env bash
#SBATCH --job-name=frame_run
# NODE TARGETING (updated 2026-08-19). BOTH systems now STAGE their on-disk state onto
# node-local /scratch to escape the NFS random-read latency that made filtered searches
# take minutes (pgvector: 100-700 s; chroma: ~35 min/filtered-query, ~7 h/item -- job
# 104552). /home is NFS over 44x 16TB SATA RAID10; HNSW graph traversal is RANDOM 8KB
# reads that crawl at ~1-2 MB/s there when the index doesn't fit in RAM. /scratch is only
# USER-WRITABLE on the big Infiniband nodes -- confirmed cn3 and cn6 (2.3 TB each, 100 Gbps
# IB, 192/384 GB RAM). The smaller nodes (cn14/15/16...) have a root-owned /scratch and a
# ~50 GB /tmp, too small for the ~67 GB DB. So we pin to cn3/cn6 via cores_any + an exclude
# list. We do NOT use --exclusive: cn3/cn6 are GPU nodes and bare --exclusive grabs their
# GPUs (rejected by QOS MaxGRESPerJob). Acceptable trade -- local-disk staging removes the
# NFS I/O noise that --exclusive used to guard against; residual co-tenancy is minor, and IB
# gives the fastest one-time staging copy. Opt out per system with FRAME_STAGE_PGDATA=0 /
# FRAME_STAGE_CHROMA=0 (then filtered latency is NOT quotable -- the reason staging exists).
#SBATCH --partition=cores_any
#SBATCH --exclude=cn4,cn5,cn7,cn12,cn16,cn17,cn18   # within cores_any -> leaves cn3,cn6
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/frame_run_%j.out

# Run a system end-to-end and score it. Reads the GT-enriched data/benchmark.jsonl
# produced by build_gt.sh (run that FIRST) and the conda `embeddings` env (torch/
# transformers/psycopg2 already there — no uv on the HPC). Encoding ~56 short texts
# runs fine on CPU, so this uses cores_any, not a GPU node.
#
# The system under test is chosen with --system (default pgvector); the infra it
# needs differs, so this script provisions it CONDITIONALLY:
#   * pgvector -> start an in-job postgres via Apptainer (the pgvector container)
#   * chroma   -> nothing to start; the embedded client reads its persist dir
# Both stage their on-disk state to node-local /scratch first (see the helpers below).
# So the same script serves both. Submit from the repo root, from `ssh hpc3`:
#     sbatch run_benchmark.sh                          # pgvector (default)
#     sbatch run_benchmark.sh --system pgvector --k 1000
#     sbatch run_benchmark.sh --system chroma          # needs chroma_load.sh first
# Extra args pass straight through to run_benchmark.py.
#
# If `embeddings` is ever lost, rebuild a replacement env with `sbatch setup_env.sh`.

set -euo pipefail

# Admin-mandated in every job script (keeps temp off the shared system /tmp).
# The postgres socket below deliberately stays on node-local /tmp — $HOME/tmp is
# NFS, and a unix socket there would be slow and fragile.
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"   # run `sbatch` from the FRAME-kis root

# ── Shared node-local staging (the latency fix, used by BOTH systems) ───────────────
# Copy an NFS on-disk state dir (PGDATA, or chroma's persist dir) to node-local /scratch
# so every query does node-local I/O instead of ~1-2 MB/s random NFS reads. The copy is a
# ONE-TIME SEQUENTIAL read (fast over IB); the canonical NFS source stays pristine (all
# benchmark writes land in the disposable local copy). Everything staged is torn down on
# exit — success, error under `set -e`, or scancel/timeout (the INT/TERM trap converts
# SIGTERM into a normal exit so the EXIT trap runs). A hard SIGKILL cannot run the trap;
# the orphan reaper in pick_stage_base cleans such leftovers on the next job on that node.
STAGED_DIRS=()            # node-local copies to remove on exit
STAGE_BASE=""
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

# Single cleanup for both systems: stop postgres if we started it, then remove every
# node-local staging copy. Set the trap up-front so it also covers chroma's staging (which
# happens after conda activation, below) and a cancel mid-copy.
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

# Which system? Scan the pass-through args (default pgvector) so we know whether to
# stand up postgres. Supports `--system chroma` and `--system=chroma`.
SYSTEM="pgvector"
_args=("$@")
for ((i = 0; i < ${#_args[@]}; i++)); do
    case "${_args[i]}" in
        --system)   SYSTEM="${_args[i+1]:-pgvector}" ;;
        --system=*) SYSTEM="${_args[i]#*=}" ;;
    esac
done
echo "[$(date)] system under test: $SYSTEM"

# ── pgvector: bring up the in-job postgres (Apptainer). Skipped for other systems ──
if [ "$SYSTEM" = "pgvector" ]; then
    SIF="$HOME/containers/pgvector-pg16.sif"
    CANON_PGDATA="$HOME/pgdata"        # canonical DB (NFS, source of truth; loaded once)
    PGSOCKET="/tmp/pg_${SLURM_JOB_ID:-$$}"
    mkdir -p "$PGSOCKET"

    # Fail FAST if apptainer can't run a container here -- BEFORE the 66GB staging copy,
    # so we never copy to a node that can't start postgres anyway. Two acltr failure modes
    # seen 2026-08-18: cn12 has no apptainer; cn4 had user namespaces exhausted. A bare
    # `command -v` misses the second, so runtime-test with a throwaway exec (~2s vs the
    # 300s pg_isready timeout the silent background failure would otherwise cost).
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
        echo "[$(date)] staged PGDATA to node-local disk."
    else
        PGDATA="$CANON_PGDATA"
        echo "[$(date)] FRAME_STAGE_PGDATA=0 -> running straight off NFS $PGDATA"
    fi

    # Apptainer only auto-binds $HOME, /tmp, /dev/shm, /proc, /sys and cwd. If PGDATA now
    # lives on node-local /scratch, the container CANNOT SEE it unless we bind it in --
    # postgres then fails with 'could not access directory ... No such file' (observed
    # job 104521). Bind the staging base for /scratch; empty otherwise ($HOME and /tmp
    # are already auto-bound, so the NFS and /tmp-fallback paths need nothing extra).
    STAGE_BIND=()
    case "$PGDATA" in
        /scratch/*) STAGE_BIND=(--bind "$(dirname "$PGDATA")") ;;
    esac

    echo "[$(date)] Starting postgres..."
    apptainer exec --bind /dev/shm --bind /tmp ${STAGE_BIND[@]+"${STAGE_BIND[@]}"} --bind "$PGSOCKET:$PGSOCKET" "$SIF" \
        postgres -D "$PGDATA" -k "$PGSOCKET" -c listen_addresses='' -c logging_collector=off &
    PG_PID=$!
    started_pg=1

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
fi

module load Anaconda3
set +u
source activate embeddings
set -u

if [ "$SYSTEM" = "pgvector" ]; then
    # libpq env so the PgvectorAdapter's psycopg2.connect() finds the in-job socket
    # as the postgres superuser (auth=trust over the unix socket).
    export PGHOST="$PGSOCKET"
    export PGUSER="postgres"
    export PGDATABASE="postgres"
elif [ "$SYSTEM" = "chroma" ]; then
    # Read back the persisted collection chroma_load.sh built. Its persist dir lives on
    # NFS $HOME/chroma; STAGE it to node-local /scratch first, or filtered HNSW search over
    # the 4.1M-keyframe collection reads the index off NFS and takes ~35 min/query (job
    # 104552). Opt out with FRAME_STAGE_CHROMA=0. chromadb isn't in the embeddings env; add
    # it (search needs only chromadb, not pyarrow/h5py).
    CANON_CHROMA="${FRAME_CHROMA_PATH:-$HOME/chroma}"
    if [ "${FRAME_STAGE_CHROMA:-1}" = "1" ] && [ -d "$CANON_CHROMA" ]; then
        FRAME_CHROMA_PATH="$(stage_local "$CANON_CHROMA" chroma)"
        export FRAME_CHROMA_PATH
        echo "[$(date)] staged chroma persist dir to node-local disk: $FRAME_CHROMA_PATH"
    else
        export FRAME_CHROMA_PATH="$CANON_CHROMA"
        echo "[$(date)] FRAME_CHROMA_PATH=$FRAME_CHROMA_PATH (not staged)"
    fi
    echo "[$(date)] Ensuring chromadb>=1.5.0 ..."
    python3 -c "import chromadb,sys; v=tuple(int(x) for x in chromadb.__version__.split('.')[:2]); sys.exit(0 if v>=(1,5) else 1)" 2>/dev/null \
        || pip install --quiet -U 'chromadb>=1.5.0'
fi

cd "$PROJECT_DIR"   # so `import frame` resolves and data/ paths line up
echo "[$(date)] Running run_benchmark.py $* ..."
python3 -u run_benchmark.py "$@"

echo "[$(date)] Stopping postgres (if any) + removing node-local staging..."
cleanup            # EXIT trap re-runs it as an idempotent no-op
echo "[$(date)] Done."

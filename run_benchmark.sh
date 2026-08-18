#!/usr/bin/env bash
#SBATCH --job-name=frame_run
# NODE TARGETING (updated 2026-08-18). The pgvector path now STAGES PGDATA onto
# node-local /scratch to escape the NFS random-read latency that made filtered searches
# take 100-700 s (see the staging block below + Thesis/HPC 'pgvector on hpc guide' ->
# node-local staging). /scratch is only USER-WRITABLE on the big Infiniband nodes --
# confirmed cn3 and cn6 (2.3 TB each, 100 Gbps IB, 192/384 GB RAM). The smaller nodes
# (cn14/15/16...) have a root-owned /scratch and a ~50 GB /tmp, too small for the ~67 GB
# DB. So we pin to cn3/cn6 via cores_any + an exclude list. We do NOT use --exclusive:
# cn3/cn6 are GPU nodes and bare --exclusive grabs their GPUs (rejected by QOS
# MaxGRESPerJob). Acceptable trade -- local-disk staging removes the NFS I/O noise that
# --exclusive used to guard against; residual CPU/mem-bandwidth co-tenancy is minor, and
# IB gives the fastest one-time staging copy. Set FRAME_STAGE_PGDATA=0 to run straight
# off NFS (then any node schedules, but filtered latency is NOT quotable -- the reason
# staging exists). Chroma ignores staging (its persist dir is separate) but still
# benefits from the IB nodes' faster NFS.
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
started_pg=0
if [ "$SYSTEM" = "pgvector" ]; then
    SIF="$HOME/containers/pgvector-pg16.sif"
    CANON_PGDATA="$HOME/pgdata"        # canonical DB (NFS, source of truth; loaded once)
    PGSOCKET="/tmp/pg_${SLURM_JOB_ID:-$$}"
    mkdir -p "$PGSOCKET"

    # ── Stage PGDATA onto NODE-LOCAL disk (the latency fix) ──────────────────────────
    # /home is NFS over 44x 16TB SATA RAID10; HNSW graph traversal is RANDOM 8KB reads,
    # which crawl at ~1-2 MB/s there when the 16GB index doesn't fit in RAM (measured:
    # a filtered search was ~99.6% NFS I/O wait). Node-local /scratch (documented ~1TB,
    # see Thesis/HPC/hpc3-access.md) avoids that entirely. The copy is a ONE-TIME
    # SEQUENTIAL read (fast over IB); every query then does node-local I/O. All benchmark
    # writes (ANALYZE stats etc.) land in the disposable local copy, so the canonical NFS
    # pgdata stays pristine. Opt out with FRAME_STAGE_PGDATA=0 (runs straight off NFS).
    LOCAL_PGDATA=""                    # empty => not staged => nothing to remove later
    if [ "${FRAME_STAGE_PGDATA:-1}" = "1" ]; then
        STAGE_BASE=""
        for base in "/scratch/$USER" "/tmp/$USER"; do
            if mkdir -p "$base" 2>/dev/null && [ -w "$base" ]; then STAGE_BASE="$base"; break; fi
        done
        : "${STAGE_BASE:=/tmp/$USER}"
        # Reap OUR OWN orphaned stagings from jobs that were HARD-killed (a SIGKILL can't
        # run the cleanup trap below). A staging dir whose job id is no longer in the
        # queue is safe to delete -- this is what keeps node-local disk from filling up.
        for d in "$STAGE_BASE"/frame_pgdata_*; do
            [ -d "$d" ] || continue
            jid="${d##*_}"
            if [ -z "$(squeue -h -j "$jid" -o %i 2>/dev/null)" ]; then
                echo "[$(date)] reaping orphan staging $d (job $jid not in queue)"; rm -rf "$d"
            fi
        done
        LOCAL_PGDATA="$STAGE_BASE/frame_pgdata_${SLURM_JOB_ID:-$$}"
        need_kb=$(du -sk "$CANON_PGDATA" | awk '{print $1}')
        free_kb=$(df -Pk "$STAGE_BASE" | awk 'NR==2{print $4}')
        echo "[$(date)] staging PGDATA $CANON_PGDATA -> $LOCAL_PGDATA "\
"(need $((need_kb/1048576))G, free $((free_kb/1048576))G on $STAGE_BASE)"
        if [ "$free_kb" -lt "$((need_kb + need_kb/10))" ]; then
            echo "[$(date)] FATAL: not enough node-local space to stage PGDATA." >&2; exit 1
        fi
        rm -rf "$LOCAL_PGDATA"
        cp -a "$CANON_PGDATA" "$LOCAL_PGDATA"
        rm -f "$LOCAL_PGDATA/postmaster.pid"   # never carry a stale lock into the copy
        chmod 700 "$LOCAL_PGDATA"
        PGDATA="$LOCAL_PGDATA"
        echo "[$(date)] staged to node-local disk."
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

    # Stop postgres AND remove the node-local staging copy however this job ends --
    # success, an error under `set -e`, or scancel/timeout (the INT/TERM trap converts
    # SIGTERM into a normal exit so the EXIT trap runs). A hard SIGKILL cannot run this;
    # the orphan reaper above cleans such leftovers on the next job on that node.
    stop_pg() {
        apptainer exec --bind /dev/shm --bind /tmp ${STAGE_BIND[@]+"${STAGE_BIND[@]}"} "$SIF" \
            pg_ctl stop -D "$PGDATA" -m fast 2>/dev/null || true
        wait "$PG_PID" 2>/dev/null || true
    }
    cleanup() {
        stop_pg
        if [ -n "${LOCAL_PGDATA:-}" ] && [ -d "$LOCAL_PGDATA" ]; then
            echo "[$(date)] removing node-local staging $LOCAL_PGDATA"
            rm -rf "$LOCAL_PGDATA"
        fi
    }
    trap cleanup EXIT
    trap 'exit 143' INT TERM   # let the EXIT trap do the shutdown + cleanup, then die

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
    # Read back the persisted collection chroma_load.sh built. chromadb isn't in the
    # embeddings env; add it (search needs only chromadb, not pyarrow/h5py).
    export FRAME_CHROMA_PATH="${FRAME_CHROMA_PATH:-$HOME/chroma}"
    echo "[$(date)] Ensuring chromadb>=1.5.0 ..."
    python3 -c "import chromadb,sys; v=tuple(int(x) for x in chromadb.__version__.split('.')[:2]); sys.exit(0 if v>=(1,5) else 1)" 2>/dev/null \
        || pip install --quiet -U 'chromadb>=1.5.0'
fi

cd "$PROJECT_DIR"   # so `import frame` resolves and data/ paths line up
echo "[$(date)] Running run_benchmark.py $* ..."
python3 -u run_benchmark.py "$@"

if [ "$started_pg" -eq 1 ]; then
    echo "[$(date)] Stopping postgres + removing node-local staging..."
    cleanup            # stop postgres and delete the staged copy (EXIT trap re-runs it as a no-op)
fi
echo "[$(date)] Done."

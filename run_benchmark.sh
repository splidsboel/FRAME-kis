#!/usr/bin/env bash
#SBATCH --job-name=frame_run
#SBATCH --partition=cores_any
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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
    PGDATA="$HOME/pgdata"
    PGSOCKET="/tmp/pg_${SLURM_JOB_ID:-$$}"
    mkdir -p "$PGSOCKET"

    echo "[$(date)] Starting postgres..."
    apptainer exec --bind /dev/shm --bind /tmp --bind "$PGSOCKET:$PGSOCKET" "$SIF" \
        postgres -D "$PGDATA" -k "$PGSOCKET" -c listen_addresses='' -c logging_collector=off &
    PG_PID=$!
    started_pg=1

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
    echo "[$(date)] Stopping postgres..."
    stop_pg
fi
echo "[$(date)] Done."

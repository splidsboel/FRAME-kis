#!/usr/bin/env bash
#SBATCH --job-name=build_gt
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/build_gt_%j.out

# Enrich data/benchmark.jsonl with DB-computed ground truth. Needs BOTH a GPU
# (SigLIP text encoder) and the V3C postgres, so it starts the in-job postgres,
# then runs the oracle. Submit from the repo root:  sbatch build_gt.sh [args]
#
# Args after the script name pass straight to oracle/build_gt.py, e.g.:
#     sbatch build_gt.sh                                          # diagnostics pass
#     sbatch build_gt.sh --scene-threshold 0.10 --object-threshold 0.30
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

apptainer exec --bind /dev/shm --bind /tmp "$SIF" \
    psql -h "$PGSOCKET" -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;" postgres

module load Anaconda3
set +u
source activate embeddings
set -u

export PGHOST="$PGSOCKET"

# Always recompile benchmark.jsonl from the authored queries/*.json first, so a
# stale/hand-edited benchmark.jsonl can never feed the GT run (queries/*.json is
# the single source of truth).
echo "[$(date)] Rebuilding data/benchmark.jsonl from queryset/queries/ ..."
python3 -u "$PROJECT_DIR/queryset/build.py"

echo "[$(date)] Running oracle/build_gt.py $* ..."
python3 -u "$PROJECT_DIR/oracle/build_gt.py" "$@"

echo "[$(date)] Stopping postgres..."
apptainer exec --bind /dev/shm --bind /tmp "$SIF" pg_ctl stop -D "$PGDATA" -m fast
wait "$PG_PID" 2>/dev/null || true
echo "[$(date)] Done."

#!/usr/bin/env bash
#SBATCH --job-name=chroma_load
#SBATCH --partition=cores_any
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/chroma_load_%j.out

# Ingest a Tier-2 canonical shard (or the union) into the Chroma system under test
# via ChromaAdapter.load_data() — the denormalised single-collection layout.
#
# NOTE (why this is simpler than load_dataset.sh): Chroma is EMBEDDED. There is no
# server and no Apptainer container — so this job does NOT hit the cores_any user-
# namespace exhaustion that is currently blocking the pgvector jobs (see the diary,
# 2026-08-15). It only needs the conda env + a persist directory on $HOME.
#
# Submit from the FRAME-kis repo root, from `ssh hpc3`:
#     sbatch chroma_load.sh --dataset data/canonical/v3c1
#     sbatch chroma_load.sh --dataset data/canonical/v3c1 data/canonical/v3c2 data/canonical/v3c3
#     sbatch chroma_load.sh --check          # tiny synthetic roundtrip, no real data
# Extra args pass straight through to scripts/load_dataset.py (--system is forced to
# chroma below).
#
# Wall time is dominated by Chroma's INCREMENTAL HNSW build: unlike pgvector there
# is no build-at-end, every add() inserts into the live hnswlib graph, which is
# single-threaded per collection. For the ~4.1M-vector union this is slow — hence
# the 24h ceiling. Memory holds the whole graph in RAM plus one shard's grouping
# maps, so --mem is generous. The job is safe to re-run: load_datasets() skips when
# the record count already matches (use --force to rebuild).

set -euo pipefail

# Admin-mandated in every job script (keeps temp off the shared system /tmp).
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# Where the persisted Chroma collection lives (read back by run_benchmark.sh). On
# $HOME (NFS) so it survives the job and is visible to the run job on any node —
# the embedded client is single-process, so NFS sqlite locking is not a concern.
export FRAME_CHROMA_PATH="${FRAME_CHROMA_PATH:-$HOME/chroma}"
mkdir -p "$FRAME_CHROMA_PATH"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

module load Anaconda3
set +u
source activate embeddings
set -u

# chromadb (>=1.5.0, for array metadata + $contains) and the ingest deps (pyarrow,
# h5py) are not in the embeddings env; add them (wheels, no build). No-op if present.
echo "[$(date)] Ensuring chromadb>=1.5.0 + pyarrow + h5py ..."
python3 -c "import chromadb,sys; v=tuple(int(x) for x in chromadb.__version__.split('.')[:2]); sys.exit(0 if v>=(1,5) else 1)" 2>/dev/null \
    || pip install --quiet -U 'chromadb>=1.5.0'
python3 -c "import pyarrow, h5py" 2>/dev/null || pip install --quiet pyarrow h5py

cd "$PROJECT_DIR"   # so `import frame` resolves and data/ paths line up
if [ "${1:-}" = "--check" ]; then
    # Synthetic-shard roundtrip for the denormalised load; touches nothing real.
    shift
    echo "[$(date)] Running scripts/check_chroma_roundtrip.py $* ..."
    python3 -u scripts/check_chroma_roundtrip.py "$@"
else
    echo "[$(date)] Running scripts/load_dataset.py --system chroma $* ..."
    python3 -u scripts/load_dataset.py --system chroma "$@"
fi

echo "[$(date)] Done."

#!/usr/bin/env bash
#SBATCH --job-name=frame_meta
#SBATCH --partition=cores_any
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/frame_meta_%j.out

# Build the non-model canonical tables (videos/shots/keyframes.parquet) for an
# extracted V3C shard. CPU only, no postgres. See [[Data pipeline and adapter
# load refactor]]. Submit from the FRAME-kis repo root:
#     sbatch scripts/prep_metadata.sh                              # V3C2 -> data/canonical/v3c2
#     sbatch scripts/prep_metadata.sh ~/datasets/V3C/V3C3 v3c3

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SHARD_ROOT="${1:-$HOME/datasets/V3C/V3C2}"
DATASET="${2:-v3c2}"

module load Anaconda3
set +u; source activate embeddings; set -u
python3 -c "import pyarrow" 2>/dev/null || pip install --quiet pyarrow

cd "$PROJECT_DIR"
python3 -u scripts/prep_metadata.py --shard-root "$SHARD_ROOT" --dataset "$DATASET"
echo "[$(date)] Done."

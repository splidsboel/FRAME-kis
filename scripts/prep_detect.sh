#!/usr/bin/env bash
#SBATCH --job-name=frame_detect
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/frame_detect_%A_%a.out

# OWLv2 object detection for an extracted V3C shard -> per-video parquet staging.
# GPU array; no postgres. Uses ~/object_vocab.txt (same vocab as V3C1). Resumable.
# See [[Data pipeline and adapter load refactor]]. From the FRAME-kis repo root:
#     sbatch scripts/prep_detect.sh                            # V3C2
#     sbatch scripts/prep_detect.sh ~/datasets/V3C/V3C3 v3c3

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SHARD_ROOT="${1:-$HOME/datasets/V3C/V3C2}"
DATASET="${2:-v3c2}"
# Fixed at 8 to match `#SBATCH --array=0-7` — deliberately NOT SLURM_ARRAY_TASK_COUNT.
# Resubmitting a subset after a failure (e.g. --array=1,6,7) sets that to 3, which
# would silently re-map every video to a different shard and corrupt the staging.
NUM_SHARDS="${NUM_SHARDS:-8}"

[ -f "$HOME/object_vocab.txt" ] || { echo "missing ~/object_vocab.txt (object vocab)"; exit 1; }

module load Anaconda3
set +u; source activate embeddings; set -u
python3 -c "import pyarrow" 2>/dev/null || pip install --quiet pyarrow

cd "$PROJECT_DIR"
python3 -u scripts/prep_detect.py --shard-root "$SHARD_ROOT" --dataset "$DATASET" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NUM_SHARDS"
echo "[$(date)] Done (shard ${SLURM_ARRAY_TASK_ID:-0})."

#!/usr/bin/env bash
#SBATCH --job-name=frame_scenes
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/frame_scenes_%A_%a.out

# Places365 (ResNet50) scene classification for an extracted V3C shard ->
# per-video parquet staging. GPU array; no postgres. Uses the same external
# weights/labels under ~/models/places365/ as V3C1. Resumable. From the repo root:
#     sbatch scripts/prep_scenes.sh                            # V3C2
#     sbatch scripts/prep_scenes.sh ~/datasets/V3C/V3C3 v3c3

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SHARD_ROOT="${1:-$HOME/datasets/V3C/V3C2}"
DATASET="${2:-v3c2}"

[ -f "$HOME/models/places365/resnet50_places365.pth.tar" ] || { echo "missing Places365 weights"; exit 1; }

module load Anaconda3
set +u; source activate embeddings; set -u
python3 -c "import pyarrow" 2>/dev/null || pip install --quiet pyarrow

cd "$PROJECT_DIR"
python3 -u scripts/prep_scenes.py --shard-root "$SHARD_ROOT" --dataset "$DATASET" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}"
echo "[$(date)] Done (shard ${SLURM_ARRAY_TASK_ID:-0})."

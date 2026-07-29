#!/usr/bin/env bash
#SBATCH --job-name=frame_ocr
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/frame_ocr_%A_%a.out

# EasyOCR in-frame text for an extracted V3C shard -> per-video parquet staging.
# GPU array; no postgres. Uses pre-fetched models under ~/models/easyocr
# (download_enabled=False). Resumable. From the FRAME-kis repo root:
#     sbatch scripts/prep_ocr.sh                               # V3C2
#     sbatch scripts/prep_ocr.sh ~/datasets/V3C/V3C3 v3c3

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SHARD_ROOT="${1:-$HOME/datasets/V3C/V3C2}"
DATASET="${2:-v3c2}"
export OCR_GPU=1

module load Anaconda3
set +u; source activate embeddings; set -u
python3 -c "import pyarrow" 2>/dev/null || pip install --quiet pyarrow
python3 -c "import easyocr" 2>/dev/null || pip install --quiet easyocr

cd "$PROJECT_DIR"
python3 -u scripts/prep_ocr.py --shard-root "$SHARD_ROOT" --dataset "$DATASET" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}"
echo "[$(date)] Done (shard ${SLURM_ARRAY_TASK_ID:-0})."

#!/usr/bin/env bash
#SBATCH --job-name=frame_detect
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/frame_detect_%A_%a.out

# --array was 0-7: at 8-way this pass took 12h41-17h31 per task on V3C2 and was
# ~60% of the whole pipeline's wall clock, with every other pass idle waiting on
# it. 16-way halves it to ~7-9h. Widening is only safe on a dataset with NO
# existing staging (see the NUM_SHARDS note below) — V3C3 is fresh.

# OWLv2 object detection for an extracted V3C shard -> per-video parquet staging.
# GPU array; no postgres. Uses ~/object_vocab.txt (same vocab as V3C1). Resumable.
# See [[Data pipeline and adapter load refactor]]. From the FRAME-kis repo root:
#     sbatch scripts/prep_detect.sh                            # V3C2
#     sbatch scripts/prep_detect.sh ~/datasets/V3C/V3C3 v3c3

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SHARD_ROOT="${1:-$HOME/datasets/V3C/V3C2}"
DATASET="${2:-v3c2}"
# Fixed at 16 to match `#SBATCH --array=0-15` — deliberately NOT SLURM_ARRAY_TASK_COUNT.
# video_dirs() assigns videos round-robin (`i % num_shards == shard`), so the
# INVARIANT is that the submitted array covers every index 0..NUM_SHARDS-1; then
# each video is owned by exactly one task. Resubmitting a subset after a failure
# (e.g. --array=1,6,7) sets SLURM_ARRAY_TASK_COUNT to 3, which would silently
# re-map every video to a different shard and leave most of them unprocessed —
# which is why this is a literal. Changing the width is safe only when you change
# BOTH lines together and submit the full array.
NUM_SHARDS="${NUM_SHARDS:-16}"

[ -f "$HOME/object_vocab.txt" ] || { echo "missing ~/object_vocab.txt (object vocab)"; exit 1; }

module load Anaconda3
set +u; source activate embeddings; set -u
python3 -c "import pyarrow" 2>/dev/null || pip install --quiet pyarrow

cd "$PROJECT_DIR"
python3 -u scripts/prep_detect.py --shard-root "$SHARD_ROOT" --dataset "$DATASET" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NUM_SHARDS"
echo "[$(date)] Done (shard ${SLURM_ARRAY_TASK_ID:-0})."

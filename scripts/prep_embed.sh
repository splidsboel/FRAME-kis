#!/usr/bin/env bash
#SBATCH --job-name=frame_embed
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/frame_embed_%A_%a.out

# --time was 1-00:00:00: tasks 6+7 of job 100871 hung on cn12 and burned the full
# 24h without writing a single log line. A full 1220-video task takes ~3.5h, so 8h
# is ample headroom and a hang now fails fast. The pass is resumable, so a
# wall-clock kill only ever costs the video in flight.

# SigLIP image embeddings for an extracted V3C shard -> per-video npz staging.
# GPU array (one shard of videos per task); no postgres. Resumable — re-submit
# after a preemption and finished videos are skipped. See [[Data pipeline and
# adapter load refactor]]. Submit from the FRAME-kis repo root:
#     sbatch scripts/prep_embed.sh                             # V3C2
#     sbatch scripts/prep_embed.sh ~/datasets/V3C/V3C3 v3c3    # V3C3
# --num-shards is pinned to 8 below; keep it in sync with #SBATCH --array=0-7.

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SHARD_ROOT="${1:-$HOME/datasets/V3C/V3C2}"
DATASET="${2:-v3c2}"
# Fixed at 8 to match `#SBATCH --array=0-7` — deliberately NOT SLURM_ARRAY_TASK_COUNT.
# Resubmitting a subset after a failure (e.g. --array=1,6,7) sets that to 3, which
# would silently re-map every video to a different shard and corrupt the staging.
NUM_SHARDS="${NUM_SHARDS:-8}"

module load Anaconda3
set +u; source activate embeddings; set -u
python3 -c "import pyarrow" 2>/dev/null || pip install --quiet pyarrow

cd "$PROJECT_DIR"
python3 -u scripts/prep_embed.py --shard-root "$SHARD_ROOT" --dataset "$DATASET" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NUM_SHARDS"
echo "[$(date)] Done (shard ${SLURM_ARRAY_TASK_ID:-0})."

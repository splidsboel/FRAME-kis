#!/usr/bin/env bash
#SBATCH --job-name=frame_consolidate
#SBATCH --partition=cores_any
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/frame_consolidate_%j.out

# Merge the per-video staging (embed/detect/scenes/ocr) into the canonical
# single-file dataset + MANIFEST.json. Run AFTER prep_metadata + the four model
# passes are complete. CPU only, no postgres. From the FRAME-kis repo root:
#     sbatch scripts/prep_consolidate.sh                       # v3c2
#     sbatch scripts/prep_consolidate.sh v3c3

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATASET="${1:-v3c2}"

module load Anaconda3
set +u; source activate embeddings; set -u
python3 -c "import pyarrow, h5py" 2>/dev/null || pip install --quiet pyarrow h5py

cd "$PROJECT_DIR"
python3 -u scripts/prep_consolidate.py --dataset "$DATASET"
echo "[$(date)] Done."

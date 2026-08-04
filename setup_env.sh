#!/usr/bin/env bash
#SBATCH --job-name=frame_setup_env
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/frame_setup_env_%j.out

# Build the FRAME conda environment on the ITU HPC — FROM A JOB, never on a head
# node. Follows the HPC admin's recipe (APRIL_EXERCISES_NLP slides, Inna Ermilova):
# module Miniconda3 + explicit CONDA_PKGS_DIRS/TMPDIR + ToS accept + `conda create
# -p <prefix>` + pip into the activated prefix.
#
#     sbatch setup_env.sh            # build/refresh ~/.conda/envs/frame
#
# NOTE: this is normally NOT needed. The pre-existing `embeddings` env already has
# everything FRAME imports (torch 2.5.1+cu121, transformers, psycopg2, numpy,
# matplotlib) and is what build_gt.sh / run_benchmark.sh activate. This script
# exists so the environment is *reproducible* — run it if `embeddings` is lost, or
# to move to Python 3.12 (FRAME's pyproject asks for >=3.11; `embeddings` is 3.10,
# which works because nothing in the code uses 3.11-only features).
#
# Takes ~30-60 min, mostly the torch download.

set -euo pipefail
set -x

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"      # run `sbatch` from the FRAME-kis root
ENV_PREFIX="$HOME/.conda/envs/frame"
MINICONDA="/opt/itu/easybuild/software/Miniconda3/25.5.1-1"

module load Miniconda3/25.5.1-1
module load GCCcore/13.3.0

# Admin-mandated: keep temp + package caches inside $HOME, off the shared /tmp.
export TMPDIR="$HOME/tmp"
export CONDA_PKGS_DIRS="$HOME/conda_pkgs_cache"
export PIP_CACHE_DIR="$HOME/.cache/pip"
mkdir -p "$TMPDIR" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

# Anaconda ToS must be accepted non-interactively before `conda create` will run.
"$MINICONDA/bin/conda" tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/main
"$MINICONDA/bin/conda" tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/r

if [ ! -d "$ENV_PREFIX" ]; then
    "$MINICONDA/bin/conda" create -y -p "$ENV_PREFIX" python=3.12
fi

# `conda activate` needs the shell function, which batch shells don't have.
source "$MINICONDA/etc/profile.d/conda.sh"
set +u                       # conda's activate scripts touch unbound vars
conda activate "$ENV_PREFIX"
set -u

# Editable install of FRAME itself + the heavy extras, so `import frame` works
# from anywhere and torch/transformers/psycopg2/matplotlib come along.
python3 -m pip install --upgrade pip
python3 -m pip install -e "$PROJECT_DIR[oracle,viz]"

# Smoke test. GPU visibility is checked by a separate GPU job — scavenge may put
# this one on a CPU-only or old-driver node, so a False here means nothing.
python3 -u - <<'EOF'
import sys
print("python:", sys.version)
import numpy, torch, transformers, psycopg2, matplotlib, frame
print("numpy", numpy.__version__, "| torch", torch.__version__,
      "| transformers", transformers.__version__,
      "| psycopg2", psycopg2.__version__.split()[0],
      "| matplotlib", matplotlib.__version__)
print("frame ->", frame.__file__)
print("cuda available (ignore on a CPU node):", torch.cuda.is_available())
EOF

set +x
echo
echo "Done. To use this env instead of \`embeddings\`, point build_gt.sh /"
echo "run_benchmark.sh at it:"
echo "    source $MINICONDA/etc/profile.d/conda.sh"
echo "    conda activate $ENV_PREFIX"

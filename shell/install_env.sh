#!/bin/bash
# Create the env described in environment.yml under /work/btang1/envs.
# conda only provides python; pytorch comes from the official CUDA pip wheels
# (the pytorch/nvidia conda channels with pytorch-cuda=11.7 no longer solve).
#
# Usage: ./shell/install_env.sh [env_name]      (default: DiT)
#        FORCE=1 ./shell/install_env.sh         (remove and recreate an existing env)
#        CUDA=cu118 ./shell/install_env.sh      (pick wheel CUDA version, default cu121)

set -e

ENV_NAME=${1:-DiT}
PREFIX=/work/btang1/envs/$ENV_NAME
CUDA=${CUDA:-cu121}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}

source ~/miniforge3/etc/profile.d/conda.sh

# make `conda` available in non-interactive shells
if ! command -v conda >/dev/null 2>&1; then
    module load conda 2>/dev/null || module load miniconda 2>/dev/null || module load anaconda 2>/dev/null || true
fi
command -v conda >/dev/null 2>&1 || { echo "conda not found; load it first" >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ -d "$PREFIX" ]; then
    if [ "${FORCE:-0}" = "1" ]; then
        echo "removing existing env at $PREFIX"
        conda env remove -p "$PREFIX" -y || rm -rf "$PREFIX"
    else
        echo "$PREFIX already exists; rerun with FORCE=1 to recreate, or pass another name" >&2
        exit 1
    fi
fi

mkdir -p /work/btang1/envs
echo "creating env at $PREFIX (python $PYTHON_VERSION, torch $CUDA)"
conda create -p "$PREFIX" -y -c conda-forge --override-channels python="$PYTHON_VERSION" pip

conda activate "$PREFIX"
pip install --upgrade pip
pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA"
pip install timm diffusers accelerate

python -c "import torch, torchvision, timm, diffusers, accelerate; \
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| cuda', torch.version.cuda)"

echo "done. activate with: conda activate $PREFIX"

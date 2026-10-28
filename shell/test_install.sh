#!/bin/bash
#SBATCH -J test_install
#SBATCH -p gpu-hp
#SBATCH --qos=ncsu_h200_hp
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Initialization
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /work/btang1/envs/DiT

python -c "import torch, torchvision, timm, diffusers, accelerate; \
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| cuda', torch.version.cuda)"

echo "done. activate with: conda activate $PREFIX"

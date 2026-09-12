#!/bin/bash
#SBATCH -J sample-dit-xl2_ckpt350k
#SBATCH -p gpu-hp
#SBATCH --qos=ncsu_h200_hp
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -e 

# monitor gpu usage
mkdir -p gpu_usage
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 30 > "gpu_usage/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" &
NVSMI_PID=$!
trap 'kill $NVSMI_PID 2>/dev/null' EXIT

# Initialization
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /work/btang1/envs/DiT

torchrun --nnodes=1 --nproc_per_node=1 sample_ddp.py --vae mse \
    --cfg_scale 1.0 \
    --ckpt results/DiT-XL-2/checkpoints/0350000.pt \
    --global-seed 42
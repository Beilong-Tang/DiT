#!/bin/bash
#SBATCH -J train-dit-b2
#SBATCH -p gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -e 

# monitor gpu usage
mkdir -p gpu_usage
gpumon 30 > >(tee "gpu_usage/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log") &
NVSMI_PID=$!
trap 'kill $NVSMI_PID 2>/dev/null' EXIT

# 00 transfer the imagenet data to the scratch folder for faster loading
SRC="/work/btang1/data/Imagenet_process/imagenet_train"
./shell/00_copy_data.sh $SRC


# Initialization
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /work/btang1/envs/DiT

# Verify CUDA availability
python - <<'PY'
import torch

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
PY

LEARN_SIGMA=false
RESULTS_DIR=results/no_learned_sigma

# 01 training 
MEM_ALLOC=0
torchrun --nnodes=1 --nproc_per_node=1 train.py --model DiT-B/2 \
    --data-path /scratch/imagenet \
    --global-batch-size 256 \
    --vae "mse" \
    --epochs 80 \
    --resume \
    --mem-alloc $MEM_ALLOC \
    --learn-sigma $LEARN_SIGMA \
    --results-dir $RESULTS_DIR


# 02 sampling
torchrun --nnodes=1 --nproc_per_node=1 sample_ddp.py --model DiT-B/2 --vae mse \
    --cfg-scale 1.0 \
    --ckpt $RESULTS_DIR/DiT-B-2/checkpoints/0400000.pt \
    --global-seed 42 \
    --mem-alloc 60 \
    --learn-sigma $LEARN_SIGMA \
    --sample-dir samples_no_sigma

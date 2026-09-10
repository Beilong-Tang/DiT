#!/bin/bash
#SBATCH -J train-dit-xl2
#SBATCH -p gpu-hp
#SBATCH --qos=ncsu_h200_hp
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH -t 56:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -e 

# monitor gpu usage
mkdir -p gpu_usage
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 30 > "gpu_usage/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" &
NVSMI_PID=$!
trap 'kill $NVSMI_PID 2>/dev/null' EXIT


# 00 transfer the imagenet data to the scratch folder for faster loading
SRC="/work/btang1/data/Imagenet_process/imagenet_train"
./shell/00_copy_data.sh $SRC

# 01 training 
MEM_ALLOC=30
torchrun --nnodes=1 --nproc_per_node=4 train.py --model DiT-XL/2 \
    --data-path /scratch/imagenet \
    --global-batch-size 256 \
    --vae "mse" \
    --epochs 80 \
    --resume \
    --mem-alloc $MEM_ALLOC
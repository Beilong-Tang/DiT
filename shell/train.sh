#!/bin/bash

set -e 

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 30 > gpu_usage.log &


# 00 transfer the imagenet data to the scratch folder for faster loading
# SRC="/work/btang1/data/Imagenet_process/imagenet_train"
# ./shell/00_copy_data.sh $SRC


torchrun --nnodes=1 --nproc_per_node=1 train.py --model DiT-XL/2 --data-path /scratch/imagenet --global-batch-size 64 --log-every 10

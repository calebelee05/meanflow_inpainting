#!/bin/bash
#SBATCH --job-name=MF00
#SBATCH --account=cw220
#SBATCH --partition=commons
#SBATCH --time=2-00:00:00    
#SBATCH --ntasks=1                # 任务数
#SBATCH --cpus-per-task=8         # 每个任务分配 CPU 核数（可根据需要调整）
#SBATCH --gres=gpu:h200:1              # 使用 4 张 GPU
#SBATCH --mem=96G                  # <-- 加上内存限制（避免 OOM 被杀，可以写 64G / 96G / 0 全部）
#SBATCH --mail-user=yw251@rice.edu
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x-%j.out 

export PYTORCH_ENABLE_FUNC_IMPL=1 && \
export PYTORCH_DDP_NO_REBUILD_BUCKETS=1 && \
export TORCH_NCCL_IB_TIMEOUT=23 && \
export NCCL_TIMEOUT=3600 && \
export SETUPTOOLS_USE_DISTUTILS=local && \

torchrun --standalone --nproc_per_node=1 train_mf.py \
    --detach_tgt=1 \
    --outdir=logs/mf/MF00 \
    --data=../datasets/cifar10-32x32.zip \
    --cond=0 --arch=ddpmpp --lr 6e-4 --batch 200 \
    # --resume /home/william/easy_meanflow/logs/mf/MF00/00002-cifar10-32x32-uncond-ddpmpp-mf-gpus4-batch256-fp32/training-state-200000.pt \

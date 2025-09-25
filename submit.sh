#!/bin/bash

#SBATCH --job-name=exp_023_vp_extend
#SBATCH --output=exp/exp_023_vp_extend/exp_023_vp_extend_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source /lus/lfs1aip2/home/u5au/chexuan.u5au/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 scripts/train.py -E exp/exp_023_vp_extend -L exp/exp_022_vp/saved_models/last.ckpt
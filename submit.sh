#!/bin/bash

#SBATCH --job-name=exp_018_vc_extend
#SBATCH --output=exp/exp_018_vc_extend/exp_018_vc_%j.out
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source /lus/lfs1aip2/home/u5au/chexuan.u5au/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 scripts/train.py -E exp/exp_018_vc_extend -L exp/exp_017_vc/saved_models/last.ckpt
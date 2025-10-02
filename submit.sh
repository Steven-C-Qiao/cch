#!/bin/bash

#SBATCH --job-name=exp_028_vp_template
#SBATCH --output=exp/exp_028_vp_template/exp_028_vp_template-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source /lus/lfs1aip2/home/u5au/chexuan.u5au/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 scripts/train.py -E exp/exp_028_vp_template -L exp/exp_027_vc_template/saved_models/last.ckpt
#!/bin/bash

#SBATCH --job-name=exp_026_smplx_vp
#SBATCH --output=exp/exp_026_smplx_vp/exp_026_smplx_vp-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=1-00:00:00 

source /lus/lfs1aip2/home/u5au/chexuan.u5au/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 scripts/train.py -E exp/exp_026_smplx_vp  -L exp/exp_025_smplx/saved_models/last.ckpt
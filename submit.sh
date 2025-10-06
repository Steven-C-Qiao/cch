#!/bin/bash

#SBATCH --job-name=exp_034_vp_extend_033
#SBATCH --output=exp/exp_034_vp_extend_033/exp_034_vp_extend_033-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 scripts/train.py -E exp/exp_034_vp_extend_033  -L exp/exp_033_vp_extend_031/saved_models/last.ckpt
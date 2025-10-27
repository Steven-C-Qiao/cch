#!/bin/bash

#SBATCH --job-name=exp_054_vp_norm
#SBATCH --output=exp/exp_054_vp_norm/exp_054-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 scripts/train.py \
    -E exp/exp_054_vp_norm \
    -R exp/exp_054_vp_norm/saved_models/last.ckpt
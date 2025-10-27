#!/bin/bash

#SBATCH --job-name=exp_055_vp_asaploss
#SBATCH --output=exp/exp_055_vp_asaploss/exp_055-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 scripts/train.py \
    -E exp/exp_055_vp_asaploss \
    -R exp/exp_054_vp_norm/saved_models/last.ckpt
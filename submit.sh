#!/bin/bash

#SBATCH --job-name=exp_038
#SBATCH --output=exp/exp_038_thuman/exp_038_thuman-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 scripts/train.py \
    -E exp/exp_038_thuman \
    -L exp/exp_038_thuman/saved_models/val_vp_cfd_epoch=001.ckpt
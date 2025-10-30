#!/bin/bash

#SBATCH --job-name=exp_059_alt_rend_thuman
#SBATCH --output=exp/exp_059_alt_rend_thuman/exp-%j.out
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 scripts/train.py \
    -E exp/exp_059_alt_rend_thuman \
    -L exp/exp_055_vp_f_vc/saved_models/val_vc_cfd_epoch=001.ckpt
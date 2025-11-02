#!/bin/bash

#SBATCH --job-name=exp_079_l055_scale
#SBATCH --output=exp/exp_079_l055_scale/exp-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=23:00:00

source ~/miniforge3/bin/activate dev

srun python3 scripts/train.py \
    -E exp/exp_079_l055_scale \
    -L exp/exp_055_vp_f_vc/saved_models/val_vp_cfd_epoch=000.ckpt
#!/bin/bash

#SBATCH --job-name=exp_101_vp_l090
#SBATCH --output=exp/exp_101_vp_l090/exp-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=20:00:00

source ~/miniforge3/bin/activate dev

srun python3 scripts/train.py \
    -E exp/exp_101_vp_l090 \
    -R exp/exp_101_vp_l090/saved_models/last.ckpt
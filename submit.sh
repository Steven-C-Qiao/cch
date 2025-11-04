#!/bin/bash

#SBATCH --job-name=exp_085_normal_m
#SBATCH --output=exp/exp_085_normal_m/exp-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=23:00:00

source ~/miniforge3/bin/activate dev

srun python3 scripts/train.py \
    -E exp/exp_085_normal_m \
    -R exp/exp_085_normal_m/saved_models/last.ckpt
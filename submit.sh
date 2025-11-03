#!/bin/bash

#SBATCH --job-name=exp_083_normals_tune
#SBATCH --output=exp/exp_083_normals_tune/exp-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=23:00:00

source ~/miniforge3/bin/activate dev

srun python3 scripts/train.py \
    -E exp/exp_083_normals_tune \
    -L exp/exp_081_l055_det/saved_models/last.ckpt
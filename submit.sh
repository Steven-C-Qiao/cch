#!/bin/bash

#SBATCH --job-name=exp_100_5_vp_extend
#SBATCH --output=exp/exp_100_5_vp_extend/exp-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00

source ~/miniforge3/bin/activate dev

srun python3 scripts/train.py \
    -E exp/exp_100_5_vp_extend \
    -R exp/exp_100_5_vp/saved_models/last.ckpt
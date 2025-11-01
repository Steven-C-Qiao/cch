#!/bin/bash

#SBATCH --job-name=exp_071_r047
#SBATCH --output=exp/exp_071_r047/exp-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=23:00:00

source ~/miniforge3/bin/activate dev

srun python3 scripts/train.py \
    -E exp/exp_071_r047 \
    -L exp/exp_047_vc_sapiens/saved_models/last.ckpt
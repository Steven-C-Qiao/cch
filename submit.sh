#!/bin/bash

#SBATCH --job-name=exp_081_l055_det
#SBATCH --output=exp/exp_081_l055_det/exp-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=23:00:00

source ~/miniforge3/bin/activate dev

srun python3 scripts/train.py \
    -E exp/exp_081_l055_det \
    -R exp/exp_081_l055_det/saved_models/last.ckpt
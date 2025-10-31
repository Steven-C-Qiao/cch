#!/bin/bash

#SBATCH --job-name=exp_067_tune_066
#SBATCH --output=exp/exp_067_tune_066/exp-%j.out
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

srun python3 scripts/train.py \
    -E exp/exp_067_tune_066 \
    -R exp/exp_066_asap_moresamples_largerender/saved_models/last.ckpt
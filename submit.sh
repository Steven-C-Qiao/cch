#!/bin/bash

#SBATCH --job-name=exp_100_1_vc
#SBATCH --output=exp/exp_100_1_vc/exp-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=20:00:00

source ~/miniforge3/bin/activate dev

srun python3 scripts/train.py \
    -O TRAIN.NUM_EPOCHS 25 \
    -O MODEL.POSE_BLENDSHAPES False \
    -O MODEL.FREEZE_CANONICAL_MODULES False \
    -E exp/exp_100_1_vc \
    -R exp/exp_100_1_vc/saved_models/last.ckpt
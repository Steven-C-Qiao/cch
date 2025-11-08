#!/bin/bash

#SBATCH --job-name=exp_100_2_vp_fvc
#SBATCH --output=exp/exp_100_2_vp_fvc/exp-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=20:00:00

source ~/miniforge3/bin/activate dev

srun python3 scripts/train.py \
    -O TRAIN.NUM_EPOCHS 1 \
    -O MODEL.POSE_BLENDSHAPES True \
    -O MODEL.FREEZE_CANONICAL_MODULES True \
    -E exp/exp_100_2_vp_fvc \
    -L exp/exp_100_1_vc/saved_models/last.ckpt
#!/bin/bash

#SBATCH --job-name=exp_100_3_vp
#SBATCH --output=exp/exp_100_3_vp/exp-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=20:00:00

source ~/miniforge3/bin/activate dev

srun python3 scripts/train.py \
    -O TRAIN.NUM_EPOCHS 20 \
    -O MODEL.POSE_BLENDSHAPES True \
    -O MODEL.FREEZE_CANONICAL_MODULES False \
    -O LOSS.SCALE_GT2PRED 5.0 \
    -E exp/exp_100_3_vp \
    -L exp/exp_100_2_vp_fvc/saved_models/last.ckpt
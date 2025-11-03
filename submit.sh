#!/bin/bash

#SBATCH --job-name=exp_084_vc_vp_normals
#SBATCH --output=exp/exp_084_vc_vp_normals/exp-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=23:00:00

source ~/miniforge3/bin/activate dev

srun python3 scripts/train.py \
    -E exp/exp_084_vc_vp_normals \
    -L exp/exp_083_normals_tune/saved_models/last.ckpt
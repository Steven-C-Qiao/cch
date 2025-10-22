#!/bin/bash

#SBATCH --job-name=exp_041
#SBATCH --output=exp/exp_042_th2p/exp_042-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 scripts/train.py \
    -E exp/exp_042_th2p \
    -L exp/exp_035_vp_extend_031_scale_gt2pred_loss_5k/saved_models/last-v1.ckpt

#!/bin/bash

#SBATCH --job-name=exp_056_vc
#SBATCH --output=exp/exp_056_vc/exp-%j.out
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 scripts/train.py \
    -E exp/exp_056_vc
    # -L exp/exp_055_vp_asaploss/saved_models/val_vp_cfd_epoch=022.ckpt
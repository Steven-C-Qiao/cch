#!/bin/bash

#SBATCH --job-name=exp_044_th2_smpl_vc_loss_init_conf
#SBATCH --output=exp/exp_044_th2_smpl_vc_loss_init_conf/exp_044-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 scripts/train.py \
    -E exp/exp_044_th2_smpl_vc_loss_init_conf \
    -R exp/exp_044_th2_smpl_vc_loss_init_conf/saved_models/last.ckpt
    # -R exp/exp_045_fast/saved_models/last.ckpt

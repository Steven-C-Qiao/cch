#!/bin/bash

#SBATCH --job-name=exp_040
#SBATCH --output=exp/exp_040_thuman2.1/exp_040-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 scripts/train.py \
    -E exp/exp_040_thuman2.1 \
    -R /scratch/u5aa/chexuan.u5aa/cch/exp/exp_040_thuman2.1/saved_models/train_vp_cfd_epoch=006.ckpt
    # -L exp/exp_035_vp_extend_031_scale_gt2pred_loss_5k/saved_models/last-v1.ckpt
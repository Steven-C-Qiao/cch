#!/bin/bash

#SBATCH --job-name=exp_plot
#SBATCH --output=exp/exp_035_vp_extend_031_scale_gt2pred_loss_5k/plot-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev
srun python3 scripts/train.py \
    -E exp/exp_035_vp_extend_031_scale_gt2pred_loss_5k \
    -L /scratch/u5aa/chexuan.u5aa/cch/exp/exp_035_vp_extend_031_scale_gt2pred_loss_5k/saved_models/last-v1.ckpt \
    --dev 
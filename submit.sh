#!/bin/bash

#SBATCH --job-name=exp_063_alt_asaploss
#SBATCH --output=exp/exp_063_alt_asaploss/exp-%j.out
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

srun python3 scripts/train.py \
    -E exp/exp_063_alt_asaploss \
    -L exp/exp_055_vp_f_vc/saved_models/val_vc_cfd_epoch=001.ckpt
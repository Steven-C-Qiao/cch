#!/bin/bash

#SBATCH --job-name=exp_074_vc_sapiens_avgpool
#SBATCH --output=exp/exp_074_vc_sapiens_avgpool/exp-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=23:00:00

source ~/miniforge3/bin/activate dev

srun python3 scripts/train.py \
    -E exp/exp_074_vc_sapiens_avgpool
    # -L exp/exp_047_vc_sapiens/saved_models/last.ckpt
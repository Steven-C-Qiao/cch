#!/bin/bash

#SBATCH --job-name=exp_037_vp_sapiens
#SBATCH --output=exp/exp_037_vp_sapiens/exp_037_vp_sapiens-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 scripts/train.py \
    -E exp/exp_037_vp_sapiens \
    -L exp/exp_037_vp_sapiens/saved_models/last-v1.ckpt
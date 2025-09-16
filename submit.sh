#!/bin/bash

#SBATCH --job-name=exp_005
#SBATCH --output=exp/exp_005_bb_sapiens/exp_005_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

nvidia-smi

# export NCCL_DEBUG=INFO

srun python3 scripts/train.py -E exp/exp_005_bb_sapiens 
#!/bin/bash

#SBATCH --job-name=exp_003
#SBATCH --output=exp/exp_003_bb_224/exp_003_%j.out
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

nvidia-smi

# export NCCL_DEBUG=INFO

srun python3 scripts/train.py -E exp/exp_003_bb_224
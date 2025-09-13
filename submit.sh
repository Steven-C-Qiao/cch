#!/bin/bash

#SBATCH --job-name=exp_001
#SBATCH --output=exp/exp_001_isambard/exp_001_%j.out
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate debug

# export NCCL_DEBUG=INFO

srun python3 scripts/train.py -E exp/exp_001_isambard -L exp/exp_000_pbs_tune_best/saved_models/last.ckpt
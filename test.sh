#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=exp/exp_test/test-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 scripts/train.py -E exp/exp_test  -L exp/exp_032_vc_sapiens/saved_models/last-v1.ckpt --dev
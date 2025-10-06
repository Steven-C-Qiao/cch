#!/bin/bash

#SBATCH --job-name=avatar
#SBATCH --output=avatar.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 app/avatar.py -L exp/exp_033_vp_extend_031/saved_models/last.ckpt
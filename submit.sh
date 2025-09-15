#!/bin/bash

#SBATCH --job-name=exp_002
#SBATCH --output=exp/exp_002_bb/exp_002_%j.out
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev

nvidia-smi

# export NCCL_DEBUG=INFO

srun python3 scripts/train.py -E exp/exp_002_bb -R exp/exp_002_bb/saved_models/vc_cfd_epoch=000.ckpt
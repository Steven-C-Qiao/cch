#!/bin/bash

#SBATCH --job-name=exp_012_s2_no_vc_cfd
#SBATCH --output=exp/exp_012_s2_no_vc_cfd/exp_012_s2_%j.out
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00 

source /lus/lfs1aip2/home/u5au/chexuan.u5au/miniforge3/bin/activate dev

# nvidia-smi
# export NCCL_DEBUG=INFO

srun python3 scripts/train.py -E exp/exp_012_s2_no_vc_cfd  -L /scratch/u5au/chexuan.u5au/cch/exp/exp_011_s2_freeze_vc/saved_models/last.ckpt
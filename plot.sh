#!/bin/bash

#SBATCH --job-name=exp_033_vp_extend_031
#SBATCH --output=exp/exp_033_vp_extend_031/exp_033_vp_extend_031-plot-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00 

source ~/miniforge3/bin/activate dev
srun python3 scripts/train.py -E exp/exp_033_vp_extend_031 -L /scratch/u5aa/chexuan.u5aa/cch/exp/exp_033_vp_extend_031/saved_models/last.ckpt --dev
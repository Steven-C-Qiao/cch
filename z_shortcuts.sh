# Login
clifton auth

# SSH
ssh u5au.aip2.isambard
ssh u5aa.aip2.isambard

# Conda Init
source ~/miniforge3/bin/activate dev

/lus/lfs1aip2/home/u5au/chexuan.u5au

source /lus/lfs1aip2/home/u5au/chexuan.u5au/miniforge3/bin/activate
conda activate dev

python thuman_preprocess/mesh_decimation.py --job 6 --num_jobs 8

source ~/miniforge3/bin/activate dev



sbatch submit.sh
sbatch --dependency=afterok:1508361 submit_vp_fvc.sh
sbatch --dependency=afterok:1508362 submit_vp.sh
#!/bin/bash
#SBATCH -J gAFNF19a
#SBATCH --gpus 8
###SBATCH -N 1
###SBATCH -t 00:10:00
#SBATCH -t 3-00:00:00
#SBATCH --chdir=/proj/afraid/users/x_wayan/slurm_log/
#SBATCH --error=%x-%j.error
#SBATCH --output=%x-%j.out
#SBATCH -A Berzelius-2025-59 

set -exu

echo "The run starts at $(date)"

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate /proj/afraid/users/x_wayan/software/stylegan3
module load buildenv-gcccuda/12.1.1-gcc12.3.0

cd /proj/afraid/users/x_wayan/Work/Scripts/stylegan3/

# No pretrain, not used, for reference only
#python train.py --outdir=/proj/afraid/users/x_wayan/Results/stylegan3/training-runs \
#  --cfg=stylegan3-t --data=/proj/afraid/users/x_wayan/Data/AFF_train  \
#  --gpus=1 --batch=32 --gamma=2 --batch-gpu=16 --snap=10


# Use pkl file from NVlabs/stylegan3 website
#python train.py --outdir=/proj/afraid/users/x_wayan/Results/stylegan3/pretrain_model/AFF \
#	--cfg=stylegan3-t \
#	--data=/proj/afraid/users/x_wayan/Data/AFF_train_color_intermediate \
#	--gpus=8 --batch=32 --gamma=2 --snap=10 --mirror=1 --aug='ada' \
#	--cbase=16384 \
# 	--resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl
#########################

python train.py --outdir=/proj/afraid/users/x_wayan/Results/stylegan3/training-runs \
	--cfg=stylegan3-r \
	--data=/proj/afraid/users/x_wayan/Data/AFF_NFF_train_color_labels_intermediate \
	--gpus=8 --batch=32 --gamma=2 --snap=10 --mirror=1 --aug='ada' --cond=True \
	--cbase=16384


echo "The run ends at $(date)"

exit 0
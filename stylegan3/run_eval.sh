#!/bin/bash
#SBATCH -J ganeval
#SBATCH --gpus 1
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
#module load buildenv-gcccuda/12.1.1-gcc12.3.0

cd /proj/afraid/users/x_wayan/Work/Scripts/stylegan3/

python calc_metrics.py --metrics=kid50k_full,pr50k3_full,ppl2_wend  \
	--data=/proj/afraid/users/x_wayan/Results/stylegan3/training-runs/00018-stylegan3-t-AFFtrainimages-gpus1-batch32-gamma2  \
	--network=/proj/afraid/users/x_wayan/Results/stylegan3/training-runs/00018-stylegan3-t-AFFtrainimages-gpus1-batch32-gamma2/network-snapshot-000280.pkl


echo "The run ends at $(date)"

exit 0

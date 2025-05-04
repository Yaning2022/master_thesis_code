#!/bin/bash
#SBATCH -J gandtool
#SBATCH --gpus 1
###SBATCH -N 1
#SBATCH -t 01:00:00
###SBATCH -t 3-00:00:00
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

#python dataset_tool.py \
#	--source /proj/afraid/users/x_wayan/Data/AFF_train_color \
#	--dest /proj/afraid/users/x_wayan/Data/AFF_train_color_intermediate

#python dataset_tool.py \
#	--source /proj/afraid/users/x_wayan/Data/NFF_train_color \
#	--dest /proj/afraid/users/x_wayan/Data/NFF_train_color_intermediate

#python dataset_tool.py \
#       --source /proj/afraid/users/x_wayan/Data/AFF_NFF_train_color_labels \
#       --dest /proj/afraid/users/x_wayan/Data/AFF_NFF_train_color_labels_intermediate

python dataset_tool.py \
       --source /proj/afraid/users/x_wayan/Data/NFF_train_color_64 \
       --dest /proj/afraid/users/x_wayan/Data/NFF_train_color_intermediate_64

echo "The run ends at $(date)"

exit 0

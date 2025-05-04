#!/bin/bash
#SBATCH -J diffFID
###SBATCH --gpus 8 
#SBATCH --gpus 1
###SBATCH -t 00:10:00
#SBATCH -t 3-00:00:00
#SBATCH --chdir=/proj/afraid/users/x_wayan/slurm_log/
#SBATCH --error=%x-%j.error
#SBATCH --output=%x-%j.out
#SBATCH -A Berzelius-2025-59 

set -exu

echo "The run starts at $(date)"

module load Mambaforge
#mamba deactivate
mamba activate /proj/afraid/users/x_wayan/software/diffusion_eval
#module load buildenv-gcccuda/12.1.1-gcc12.3.0

# Ensure the environment variable is set
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "No conda environment is activated."
    exit 1
fi

# Continue with your script...
echo "Running in conda environment: $CONDA_DEFAULT_ENV"

cd /proj/afraid/users/x_wayan/Work/Scripts/guided-diffusion/evaluations/

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENAI_LOGDIR="/proj/afraid/users/x_wayan/Data/generate_image/stylegan3/AFFNFF256_00022/AFF_subfolder_1_evaluation"
#export OPENAI_LOGDIR="/proj/afraid/users/x_wayan/Data/generate_image/diffusion/AFF64run1NFF64run1_evaluation"
##################

python evaluator.py \
	/proj/afraid/users/x_wayan/Data/generate_image/stylegan3/AFFNFF256_00022/AFF_subfolder_1_npz.npz \
	/proj/afraid/users/x_wayan/Data/AFF_train_color_npz.npz \
	> ${OPENAI_LOGDIR} \
	2>&1

#python evaluator.py \
#       /proj/afraid/users/x_wayan/Data/generate_image/diffusion/AFF64run1NFF64run1_npz.npz \
#       /proj/afraid/users/x_wayan/Data/AFF_NFF_train_color_npz.npz \
#       > ${OPENAI_LOGDIR} \
        #2>&1

echo "The run ends at $(date)"

exit 0

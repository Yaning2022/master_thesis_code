#!/bin/bash
#SBATCH -J difAFF64
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
mamba activate /proj/afraid/users/x_wayan/software/diffusion2
#module load buildenv-gcccuda/12.1.1-gcc12.3.0

# Ensure the environment variable is set
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "No conda environment is activated."
    exit 1
fi

# Continue with your script...
echo "Running in conda environment: $CONDA_DEFAULT_ENV"

cd /proj/afraid/users/x_wayan/Work/Scripts/guided-diffusion/

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENAI_LOGDIR="/proj/afraid/users/x_wayan/Results/diffusion/AFF/AFF64_run2"

path_to_images="/proj/afraid/users/x_wayan/Data/AFF_train_color_64"


# change these 4 parameters
#####################
#AFF64_run1

#MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
#DIFFUSION_FLAGS="--diffusion_steps 2000 --noise_schedule linear"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
####################
#####################
#AFF64_run2

MODEL_FLAGS="--image_size 64 --num_channels 64 --num_res_blocks 2"
DIFFUSION_FLAGS="--diffusion_steps 2000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 256"
####################


python scripts/image_train.py --data_dir $path_to_images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
#mpiexec -n 8 python scripts/image_train.py --data_dir $path_to_images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

echo "The run ends at $(date)"

exit 0
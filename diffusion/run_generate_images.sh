#!/bin/bash
#SBATCH -J difgeAFF
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
export OPENAI_LOGDIR="/proj/afraid/users/x_wayan/Results/diffusion/NFF/NFF64_run2/generate_NFF"

#####################
#AFF256_run1

# change these 4 parameters
#path_to_model="/proj/afraid/users/x_wayan/Results/diffusion/AFF/AFF256_run1/ema_0.9999_080000.pt"
#MODEL_FLAGS="--image_size 256 --num_channels 64 --num_res_blocks 2"
#DIFFUSION_FLAGS="--diffusion_steps 2000 --noise_schedule linear"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --microbatch 4"
#####################
#AFF256_run2
#path_to_model="/proj/afraid/users/x_wayan/Results/diffusion/AFF/AFF256_run2/ema_0.9999_070000.pt"
#MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3"
#DIFFUSION_FLAGS="--diffusion_steps 2000 --noise_schedule linear"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --microbatch 4"
#####################
#AFF64_run1
#path_to_model="/proj/afraid/users/x_wayan/Results/diffusion/AFF/AFF64_run1/ema_0.9999_340000.pt"

#MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
#DIFFUSION_FLAGS="--diffusion_steps 2000 --noise_schedule linear"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
####################

#NFF64_run1
path_to_model="/proj/afraid/users/x_wayan/Results/diffusion/NFF/NFF64_run1/ema_0.9999_370000.pt"

MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 2000 --noise_schedule linear"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
######################
#650 AFF, 2561 NFF

#python scripts/image_sample.py --model_path ${path_to_model} $MODEL_FLAGS $DIFFUSION_FLAGS
python scripts/image_sample.py --model_path ${path_to_model} $MODEL_FLAGS $DIFFUSION_FLAGS \
        --num_samples 2561

echo "The run ends at $(date)"

exit 0

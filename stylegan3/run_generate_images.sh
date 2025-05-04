#!/bin/bash
#SBATCH -J gprgeimg
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

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate /proj/afraid/users/x_wayan/software/stylegan3
module load buildenv-gcccuda/12.1.1-gcc12.3.0

cd /proj/afraid/users/x_wayan/Work/Scripts/stylegan3/


#NFF: 0-2560,2561-5121,5122-7682,7683-10243,10244-12804,12805-15365,15366-17926,17927-20487,20488-23048,23049-25609, AFF:0-649,650-1299,1300-1949,1950-2599,2600-3249,3250-3899,3900-4549,4550-5199,5200-5849,5850-6499

python gen_images.py --outdir=/proj/afraid/users/x_wayan/Results/stylegan3/training-runs/00040-stylegan3-r-NFF_train_color_intermediate-gpus8-batch32-gamma2/generate_images10 --trunc=1 --seeds=23049-25609 \--network=/proj/afraid/users/x_wayan/Results/stylegan3/training-runs/00040-stylegan3-r-NFF_train_color_intermediate-gpus8-batch32-gamma2/network-snapshot-024960.pkl


echo "The run ends at $(date)"

exit 0

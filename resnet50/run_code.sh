#!/bin/bash

set -exu


echo "The run starts at $(date)"


#PATH_OUT folder created automatically
export PATH_OUT="/local/data1/yanwa579/Results/resnet50/Run31_l15_avepool"
#export TRAIN_DIRECTORY="/local/data1/yanwa579/Data/resnet50_data/train_gan_label"
export TRAIN_DIRECTORY="/local/data1/yanwa579/Data/resnet50_data/train"
#export TRAIN_DIRECTORY="/local/data1/yanwa579/Data/generate_images/diffusion/AFF64run1NFF64run1"
export VALID_DIRECTORY="/local/data1/yanwa579/Data/resnet50_data/valid"



python resnet50.py


echo "The run ends at $(date)"


exit 0

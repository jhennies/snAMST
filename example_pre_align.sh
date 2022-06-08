#!/bin/bash

#############################################################
# PARAMETERS

# Data
source_folder=/scratch/path/to/source/
target_folder=/scratch/path/to/results/

# Workflow parameters
local_auto_mask=10      # Automatically remove SIFT key-points which are at the edge of the actual (non-zero) data

# Compute settings
local_device_type=CPU   # CPU is currently more efficient for the cluster
cores=64                # Set to 0 for running locally with all cores, otherwise specify a value > 0
gpus=1                  # Has no effect if local_device_type=CPU
cluster=slurm           # Can be "none" or "slurm"

# Installation settings (normally don't change anything here)
conda_path=/g/icem/hennies/software/miniconda3
env_path=/g/icem/hennies/envs
src_path=/g/icem/hennies/src/github/jhennies/snamst

#############################################################

if [ $local_device_type = GPU ]; then
  env_path=${env_path}/snamst-env
else
  env_path=${env_path}/snamst-no-gpu-env
fi

export PYTHONPATH="$src_path:$PYTHONPATH"
source "${conda_path}/bin/activate" "${env_path}"

which python

python ${src_path}/run_pre_align.py \
-sf $source_folder \
-tf $target_folder \
-lam sift \
-lau $local_auto_mask \
-ldt $local_device_type \
-apd \
-c $cores \
-g $gpus \
--cluster $cluster \
-v

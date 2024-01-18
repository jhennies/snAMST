#!/bin/bash

#############################################################
# PARAMETERS

# Data
source_folder=/scratch/path/to/source/
target_folder=/scratch/path/to/results/
template_filepath=/scratch/path/to/template/

# Workflow parameters
local_align_method=sift         # "sift" or "xcorr", in most cases SIFT is better
local_auto_mask=10              # Automatically remove SIFT key-points which are at the edge of the actual (non-zero) data
local_norm_quantile_low=0.01    # How to normalize the data (lower and upper bound using quantiles)
local_norm_quantile_high=0.99

# These parameters can be used to the threshold the data before runnin SIFT alignment. However, I currently don't
# recommend using these.
local_mask_range_low=0
local_mask_range_high=0
local_thresh_low=0
local_thresh_high=0

# Template matching parameters
tm_threshold_low=0
tm_threshold_high=0

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
-lam $local_align_method \
-lau $local_auto_mask \
-lnq $local_norm_quantile_low $local_norm_quantile_high \
-lmr $local_mask_range_low $local_mask_range_high \
-lth $local_thresh_low $local_thresh_high \
-ldt $local_device_type \
-tm $template_filepath \
-tmt $tm_threshold_low $tm_threshold_high \
-apd \
-c $cores \
-g $gpus \
--cluster $cluster \
-v

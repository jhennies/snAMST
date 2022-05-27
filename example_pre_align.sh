#!/bin/bash

#############################################################
# PARAMETERS

# Data
source_folder=/g/icem/hennies/datasets/steyer_Instruments_Crossbeam_Volume-imaging/2022-02-16_PF_as43a_Acantharia/Original/Images_no_defects/
target_folder=/scratch/hennies/tmp/snamst_test/

# Workflow parameters
local_auto_mask=10

# Cluster settings
cores=64
gpus=32

# Installation settings (normally don't change anything here)
conda_path=/g/icem/hennies/software/miniconda3
env_path=/g/icem/hennies/envs/snamst-env
src_path=/g/icem/hennies/src/github/jhennies/snamst


#############################################################


export PYTHONPATH="$src_path:$PYTHONPATH"
source "${conda_path}/bin/activate" "${env_path}"

which python

cd $src_path

python run_pre_align.py -sf $source_folder -tf $target_folder -lam sift -lau $local_auto_mask -apd -c $cores -g $gpus -v --cluster slurm



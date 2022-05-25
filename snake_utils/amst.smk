
import sys
sys.path.append('..')

from amst_utils.common.settings import get_run_info, get_params

import os
from glob import glob
import json

src_path = os.getcwd()

# Get the run info
run_info = get_run_info()
params = get_params()

# Parameters set by user
source_folder = run_info['source_folder']
target_folder = run_info['target_folder']
im_list = run_info['im_list']
im_names = run_info['im_names']
verbose = run_info['verbose']

# Parameters which are relevant to build the snakemake workflow
median_with_one_rule = params['median_with_one_rule']
use_coarse_align = params['use_coarse_align']


rule all:
    input:
        expand(os.path.join(target_folder, "amst", "{name}"), name=im_names)


def affine_registration_inputs(wildcards):
    inputs = [os.path.join(source_folder, "{name}")]
    if use_coarse_align:
        raise NotImplementedError
    else:
        inputs.append(os.path.join(target_folder, "amst_cache", "median_smoothing", "{name}.json"))
    return inputs

rule affine_registration:
    input:
        affine_registration_inputs
    output:
        os.path.join(target_folder, "amst", "{name}")
    threads: 1
    script:
        os.path.join(src_path, "amst_utils", "affine_registration.py")


if use_coarse_align:
    # This step is not needed if the pre-align workflow with autopad is used.
    # However, it might improve the result for a pre-align without local alignment.
    raise NotImplementedError
    rule sift_coarse_align:
        input:


if median_with_one_rule:
    # Only one rule for the full dataset as that reduces the amount of data IO drastically
    raise NotImplementedError
    rule median_smoothing:
        input:
else:
    rule median_smoothing:
        input:
            os.path.join(source_folder, "{name}")
        output:
            os.path.join(target_folder, "amst_cache", "median_smoothing", "{name}.json")
        threads: 1
        script:
            os.path.join(src_path, "amst_utils", "median_smoothing_single.py")

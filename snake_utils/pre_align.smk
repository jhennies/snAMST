
import sys
sys.path.append('..')

from amst_utils.common.settings import get_run_info, get_params
from amst_utils.common.param_handling import replace_special

import os
from glob import glob
import json

# Get the run info
run_info = get_run_info()
params = get_params()

# Parameters set by user
source_folder = run_info['source_folder']
target_folder = run_info['target_folder']
im_list = run_info['im_list']
im_names = run_info['im_names']
verbose = run_info['verbose']

# The list of reference image slices
ref_list = [None] + im_list[:-1]
ref_dict = dict(zip(im_names, ref_list))

# Parameters which are relevant to build the snakemake workflow
use_local = 'local' in params.keys() and params['local'] is not None
use_tm = 'tm' in params.keys() and params['tm'] is not None
print(f'params["tm"] = {params["tm"]}')
print(f'use_tm = {use_tm}')
print(f'use_local = {use_local}')
if verbose:
    print(f'im_names = {im_names}')


rule all:
    input:
        expand(os.path.join(target_folder, "pre_align", "{name}"), name=im_names)
    params:
        p='htc', gres=''


rule apply_translations:
    input:
        os.path.join(source_folder, "{name}"),
        os.path.join(target_folder, "pre_align_cache", "final_offsets.json")
    output:
        os.path.join(target_folder, "pre_align", "{name}")
    threads: 1
    resources:
        cpus=1, time_min=10, mem_mb=512
    params:
        p='htc', gres=''
    script:
        os.path.join("..", "amst_utils", "apply_translations.py")


def combine_translation_inputs(wildcards):
    inputs = []
    if use_local:
        inputs.extend(
            expand(os.path.join(target_folder, "pre_align_cache", "offsets_local", "{name}.json"), name=im_names)
        )
    if use_tm:
        inputs.extend(
            expand(os.path.join(target_folder, "pre_align_cache", "offsets_tm", "{name}.json"), name=im_names)
        )
    print(inputs)
    return inputs


rule combine_translations:
    input:
        combine_translation_inputs
    output:
        os.path.join(target_folder, "pre_align_cache", "final_offsets.json")
    threads: 1
    resources:
        cpus=1, time_min=10, mem_mb=512
    params:
        p='htc', gres=''
    script:
        os.path.join("..", "amst_utils", "combine_translations.py")


if use_tm:
    rule template_matching:
        input:
            os.path.join(source_folder, "{name}")
        output:
            os.path.join(target_folder, "pre_align_cache", "offsets_tm", "{name}.json")
        threads: 1
        resources:
            cpus=1, time_min=10, mem_mb=512
        params:
            p='htc', gres=''
        script:
            os.path.join("..", "amst_utils", "template_matching.py")


def get_ref_im(wildcards):
    ref_im = replace_special(ref_dict[wildcards.name])
    return ref_im

if use_local:
    rule local_alignment:
        input:
            os.path.join(source_folder, "{name}")
        output:
            os.path.join(target_folder, "pre_align_cache", "offsets_local", "{name}.json")
        threads: 1
        resources:
            gpu=1 if params['local']['align_method'] == 'sift' and params['local']['device_type'] == 'GPU' else 0,
            cpus=1, time_min=10, mem_mb=2048
        params:
            ref_im=get_ref_im,
            p='gpu' if params['local']['align_method'] == 'sift' and params['local']['device_type'] == 'GPU' else 'htc',
            gres='--gres=gpu:1' if params['local']['align_method'] == 'sift' and params['local']['device_type'] == 'GPU' else ''
        script:
            os.path.join("..", "amst_utils", "local_alignment.py")

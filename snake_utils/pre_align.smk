
import os
from glob import glob
import json
from settings import get_run_info_fp

src_path = os.getcwd()

# Get the run info
run_info_fp = get_run_info_fp()
with open(run_info_fp, mode='r') as f:
    run_info = json.load(f)

# Parameters set by user
source_folder = run_info['source_folder']
target_folder = run_info['target_folder']
verbose = run_info['verbose']

# The list of image slices
im_list = sorted(glob(os.path.join(source_folder, '*.tif')))
im_names = [os.path.split(fp)[1] for fp in im_list]


rule all:
    input:
        expand(os.path.join(target_folder, "pre_align", "{name}"), name=im_names)


rule apply_translations:
    input:
        os.path.join(source_folder, "{name}"),
        os.path.join(target_folder, "cache", "final_offsets.json")
    output:
        os.path.join(target_folder, "pre_align", "{name}")
    threads: 1
    script:
        os.path.join(src_path, "amst_utils", "apply_translations.py")


rule combine_translations:
    input:
        expand(
            os.path.join(target_folder, "cache", "offsets_local", "{name}.json"),
            name=im_names
        ),
        expand(
            os.path.join(target_folder, "cache", "offsets_tm", "{name}.json"),
            name=im_names
        )
    output:
        os.path.join(target_folder, "cache", "final_offsets.json")
    threads: 1
    script:
        os.path.join(src_path, "amst_utils", "combine_translations.py")


rule template_matching:
    input:
        os.path.join(source_folder, "{name}")
    output:
        os.path.join(target_folder, "cache", "offsets_tm", "{name}.json")
    threads: 1
    script:
        os.path.join(src_path, "amst_utils", "template_matching.py")


rule local_alignment:
    input:
        os.path.join(source_folder, "{name}")
    output:
        os.path.join(target_folder, "cache", "offsets_local", "{name}.json")
    threads: 1
    script:
        os.path.join(src_path, "amst_utils", "local_alignment.py")

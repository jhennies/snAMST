
######## snakemake preamble start (automatically inserted, do not edit) ########
import sys; sys.path.extend(['/g/icem/hennies/envs/snamst-env/lib/python3.9/site-packages', '/g/icem/hennies/src/github/jhennies/snamst/amst_utils']); import pickle; snakemake = pickle.loads(b'\x80\x04\x95\xa3\x10\x00\x00\x00\x00\x00\x00\x8c\x10snakemake.script\x94\x8c\tSnakemake\x94\x93\x94)\x81\x94}\x94(\x8c\x05input\x94\x8c\x0csnakemake.io\x94\x8c\nInputFiles\x94\x93\x94)\x81\x94(\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00000_z=0.1555um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00001_z=0.1568um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00002_z=0.1964um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00003_z=0.2013um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00004_z=0.2288um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00005_z=0.2512um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00006_z=0.2686um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00007_z=0.2831um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00008_z=0.2963um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00009_z=0.3083um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00010_z=0.3200um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00011_z=0.3314um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00012_z=0.3503um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00013_z=0.3475um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00014_z=0.3561um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00015_z=0.3702um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00016_z=0.3781um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00017_z=0.3866um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00018_z=0.3949um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00019_z=0.4034um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00020_z=0.4117um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00021_z=0.4203um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00022_z=0.4101um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00023_z=0.4185um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00024_z=0.4269um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00025_z=0.4353um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00026_z=0.4438um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00027_z=0.4522um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00028_z=0.4608um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00029_z=0.4464um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00030_z=0.4548um.tif.json\x94\x8c^/scratch/hennies/tmp/snamst_test/pre_align_cache/offsets_local/slice_00031_z=0.4633um.tif.json\x94e}\x94(\x8c\x06_names\x94}\x94\x8c\x12_allowed_overrides\x94]\x94(\x8c\x05index\x94\x8c\x04sort\x94eh/\x8c\tfunctools\x94\x8c\x07partial\x94\x93\x94h\x06\x8c\x19Namedlist._used_attribute\x94\x93\x94\x85\x94R\x94(h5)}\x94\x8c\x05_name\x94h/sNt\x94bh0h3h5\x85\x94R\x94(h5)}\x94h9h0sNt\x94bub\x8c\x06output\x94h\x06\x8c\x0bOutputFiles\x94\x93\x94)\x81\x94\x8cC/scratch/hennies/tmp/snamst_test/pre_align_cache/final_offsets.json\x94a}\x94(h+}\x94h-]\x94(h/h0eh/h3h5\x85\x94R\x94(h5)}\x94h9h/sNt\x94bh0h3h5\x85\x94R\x94(h5)}\x94h9h0sNt\x94bub\x8c\x06params\x94h\x06\x8c\x06Params\x94\x93\x94)\x81\x94(\x8c\x03htc\x94\x8c\x00\x94e}\x94(h+}\x94(\x8c\x01p\x94K\x00N\x86\x94\x8c\x04gres\x94K\x01N\x86\x94uh-]\x94(h/h0eh/h3h5\x85\x94R\x94(h5)}\x94h9h/sNt\x94bh0h3h5\x85\x94R\x94(h5)}\x94h9h0sNt\x94bhWhShYhTub\x8c\twildcards\x94h\x06\x8c\tWildcards\x94\x93\x94)\x81\x94}\x94(h+}\x94h-]\x94(h/h0eh/h3h5\x85\x94R\x94(h5)}\x94h9h/sNt\x94bh0h3h5\x85\x94R\x94(h5)}\x94h9h0sNt\x94bub\x8c\x07threads\x94K\x01\x8c\tresources\x94h\x06\x8c\tResources\x94\x93\x94)\x81\x94(K\x01K\x01M\x00\x02M\xe8\x03\x8c\x16/scratch/jobs/40768796\x94K\x01K\ne}\x94(h+}\x94(\x8c\x06_cores\x94K\x00N\x86\x94\x8c\x06_nodes\x94K\x01N\x86\x94\x8c\x06mem_mb\x94K\x02N\x86\x94\x8c\x07disk_mb\x94K\x03N\x86\x94\x8c\x06tmpdir\x94K\x04N\x86\x94\x8c\x04cpus\x94K\x05N\x86\x94\x8c\x08time_min\x94K\x06N\x86\x94uh-]\x94(h/h0eh/h3h5\x85\x94R\x94(h5)}\x94h9h/sNt\x94bh0h3h5\x85\x94R\x94(h5)}\x94h9h0sNt\x94bh{K\x01h}K\x01h\x7fM\x00\x02h\x81M\xe8\x03h\x83hxh\x85K\x01h\x87K\nub\x8c\x03log\x94h\x06\x8c\x03Log\x94\x93\x94)\x81\x94}\x94(h+}\x94h-]\x94(h/h0eh/h3h5\x85\x94R\x94(h5)}\x94h9h/sNt\x94bh0h3h5\x85\x94R\x94(h5)}\x94h9h0sNt\x94bub\x8c\x06config\x94}\x94\x8c\x04rule\x94\x8c\x14combine_translations\x94\x8c\x0fbench_iteration\x94N\x8c\tscriptdir\x94\x8c5/g/icem/hennies/src/github/jhennies/snamst/amst_utils\x94ub.'); from snakemake.logging import logger; logger.printshellcmds = False; __real_file__ = __file__; __file__ = '/g/icem/hennies/src/github/jhennies/snamst/amst_utils/combine_translations.py';
######## snakemake preamble end #########
import os.path

from tifffile import imread
import json
import numpy as np

from amst_utils.common.settings import get_params, get_run_info
from amst_utils.common.displacement import smooth_offsets, compute_auto_pad, sequentialize_offsets


def combine_translations(
        local_offsets_fps,
        tm_offsets_fps,
        result_fp,
        im_list,
        bounds_fps=None,
        smooth_median=8,
        smooth_sigma=8,
        suppress_x=False,
        verbose=False,
):

    local_offsets = None
    tm_offsets = None
    bounds = None
    if local_offsets_fps is not None:
        local_offsets = np.array([json.load(open(fp, mode='r'))['offset'] for fp in local_offsets_fps])
    if tm_offsets_fps is not None:
        tm_offsets = np.array([json.load(open(fp, mode='r'))['offset'] for fp in tm_offsets_fps])
    if bounds_fps is not None:
        bounds = np.array([json.load(open(fp, mode='r'))['bounds'] for fp in bounds_fps])

    if verbose:
        print(f'local_offsets = {local_offsets}')
        print(f'tm_offsets = {tm_offsets}')
        print(f'bounds = {bounds}')

    if local_offsets is not None:
        local_offsets = sequentialize_offsets(local_offsets)

    if local_offsets is not None and tm_offsets is not None:

        # The final offsets according to the formula OFFSETS = LOCAL + smoothed(TM - LOCAL)
        offsets = local_offsets + smooth_offsets(
            tm_offsets - local_offsets,
            median_radius=smooth_median,
            gaussian_sigma=smooth_sigma,
            suppress_x=suppress_x
        )

    elif local_offsets is None:
        offsets = tm_offsets
    elif tm_offsets is None:
        offsets = local_offsets

    shape = None
    if bounds is not None:
        offsets, shape = compute_auto_pad(offsets, bounds)

    if verbose:
        print(f'offsets = {offsets}')
        print(f'shape = {shape}')

    with open(result_fp, mode='w') as f:
        json.dump(
            dict(
                im_list=im_list,
                offsets=offsets.tolist(),
                bounds=None if bounds is None else bounds.tolist(),
                shape=None if shape is None else shape.tolist()
            ), f, indent=2
        )


if __name__ == '__main__':

    # Snakemake inputs
    inputs = snakemake.input
    output = snakemake.output[0]
    # Get parameters
    params = get_params()
    combine_median = params['combine_median']
    combine_sigma = params['combine_sigma']
    auto_pad = params['auto_pad']
    verbose = params['verbose']
    # Get list of images
    im_list = get_run_info()['im_list']

    # Sort the inputs
    use_local = 'local' in params.keys() and params['local'] is not None
    use_tm = 'tm' in params.keys() and params['tm'] is not None
    bounds_fps = None
    if use_local and use_tm:
        local_offsets_fps = sorted(inputs[:int(len(inputs) / 2)])
        tm_offsets_fps = sorted(inputs[int(len(inputs) / 2):])
        if auto_pad:
            bounds_fps = local_offsets_fps
    elif use_local and not use_tm:
        local_offsets_fps = sorted(inputs)
        if auto_pad:
            bounds_fps = local_offsets_fps
        tm_offsets_fps = None
    elif not use_local and use_tm:
        tm_offsets_fps = sorted(inputs)
        if auto_pad:
            bounds_fps = tm_offsets_fps
        local_offsets_fps = None
    else:
        raise RuntimeError('Either local or template matching alignment must be used!')

    if verbose:
        print(f'local_offsets_fps = {local_offsets_fps}')
        print(f'tm_offsets_fps = {tm_offsets_fps}')

    combine_translations(
        local_offsets_fps,
        tm_offsets_fps,
        output,
        im_list,
        smooth_median=combine_median,
        smooth_sigma=combine_sigma,
        bounds_fps=bounds_fps,
        verbose=verbose
    )

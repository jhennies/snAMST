import os.path

from tifffile import imread
import json
import numpy as np

from amst_utils.common.settings import get_params, get_run_info
from amst_utils.common.displacement import smooth_offsets, compute_auto_pad, sequentialize_offsets
from amst_utils.common.displacement import apply_running_average


def combine_translations(
        local_offsets_fps,
        tm_offsets_fps,
        result_fp,
        im_list,
        bounds_fps=None,
        smooth_median=8,
        smooth_sigma=8,
        suppress_x=False,
        subtract_running_average=False,
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

    if subtract_running_average:
        offsets = apply_running_average(offsets)

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
    subtract_running_average = params['subtract_running_average']
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
        subtract_running_average=subtract_running_average,
        verbose=verbose
    )

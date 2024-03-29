
import json
import os.path

from amst_utils.common.param_handling import replace_special
from amst_utils.common.settings import get_params
from amst_utils.common.data import get_bounds

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

from .local_alignment import local_alignment


def local_alignment_batch(
        im_fps,
        ref_im_fps,
        result_fps,
        align_method='sift',
        mask_range=None,
        sigma=1.6,
        norm_quantiles=(0.1, 0.9),
        device_type='GPU',
        save_bounds=False,
        auto_mask=None,
        verbose=False
):

    for idx, im_fp in enumerate(im_fps):
        local_alignment(
            im_fp,
            ref_im_fps[idx],
            result_fps[idx],
            align_method=align_method,
            mask_range=mask_range,
            sigma=sigma,
            norm_quantiles=norm_quantiles,
            device_type=device_type,
            save_bounds=save_bounds,
            auto_mask=auto_mask,
            verbose=verbose
        )


if __name__ == '__main__':

    # Snakemake inputs
    ims = snakemake.input
    outputs = snakemake.output
    assert len(inputs) == len(outputs)
    ref_ims = snakemake.params['ref_ims']

    if snakemake.params['ref_im'] is not None:
        ref_im = os.path.join(
            os.path.split(im)[0],
            replace_special(snakemake.params['ref_im'], back=True)
        )
    else:
        ref_im = None
    # Get parameters from run_info
    params = get_params()
    local_align_method = params['local']['align_method']
    local_mask_range = params['local']['mask_range']
    local_sigma = params['local']['sigma']
    local_norm_quantiles = params['local']['norm_quantiles']
    local_device_type = params['local']['device_type']
    local_auto_mask = params['local']['auto_mask']
    auto_pad = params['auto_pad']
    verbose = params['verbose']

    if verbose:
        print(f'im = {im}')
        print(f'ref_im = {ref_im}')
        print(f'out = {output}')

    local_alignment_batch(
        im, ref_im,
        output,
        align_method=local_align_method,
        mask_range=local_mask_range,
        sigma=local_sigma,
        norm_quantiles=local_norm_quantiles,
        device_type=local_device_type,
        save_bounds=auto_pad,
        auto_mask=local_auto_mask,
        verbose=verbose
    )

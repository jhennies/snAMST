
import json
import os.path

from amst_utils.common.param_handling import replace_special
from amst_utils.common.settings import get_params
from amst_utils.common.data import get_bounds

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


def local_alignment(
        im_fp,
        ref_im_fp,
        result_fp,
        align_method='sift',
        mask_range=None,
        thresh=None,
        sigma=1.6,
        norm_quantiles=(0.1, 0.9),
        device_type='GPU',
        save_bounds=False,
        auto_mask=None,
        max_offset=None,
        xy_range=None,
        invert_nonzero=False,
        mask_im_fp=None,
        downsample=1,
        bias=(0., 0.),
        big_jump_prefix=False,
        verbose=False
):

    if ref_im_fp is None:

        offset = [0., 0.]
        bounds = get_bounds(im_fp) if save_bounds else None

    else:

        if align_method == 'sift':

            from amst_utils.common.sift import offset_with_sift
            out = offset_with_sift(
                im_fp, ref_im_fp,
                mask_range=mask_range,
                thresh=thresh,
                sigma=sigma,
                norm_quantiles=norm_quantiles,
                device_type=device_type,
                return_bounds=save_bounds,
                auto_mask=auto_mask,
                max_offset=max_offset,
                xy_range=xy_range,
                invert_nonzero=invert_nonzero,
                mask_im_fp=mask_im_fp,
                downsample=downsample,
                verbose=verbose
            )
            if verbose:
                print(f'out = {out}')
            offset, bounds = out if save_bounds else (out, None)

        elif align_method == 'sift_opencv':

            from amst_utils.common.sift_opencv import offset_with_sift
            out = offset_with_sift(
                im_fp, ref_im_fp,
                mask_range=mask_range,
                thresh=thresh,
                sigma=sigma,
                norm_quantiles=norm_quantiles,
                return_bounds=save_bounds,
                auto_mask=auto_mask,
                max_offset=max_offset,
                xy_range=xy_range,
                invert_nonzero=invert_nonzero,
                mask_im_fp=mask_im_fp,
                downsample=downsample,
                bias=bias,
                verbose=verbose
            )
            if verbose:
                print(f'out = {out}')
            offset, bounds = out if save_bounds else (out, None)

        elif align_method == 'elastix':

            from amst_utils.common.elastix import offset_with_elastix
            out = offset_with_elastix(
                im_fp, ref_im_fp,
                mask_range=mask_range,
                thresh=thresh,
                sigma=sigma,
                norm_quantiles=norm_quantiles,
                return_bounds=save_bounds,
                auto_mask=auto_mask,
                max_offset=max_offset,
                xy_range=xy_range,
                invert_nonzero=invert_nonzero,
                mask_im_fp=mask_im_fp,
                downsample=downsample,
                bias=bias,
                big_jump_prefix=big_jump_prefix,
                verbose=verbose
            )
            if verbose:
                print(f'out = {out}')
            offset, bounds = out if save_bounds else (out, None)

        elif align_method == 'xcorr':

            from amst_utils.common.xcorr import offsets_with_xcorr
            out = offsets_with_xcorr(
                im_fp, ref_im_fp,
                mask_range=mask_range,
                thresh=thresh,
                sigma=sigma,
                norm_quantiles=norm_quantiles,
                return_bounds=save_bounds,
                auto_mask=auto_mask,
                mask_im_fp=mask_im_fp,
                verbose=verbose
            )
            if verbose:
                print(f'out = {out}')
            offset, bounds = out if save_bounds else (out, None)

        else:
            raise ValueError(f'Unknown alignment method: {align_method}')

    if save_bounds:

        if verbose:
            print(f'bounds = {bounds}')
            print(type(bounds))
            print(f'offset = {offset}')
            print(type(offset))

        with open(result_fp, mode='w') as f:
            json.dump(dict(offset=offset, bounds=bounds), f, indent=2)

    else:

        with open(result_fp, mode='w') as f:
            json.dump(dict(offset=offset), f, indent=2)


if __name__ == '__main__':

    # Snakemake inputs
    im = snakemake.input[0]
    output = snakemake.output[0]
    if snakemake.params['ref_im'] is not None:
        ref_im = os.path.join(
            os.path.split(im)[0],
            replace_special(snakemake.params['ref_im'], back=True)
        )
    else:
        ref_im = None
    mask_im = snakemake.params['mask_im']
    # Get parameters from run_info
    params = get_params()
    local_align_method = params['local']['align_method']
    local_mask_range = params['local']['mask_range']
    local_thresh = params['local']['thresh']
    local_sigma = params['local']['sigma']
    local_norm_quantiles = params['local']['norm_quantiles']
    local_device_type = params['local']['device_type']
    local_auto_mask = params['local']['auto_mask']
    local_max_offset = params['local']['max_offset']
    local_xy_range = params['local']['xy_range']
    local_invert_nonzero = params['local']['invert_nonzero']
    local_downsample = params['local']['downsample']
    local_bias = params['local']['bias']
    local_big_jump_prefix = params['local']['big_jump_prefix']
    auto_pad = params['auto_pad']
    verbose = params['verbose']

    if verbose:
        print(f'im = {im}')
        print(f'ref_im = {ref_im}')
        print(f'out = {output}')

    local_alignment(
        im, ref_im,
        output,
        align_method=local_align_method,
        mask_range=local_mask_range,
        thresh=local_thresh,
        sigma=local_sigma,
        norm_quantiles=local_norm_quantiles,
        device_type=local_device_type,
        save_bounds=auto_pad,
        auto_mask=local_auto_mask,
        max_offset=local_max_offset,
        xy_range=local_xy_range,
        invert_nonzero=local_invert_nonzero,
        mask_im_fp=mask_im,
        downsample=local_downsample,
        bias=local_bias,
        big_jump_prefix=local_big_jump_prefix,
        verbose=verbose
    )

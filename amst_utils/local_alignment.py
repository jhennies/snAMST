
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
        sigma=1.6,
        norm_quantiles=(0.1, 0.9),
        device_type='GPU',
        save_bounds=False,
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
                sigma=sigma,
                norm_quantiles=norm_quantiles,
                device_type=device_type,
                return_bounds=save_bounds,
                verbose=verbose
            )
            if verbose:
                print(f'out = {out}')
            offset, bounds = out if save_bounds else (out, None)

        elif align_method == 'xcorr':
            raise NotImplementedError()
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
    # Get parameters from run_info
    params = get_params()
    local_align_method = params['local']['align_method']
    local_mask_range = params['local']['mask_range']
    local_sigma = params['local']['sigma']
    local_norm_quantiles = params['local']['norm_quantiles']
    local_device_type = params['local']['device_type']
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
        sigma=local_sigma,
        norm_quantiles=local_norm_quantiles,
        device_type=local_device_type,
        save_bounds=auto_pad,
        verbose=verbose
    )

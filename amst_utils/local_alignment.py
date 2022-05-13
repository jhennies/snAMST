
import json
import os.path

from amst_utils.common.param_handling import replace_special
from amst_utils.common.settings import get_params
from amst_utils.common.data import get_bounds


def local_alignment(
        im_fp,
        ref_im_fp,
        result_fp,
        align_method='sift',
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

        print(f'bounds = {bounds}')
        print(type(bounds))
        # print(bounds.dtype)
        print(f'offset = {offset}')
        print(type(offset))
        # print(offset.dtype)

        with open(result_fp, mode='w') as f:
            json.dump(dict(offset=offset, bounds=bounds), f, indent=2)

    else:

        with open(result_fp, mode='w') as f:
            json.dump(dict(offset=offset), f, indent=2)


if __name__ == '__main__':

    # Snakemake inputs
    im = snakemake.input[0]
    output = snakemake.output[0]
    ref_im = os.path.join(
        os.path.split(im)[0],
        replace_special(snakemake.params['ref_im'], back=True)
    )
    # Get parameters from run_info
    params = get_params()
    local_align_method = params['local']['align_method']
    auto_pad = params['auto_pad']
    verbose = params['verbose']

    if verbose:
        print(f'im = {im}')
        print(f'ref_im = {ref_im}')

    local_alignment(
        im, ref_im,
        output,
        align_method=local_align_method,
        save_bounds=auto_pad,
        verbose=verbose
    )


import json
import numpy as np

from amst_utils.common.settings import get_params
from amst_utils.common.displacement import displace_slice, bounds2slice


def apply_translations(
        im_fp,
        offsets_fp,
        result_fp,
        verbose=False
):

    if verbose:
        print(f'im_fp = {im_fp}')

    with open(offsets_fp, mode='r') as f:
        data = json.load(f)
        im_list = data['im_list']
        idx = im_list.index(im_fp)
        offset = data['offsets'][idx]
        bounds = data['bounds'][idx]
        shape = data['shape']
        del data

    if verbose:
        print(f'im_list = {im_list}')
        print(f'idx = {idx}')
        print(f'offsets = {offset}')
        print(f'bounds = {bounds}')
        print(f'shape = {shape}')

    bounds = bounds2slice(bounds)

    displace_slice(
        result_fp, im_fp, offset, subpx_displacement=True,
        compression=['zlib', 9], pad_zeros=16, bounds=bounds if bounds is not None else np.s_[:],
        target_shape=shape
    )


if __name__ == '__main__':

    # Snakemake inputs
    im_fp = snakemake.input[0]
    final_offsets_fp = snakemake.input[1]
    result_fp = snakemake.output[0]
    # Get parameters from run_info
    params = get_params()
    auto_pad = params['auto_pad']
    verbose = params['verbose']

    if verbose:
        print(f'im_fp = {im_fp}')
        print(f'final_offsets_fp = {final_offsets_fp}')

    apply_translations(
        im_fp,
        final_offsets_fp,
        result_fp,
        verbose=verbose
    )


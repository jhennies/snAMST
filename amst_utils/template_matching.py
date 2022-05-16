
import json
import numpy as np
from amst_utils.common.settings import get_params
from amst_utils.common.tm import offsets_with_tm


def template_matching(
        im_fp,
        template_fp,
        result_fp,
        threshold=(0, 0),
        sigma=0,
        add_offset=None,
        save_bounds=False,
        verbose=False
):

    out = offsets_with_tm(
        im_fp,
        template_fp,
        threshold=threshold,
        sigma=sigma,
        return_bounds=save_bounds
    )

    if verbose:
        print(f'out = {out}')
    offset, bounds = out if save_bounds else (out, None)

    if add_offset is not None:
        offset = (np.array(offset) + np.array(add_offset)).tolist()

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

    # Get parameters from run_info
    params = get_params()
    template_fp = params['tm']['template']
    tm_threshold = params['tm']['threshold']
    tm_sigma = params['tm']['sigma']
    tm_add_offset = params['tm']['add_offset']
    has_local = 'local' in params.keys() and params['local'] is not None
    save_bounds = params['auto_pad'] and not has_local
    verbose = params['verbose']

    if verbose:
        print(f'im = {im}')
        print(f'template_fp = {template_fp}')
        print(f'out = {output}')

    template_matching(
        im, template_fp,
        output,
        threshold=tm_threshold,
        sigma=tm_sigma,
        add_offset=tm_add_offset,
        save_bounds=save_bounds,
        verbose=verbose
    )

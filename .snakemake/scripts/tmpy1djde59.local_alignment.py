
######## snakemake preamble start (automatically inserted, do not edit) ########
import sys; sys.path.extend(['/g/icem/hennies/envs/snamst-env/lib/python3.9/site-packages', '/g/icem/hennies/src/github/jhennies/snamst/amst_utils']); import pickle; snakemake = pickle.loads(b'\x80\x04\x958\x06\x00\x00\x00\x00\x00\x00\x8c\x10snakemake.script\x94\x8c\tSnakemake\x94\x93\x94)\x81\x94}\x94(\x8c\x05input\x94\x8c\x0csnakemake.io\x94\x8c\nInputFiles\x94\x93\x94)\x81\x94\x8c\x9a/g/icem/hennies/datasets/steyer_Instruments_Crossbeam_Volume-imaging/2022-02-16_PF_as43a_Acantharia/Original/Images_no_defects/slice_00352_z=10.5300um.tif\x94a}\x94(\x8c\x06_names\x94}\x94\x8c\x12_allowed_overrides\x94]\x94(\x8c\x05index\x94\x8c\x04sort\x94eh\x10\x8c\tfunctools\x94\x8c\x07partial\x94\x93\x94h\x06\x8c\x19Namedlist._used_attribute\x94\x93\x94\x85\x94R\x94(h\x16)}\x94\x8c\x05_name\x94h\x10sNt\x94bh\x11h\x14h\x16\x85\x94R\x94(h\x16)}\x94h\x1ah\x11sNt\x94bub\x8c\x06output\x94h\x06\x8c\x0bOutputFiles\x94\x93\x94)\x81\x94\x8c`/scratch/hennies/tmp/snamst_test2/pre_align_cache/offsets_local/slice_00352_z=10.5300um.tif.json\x94a}\x94(h\x0c}\x94h\x0e]\x94(h\x10h\x11eh\x10h\x14h\x16\x85\x94R\x94(h\x16)}\x94h\x1ah\x10sNt\x94bh\x11h\x14h\x16\x85\x94R\x94(h\x16)}\x94h\x1ah\x11sNt\x94bub\x8c\x06params\x94h\x06\x8c\x06Params\x94\x93\x94)\x81\x94(\x8c\x9e/g/icem/hennies/datasets/steyer_Instruments_Crossbeam_Volume-imaging/2022-02-16_PF_as43a_Acantharia/Original/Images_no_defects/slice_00351_z\\{eq}10.5000um.tif\x94\x8c\x03gpu\x94\x8c\x0c--gres=gpu:1\x94e}\x94(h\x0c}\x94(\x8c\x06ref_im\x94K\x00N\x86\x94\x8c\x01p\x94K\x01N\x86\x94\x8c\x04gres\x94K\x02N\x86\x94uh\x0e]\x94(h\x10h\x11eh\x10h\x14h\x16\x85\x94R\x94(h\x16)}\x94h\x1ah\x10sNt\x94bh\x11h\x14h\x16\x85\x94R\x94(h\x16)}\x94h\x1ah\x11sNt\x94bh9h4h;h5h=h6ub\x8c\twildcards\x94h\x06\x8c\tWildcards\x94\x93\x94)\x81\x94\x8c\x1bslice_00352_z=10.5300um.tif\x94a}\x94(h\x0c}\x94\x8c\x04name\x94K\x00N\x86\x94sh\x0e]\x94(h\x10h\x11eh\x10h\x14h\x16\x85\x94R\x94(h\x16)}\x94h\x1ah\x10sNt\x94bh\x11h\x14h\x16\x85\x94R\x94(h\x16)}\x94h\x1ah\x11sNt\x94b\x8c\x04name\x94hLub\x8c\x07threads\x94K\x01\x8c\tresources\x94h\x06\x8c\tResources\x94\x93\x94)\x81\x94(K\x01K\x01M\x00\x08M\xe8\x03\x8c\x16/scratch/jobs/40851464\x94K\x01K\x01K\ne}\x94(h\x0c}\x94(\x8c\x06_cores\x94K\x00N\x86\x94\x8c\x06_nodes\x94K\x01N\x86\x94\x8c\x06mem_mb\x94K\x02N\x86\x94\x8c\x07disk_mb\x94K\x03N\x86\x94\x8c\x06tmpdir\x94K\x04N\x86\x94h5K\x05N\x86\x94\x8c\x04cpus\x94K\x06N\x86\x94\x8c\x08time_min\x94K\x07N\x86\x94uh\x0e]\x94(h\x10h\x11eh\x10h\x14h\x16\x85\x94R\x94(h\x16)}\x94h\x1ah\x10sNt\x94bh\x11h\x14h\x16\x85\x94R\x94(h\x16)}\x94h\x1ah\x11sNt\x94bhcK\x01heK\x01hgM\x00\x08hiM\xe8\x03hkh`h5K\x01hnK\x01hpK\nub\x8c\x03log\x94h\x06\x8c\x03Log\x94\x93\x94)\x81\x94}\x94(h\x0c}\x94h\x0e]\x94(h\x10h\x11eh\x10h\x14h\x16\x85\x94R\x94(h\x16)}\x94h\x1ah\x10sNt\x94bh\x11h\x14h\x16\x85\x94R\x94(h\x16)}\x94h\x1ah\x11sNt\x94bub\x8c\x06config\x94}\x94\x8c\x04rule\x94\x8c\x0flocal_alignment\x94\x8c\x0fbench_iteration\x94N\x8c\tscriptdir\x94\x8c5/g/icem/hennies/src/github/jhennies/snamst/amst_utils\x94ub.'); from snakemake.logging import logger; logger.printshellcmds = False; __real_file__ = __file__; __file__ = '/g/icem/hennies/src/github/jhennies/snamst/amst_utils/local_alignment.py';
######## snakemake preamble end #########

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
        auto_mask=None,
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
                auto_mask=auto_mask,
                verbose=verbose
            )
            if verbose:
                print(f'out = {out}')
            offset, bounds = out if save_bounds else (out, None)

        elif align_method == 'xcorr':
            raise NotImplementedError(f'Cross-correlation is not yet implemented!')
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
    local_auto_mask = params['local']['auto_mask']
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
        auto_mask=local_auto_mask,
        verbose=verbose
    )

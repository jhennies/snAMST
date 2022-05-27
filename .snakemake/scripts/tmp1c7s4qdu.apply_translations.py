
######## snakemake preamble start (automatically inserted, do not edit) ########
import sys; sys.path.extend(['/g/icem/hennies/envs/snamst-env/lib/python3.9/site-packages', '/g/icem/hennies/src/github/jhennies/snamst/amst_utils']); import pickle; snakemake = pickle.loads(b'\x80\x04\x95\x9b\x05\x00\x00\x00\x00\x00\x00\x8c\x10snakemake.script\x94\x8c\tSnakemake\x94\x93\x94)\x81\x94}\x94(\x8c\x05input\x94\x8c\x0csnakemake.io\x94\x8c\nInputFiles\x94\x93\x94)\x81\x94(\x8c\x99/g/icem/hennies/datasets/steyer_Instruments_Crossbeam_Volume-imaging/2022-02-16_PF_as43a_Acantharia/Original/Images_no_defects/slice_00206_z=6.1500um.tif\x94\x8cD/scratch/hennies/tmp/snamst_test2/pre_align_cache/final_offsets.json\x94e}\x94(\x8c\x06_names\x94}\x94\x8c\x12_allowed_overrides\x94]\x94(\x8c\x05index\x94\x8c\x04sort\x94eh\x11\x8c\tfunctools\x94\x8c\x07partial\x94\x93\x94h\x06\x8c\x19Namedlist._used_attribute\x94\x93\x94\x85\x94R\x94(h\x17)}\x94\x8c\x05_name\x94h\x11sNt\x94bh\x12h\x15h\x17\x85\x94R\x94(h\x17)}\x94h\x1bh\x12sNt\x94bub\x8c\x06output\x94h\x06\x8c\x0bOutputFiles\x94\x93\x94)\x81\x94\x8cF/scratch/hennies/tmp/snamst_test2/pre_align/slice_00206_z=6.1500um.tif\x94a}\x94(h\r}\x94h\x0f]\x94(h\x11h\x12eh\x11h\x15h\x17\x85\x94R\x94(h\x17)}\x94h\x1bh\x11sNt\x94bh\x12h\x15h\x17\x85\x94R\x94(h\x17)}\x94h\x1bh\x12sNt\x94bub\x8c\x06params\x94h\x06\x8c\x06Params\x94\x93\x94)\x81\x94(\x8c\x03htc\x94\x8c\x00\x94e}\x94(h\r}\x94(\x8c\x01p\x94K\x00N\x86\x94\x8c\x04gres\x94K\x01N\x86\x94uh\x0f]\x94(h\x11h\x12eh\x11h\x15h\x17\x85\x94R\x94(h\x17)}\x94h\x1bh\x11sNt\x94bh\x12h\x15h\x17\x85\x94R\x94(h\x17)}\x94h\x1bh\x12sNt\x94bh9h5h;h6ub\x8c\twildcards\x94h\x06\x8c\tWildcards\x94\x93\x94)\x81\x94\x8c\x1aslice_00206_z=6.1500um.tif\x94a}\x94(h\r}\x94\x8c\x04name\x94K\x00N\x86\x94sh\x0f]\x94(h\x11h\x12eh\x11h\x15h\x17\x85\x94R\x94(h\x17)}\x94h\x1bh\x11sNt\x94bh\x12h\x15h\x17\x85\x94R\x94(h\x17)}\x94h\x1bh\x12sNt\x94b\x8c\x04name\x94hJub\x8c\x07threads\x94K\x01\x8c\tresources\x94h\x06\x8c\tResources\x94\x93\x94)\x81\x94(K\x01K\x01M\x00\x02M\xe8\x03\x8c\x16/scratch/jobs/40852046\x94K\x01K\ne}\x94(h\r}\x94(\x8c\x06_cores\x94K\x00N\x86\x94\x8c\x06_nodes\x94K\x01N\x86\x94\x8c\x06mem_mb\x94K\x02N\x86\x94\x8c\x07disk_mb\x94K\x03N\x86\x94\x8c\x06tmpdir\x94K\x04N\x86\x94\x8c\x04cpus\x94K\x05N\x86\x94\x8c\x08time_min\x94K\x06N\x86\x94uh\x0f]\x94(h\x11h\x12eh\x11h\x15h\x17\x85\x94R\x94(h\x17)}\x94h\x1bh\x11sNt\x94bh\x12h\x15h\x17\x85\x94R\x94(h\x17)}\x94h\x1bh\x12sNt\x94bhaK\x01hcK\x01heM\x00\x02hgM\xe8\x03hih^hkK\x01hmK\nub\x8c\x03log\x94h\x06\x8c\x03Log\x94\x93\x94)\x81\x94}\x94(h\r}\x94h\x0f]\x94(h\x11h\x12eh\x11h\x15h\x17\x85\x94R\x94(h\x17)}\x94h\x1bh\x11sNt\x94bh\x12h\x15h\x17\x85\x94R\x94(h\x17)}\x94h\x1bh\x12sNt\x94bub\x8c\x06config\x94}\x94\x8c\x04rule\x94\x8c\x12apply_translations\x94\x8c\x0fbench_iteration\x94N\x8c\tscriptdir\x94\x8c5/g/icem/hennies/src/github/jhennies/snamst/amst_utils\x94ub.'); from snakemake.logging import logger; logger.printshellcmds = False; __real_file__ = __file__; __file__ = '/g/icem/hennies/src/github/jhennies/snamst/amst_utils/apply_translations.py';
######## snakemake preamble end #########

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
        bounds = None if data['bounds'] is None else data['bounds'][idx]
        shape = None if data['shape'] is None else data['shape']
        del data

    if verbose:
        print(f'im_list = {im_list}')
        print(f'idx = {idx}')
        print(f'offsets = {offset}')
        print(f'bounds = {bounds}')
        print(f'shape = {shape}')

    if bounds is not None:
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



import os
import snakemake
from glob import glob
from amst_utils.common.settings import set_run_info, set_params, get_params_fp, get_params, get_workdir


def run_amst(
        source_folder,
        target_folder,
        pre_align_folder=None,
        use_coarse_align=False,
        median_with_one_rule=False,
        amst_params=None,
        snake_kwargs=None,
        verbose=False
):

    # Get the main parameters if a parameter file is supplied
    if amst_params is not None:
        params = get_params(fp=amst_params)
        source_folder = params['source_folder']
        target_folder = params['target_folder']
        pre_align_folder = params['pre_align_folder']
        verbose=params['verbose']

    # Parse the source folder
    im_list = sorted(glob(os.path.join(source_folder, '*.tif')))
    im_names = [os.path.split(fp)[1] for fp in im_list]

    # Make run_info
    set_run_info(
        dict(
            source_folder=source_folder,
            target_folder=target_folder,
            pre_align_folder=pre_align_folder,
            params_fp=amst_params if amst_params is not None else get_params_fp(folder=os.path.join(target_folder, 'amst_cache')),
            im_list=im_list,
            im_names=im_names,
            verbose=verbose
        )
    )

    # Set up the amst parameters if no file is given
    if amst_params is None:
        set_params(
            dict(
                source_folder=source_folder,
                target_folder=target_folder,
                pre_align_folder=pre_align_folder,
                use_coarse_align=use_coarse_align,
                median_with_one_rule=median_with_one_rule,
                verbose=verbose
            )
        )

    # Call the snakemake workflow
    snakemake.snakemake(
        os.path.join('snake_utils', 'amst.smk'), **snake_kwargs
    )


if __name__ == '__main__':

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs pre-alignment workflow',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-sf', '--source_folder', type=str, default=None,
                        help='The input dataset')
    parser.add_argument('-tf', '--target_folder', type=str, default=None,
                        help='The output folder')
    parser.add_argument('-pf', '--pre_align_folder', type=str, default=None,
                        help='Pre-alignment folder '
                             'Can be ommited if the target folder is the target folder of the pre-align workflow')
    parser.add_argument('-uca', '--use_coarse_align', action='store_true',
                        help='Use coarse alignment of the raw to the median smoothed template by SIFT.')
    parser.add_argument('-mor', '--median_with_one_rule', action='store_true',
                        help='Do the median filtering as one parallelized task, which reduces IO load. '
                             'Note that, if set, the z-median filtering has to be fully completed before any other task'
                             'can be run')
    parser.add_argument('-ap', '--amst_params', type=str, default=None,
                        help='Parameter file defining all parameters. Script parameter definitions are ignored if set')
    parser.add_argument('-n', '--dryrun', action='store_true',
                        help='Trigger dry run')
    parser.add_argument('-c', '--cores', type=int, default=os.cpu_count(),
                        help='Number of cores to use')
    parser.add_argument('-g', '--gpu', type=int, default=1,
                        help='Number of available GPUs')
    parser.add_argument('--unlock', action='store_true',
                        help='Unlock snakemake directory')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    source_folder = args.source_folder
    target_folder = args.target_folder
    pre_align_folder = args.pre_align_folder
    use_coarse_align = args.use_coarse_align
    median_with_one_rule = args.median_with_one_rule
    amst_params = args.amst_params
    dryrun = args.dryrun
    cores = args.cores
    gpu = args.gpu
    unlock = args.unlock
    verbose = args.verbose

    run_amst(
        source_folder,
        target_folder,
        pre_align_folder=pre_align_folder,
        use_coarse_align=use_coarse_align,
        median_with_one_rule=median_with_one_rule,
        amst_params=amst_params,
        snake_kwargs=dict(
            resources={
                'gpu': gpu
            },
            dryrun=dryrun,
            cores=cores,
            unlock=unlock
        ),
        verbose=verbose
    )

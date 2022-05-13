
import os
import snakemake
from glob import glob
from amst_utils.common.settings import set_run_info, set_params, get_params_fp, get_params


def run_pre_align(
        source_folder,
        target_folder,
        local_align_method=None,
        combine_median=8,
        combine_sigma=8.,
        auto_pad=False,
        align_params=None,
        snake_kwargs=None,
        verbose=False
):

    # Get the main parameters if a parameter file is supplied
    if align_params is not None:
        params = get_params(fp=align_params)
        source_folder = params['source_folder']
        target_folder = params['target_folder']
        verbose = params['verbose']

    # Parse the source folder
    im_list = sorted(glob(os.path.join(source_folder, '*.tif')))
    im_names = [os.path.split(fp)[1] for fp in im_list]

    # Make run_info
    set_run_info(
        dict(
            source_folder=source_folder,
            target_folder=target_folder,
            params_fp=align_params if align_params is not None else get_params_fp(folder=os.path.join(target_folder, 'cache')),
            im_list=im_list,
            im_names=im_names,
            verbose=verbose
        )
    )

    # Set up the alignment parameters if no file is given
    if align_params is None:
        set_params(
            dict(
                source_folder=source_folder,
                target_folder=target_folder,
                local=None if local_align_method is None else dict(
                    align_method=local_align_method
                ),
                combine_median=combine_median,
                combine_sigma=combine_sigma,
                auto_pad=auto_pad,
                verbose=verbose
            )
        )

    # Call the snakemake workflow
    snakemake.snakemake(
        os.path.join('snake_utils', 'pre_align.smk'), **snake_kwargs
    )


if __name__ == '__main__':

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Runs pre-alignment workflow',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-sf', '--source_folder', type=str,
                        help='The input dataset')
    parser.add_argument('-tf', '--target_folder', type=str,
                        help='The output folder')
    parser.add_argument('-lam', '--local_align_method', type=str, default=None,
                        help='Method for local alignment: "sift", "xcorr", or "none"')
    parser.add_argument('-cm', '--combine_median', type=int, default=8,
                        help='Median smoothing of offsets when combining local and TM')
    parser.add_argument('-cs', '--combine_sigma', type=float, default=8.,
                        help='Gaussian smoothing of offsets when combining local and TM')
    parser.add_argument('-apd', '--auto_pad', action='store_true',
                        help='Automatically adjust canvas to match the final slice positions')
    parser.add_argument('-ap', '--align_params', type=str, default=None,
                        help='Parameter file for the alignment methods.')
    parser.add_argument('-n', '--dryrun', action='store_true',
                        help='Trigger dry run')
    parser.add_argument('-c', '--cores', type=int, default=os.cpu_count(),
                        help='Number of cores to use')
    parser.add_argument('-g', '--gpu', type=int, default=1,
                        help='Number of available GPUs')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    source_folder = args.source_folder
    target_folder = args.target_folder
    local_align_method = args.local_align_method
    combine_median = args.combine_median
    combine_sigma = args.combine_sigma
    auto_pad = args.auto_pad
    align_params = args.align_params
    dryrun = args.dryrun
    cores = args.cores
    gpu = args.gpu
    verbose = args.verbose

    if local_align_method == 'none':
        local_align_method = None

    run_pre_align(
        source_folder,
        target_folder,
        local_align_method=local_align_method,
        combine_median=combine_median,
        combine_sigma=combine_sigma,
        auto_pad=auto_pad,
        align_params=align_params,
        snake_kwargs=dict(
            resources={
                'gpu': gpu
            },
            dryrun=dryrun,
            cores=cores
        ),
        verbose=verbose
    )


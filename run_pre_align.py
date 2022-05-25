
import os
import snakemake
from glob import glob
from amst_utils.common.settings import set_run_info, set_params, get_params_fp, get_params


def run_pre_align(
        source_folder,
        target_folder,
        local_align_method=None,
        local_mask_range=None,
        local_sigma=1.6,
        local_norm_quantiles=(0.1, 0.9),
        local_device_type='GPU',
        local_auto_mask=None,
        template=None,
        tm_threshold=(0, 0),
        tm_sigma=0.,
        tm_add_offset=None,
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
    im_list = sorted(glob(os.path.join(source_folder, '*.tif')))[100:124]
    im_names = [os.path.split(fp)[1] for fp in im_list]

    # Make run_info
    set_run_info(
        dict(
            source_folder=source_folder,
            target_folder=target_folder,
            params_fp=align_params if align_params is not None else get_params_fp(folder=os.path.join(target_folder, 'pre_align_cache')),
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
                    align_method=local_align_method,
                    mask_range=local_mask_range,
                    sigma=local_sigma,
                    norm_quantiles=local_norm_quantiles,
                    device_type=local_device_type,
                    auto_mask=local_auto_mask
                ),
                tm=None if template is None else dict(
                    template=template,
                    threshold=tm_threshold,
                    sigma=tm_sigma,
                    add_offset=tm_add_offset
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
    parser.add_argument('-lmr', '--local_mask_range', type=float, nargs=2, default=None,
                        metavar=('lower', 'upper'),
                        help='Similar to threshold, except values above the upper threshold are set to zero')
    parser.add_argument('-lsg', '--local_sigma', type=float, default=1.6,
                        help='Smooths the data before local alignment')
    parser.add_argument('-lnq', '--local_norm_quantiles', type=float, nargs=2, default=(0.1, 0.9),
                        help='For SIFT: Histogram quantiles for normalization of the data. Default=(0.1, 0.9)')
    parser.add_argument('-ldt', '--local_device_type', type=str, default='GPU',
                        help='For SIFT: either GPU or CPU')
    parser.add_argument('-lau' '--local_auto_mask', type=int, default=None,
                        help='Generates a mask by eroding the non-zero data by the specified amount')
    parser.add_argument('-tm', '--template', type=str,
                        help='Location of template tiff image. Enables template matching step if set')
    parser.add_argument('-tmt', '--tm_threshold', type=float, nargs=2, default=[0, 0],
                        metavar=('lower', 'upper'),
                        help='Lower and upper thresholds applied before template matching')
    parser.add_argument('-tms', '--tm_sigma', type=float, default=0.,
                        help='Smooths the data before template matching alignment')
    parser.add_argument('-tmo', '--tm_add_offset', type=int, nargs=2, default=None,
                        metavar=('X', 'Y'),
                        help='Add an offset to the final alignment')
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
    parser.add_argument('--unlock', action='store_true',
                        help='Unlock snakemake directory')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    source_folder = args.source_folder
    target_folder = args.target_folder
    local_align_method = args.local_align_method
    local_mask_range = args.local_mask_range
    local_sigma = args.local_sigma
    local_norm_quantiles = args.local_norm_quantiles
    local_device_type = args.local_device_type
    local_auto_mask = args.local_auto_mask
    template = args.template
    tm_threshold = args.tm_threshold
    tm_sigma = args.tm_sigma
    tm_add_offset = args.tm_add_offset
    combine_median = args.combine_median
    combine_sigma = args.combine_sigma
    auto_pad = args.auto_pad
    align_params = args.align_params
    dryrun = args.dryrun
    cores = args.cores
    gpu = args.gpu
    unlock = args.unlock
    verbose = args.verbose

    if local_align_method == 'none':
        local_align_method = None

    run_pre_align(
        source_folder,
        target_folder,
        local_align_method=local_align_method,
        local_mask_range=local_mask_range,
        local_sigma=local_sigma,
        local_norm_quantiles=local_norm_quantiles,
        local_device_type=local_device_type,
        local_auto_mask=local_auto_mask,
        template=template,
        tm_threshold=tm_threshold,
        tm_sigma=tm_sigma,
        tm_add_offset=tm_add_offset,
        combine_median=combine_median,
        combine_sigma=combine_sigma,
        auto_pad=auto_pad,
        align_params=align_params,
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


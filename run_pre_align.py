
import os
import snakemake
from glob import glob
from amst_utils.common.settings import set_run_info, set_params, get_params_fp, get_params, remove_run_info


def run_pre_align(
        source_folder,
        target_folder,
        local_align_method=None,
        local_mask_range=None,
        local_thresh=None,
        local_sigma=1.6,
        local_norm_quantiles=(0.1, 0.9),
        local_device_type='GPU',
        local_auto_mask=None,
        local_mask_folder=None,
        local_max_offset=None,
        local_xy_range=None,
        local_invert_nonzero=False,
        local_downsample=1,
        local_bias=(0., 0.),
        local_big_jump_prefix=False,
        template=None,
        tm_threshold=(0, 0),
        tm_sigma=0.,
        tm_add_offset=None,
        combine_median=8,
        combine_sigma=8.,
        auto_pad=False,
        subtract_running_average=False,
        align_params=None,
        batch_size=1,
        snake_kwargs=None,
        cluster=None,
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
    mask_list = sorted(glob(os.path.join(local_mask_folder, '*.tif'))) if local_mask_folder is not None else None

    # Make run_info
    set_run_info(
        dict(
            source_folder=source_folder,
            target_folder=target_folder,
            params_fp=align_params if align_params is not None else get_params_fp(folder=os.path.join(target_folder, 'pre_align_cache')),
            im_list=im_list,
            im_names=im_names,
            mask_list=mask_list,
            batch_size=batch_size,
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
                    thresh=local_thresh,
                    sigma=local_sigma,
                    norm_quantiles=local_norm_quantiles,
                    device_type=local_device_type,
                    auto_mask=local_auto_mask,
                    mask_folder=local_mask_folder,
                    max_offset=local_max_offset,
                    xy_range=local_xy_range,
                    invert_nonzero=local_invert_nonzero,
                    downsample=local_downsample,
                    bias=local_bias,
                    big_jump_prefix=local_big_jump_prefix
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
                subtract_running_average=subtract_running_average,
                verbose=verbose
            )
        )

    # Add the cluster profile
    log_dir = os.path.join(target_folder, 'log')
    if verbose:
        print(f'log_dir = {log_dir}')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if cluster is not None:
        if cluster == 'slurm':
            snake_kwargs['cluster'] = (
                "sbatch "
                "-p {params.p} {params.gres} "
                "-t {resources.time_min} "
                "--mem={resources.mem_mb} "
                "-c {resources.cpus} "
                "-o " + log_dir + "/{rule}_{wildcards}d.%N.%j.out "
                "-e " + log_dir + "/{rule}_{wildcards}d.%N.%j.err "
            )
            # snake_kwargs['cluster_config'] = 'cluster/slurm/config.yaml'
            snake_kwargs['nodes'] = cores
            snake_kwargs['restart_times'] = 0
            snake_kwargs['latency_wait'] = 3
            snake_kwargs['max_jobs_per_second'] = 8
        else:
            raise RuntimeError(f'Not supporting cluster = {cluster}')

    # Call the snakemake workflow
    script_path = os.path.realpath(os.path.dirname(__file__))
    if batch_size == 1:
        snakemake.snakemake(
            os.path.join(script_path, 'snake_utils', 'pre_align.smk'), **snake_kwargs
        )
    else:
        raise NotImplementedError()
        snakemake.snakemake(
            os.path.join('snake_utils', 'pre_align_batch.smk'), **snake_kwargs
        )

    remove_run_info()


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
                        help='Similar to local_thresh, except values above the upper threshold are set to zero')
    parser.add_argument('-lth', '--local_thresh', type=float, nargs=2, default=None,
                        help='Threshold to clip the data (upper and lower bound)')
    parser.add_argument('-lsg', '--local_sigma', type=float, default=1.6,
                        help='Smooths the data before local alignment')
    parser.add_argument('-lnq', '--local_norm_quantiles', type=float, nargs=2, default=(0.1, 0.9),
                        help='For SIFT: Histogram quantiles for normalization of the data. Default=(0.1, 0.9)')
    parser.add_argument('-ldt', '--local_device_type', type=str, default='GPU',
                        help='For SIFT: either GPU or CPU')
    parser.add_argument('-lau', '--local_auto_mask', type=int, default=None,
                        help='Generates a mask by eroding the non-zero data by the specified amount')
    parser.add_argument('-lmk', '--local_mask_folder', type=str, default=None,
                        help='Supply a mask dataset with same dimensions as the source dataset')
    parser.add_argument('-lmo', '--local_max_offset', type=int, nargs=2, default=None,
                        help='Maximum offset allowed to avoid big jumps when the alignment failed')
    parser.add_argument('-lxy', '--local_xy_range', type=int, nargs=4, default=None,
                        metavar=('X', 'Y', 'W', 'H'),
                        help='Crop xy-range for computation: (x, y, width, height)')
    parser.add_argument('-liv', '--local_invert_nonzero', action='store_true',
                        help='The SIFT performs a lot better if the features of interest are bright')
    parser.add_argument('-lds', '--local_downsample', type=int, default=1,
                        help='Downsample the data before computing local alignment; only for SIFT; '
                             'default=1 (no downsampling)')
    parser.add_argument('-lbs', '--local_bias', type=float, nargs=2, default=(0., 0.),
                        metavar=('X', 'Y'),
                        help='Bias for local alignment; only for SIFT; default=(0., 0.)')
    parser.add_argument('-lpf', '--local_big_jump_prefix', action='store_true',
                        help='Activate a cross-correlation-based pre-fixing if the non-zero regions of adjacent slices'
                             'overlap by less than 50 % (IoU)')
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
    parser.add_argument('-sra', '--subtract_running_average', action='store_true',
                        help='Activates subtraction of running average to avoid banana-effect')
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
    parser.add_argument('--cluster', type=str, default=None,
                        help='Enables cluster support. '
                             'Currently only None for local computation and "slurm" is supported')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    source_folder = args.source_folder
    target_folder = args.target_folder
    local_align_method = args.local_align_method
    local_mask_range = args.local_mask_range
    local_thresh = args.local_thresh
    local_sigma = args.local_sigma
    local_norm_quantiles = args.local_norm_quantiles
    local_device_type = args.local_device_type
    local_auto_mask = args.local_auto_mask
    local_mask_folder = args.local_mask_folder
    local_max_offset = args.local_max_offset
    local_xy_range = args.local_xy_range
    local_invert_nonzero = args.local_invert_nonzero
    local_downsample = args.local_downsample
    local_bias = args.local_bias
    local_big_jump_prefix = args.local_big_jump_prefix
    template = args.template
    tm_threshold = args.tm_threshold
    tm_sigma = args.tm_sigma
    tm_add_offset = args.tm_add_offset
    combine_median = args.combine_median
    combine_sigma = args.combine_sigma
    auto_pad = args.auto_pad
    subtract_running_average = args.subtract_running_average
    align_params = args.align_params
    dryrun = args.dryrun
    cores = args.cores
    gpu = args.gpu
    unlock = args.unlock
    cluster = args.cluster
    verbose = args.verbose

    if local_align_method == 'none':
        local_align_method = None
    if cluster == 'none':
        cluster = None
    if local_norm_quantiles is not None and local_norm_quantiles[0] == 0 and local_norm_quantiles[1] == 0:
        local_norm_quantiles = None
    if local_mask_range is not None and local_mask_range[0] == 0 and local_mask_range[1] == 0:
        local_mask_range = None
    if local_thresh is not None and local_thresh[0] == 0 and local_thresh[1] == 0:
        local_thresh = None
    if local_xy_range is not None and local_xy_range[0] == 0 and local_xy_range[1] == 0:
        local_xy_range = None
    if cores == 0:
        cores = os.cpu_count()
    if local_mask_folder == 'none':
        local_mask_folder = None

    run_pre_align(
        source_folder,
        target_folder,
        local_align_method=local_align_method,
        local_mask_range=local_mask_range,
        local_thresh=local_thresh,
        local_sigma=local_sigma,
        local_norm_quantiles=local_norm_quantiles,
        local_device_type=local_device_type,
        local_auto_mask=local_auto_mask,
        local_mask_folder=local_mask_folder,
        local_max_offset=local_max_offset,
        local_xy_range=local_xy_range,
        local_invert_nonzero=local_invert_nonzero,
        local_downsample=local_downsample,
        local_bias=local_bias,
        local_big_jump_prefix=local_big_jump_prefix,
        template=template,
        tm_threshold=tm_threshold,
        tm_sigma=tm_sigma,
        tm_add_offset=tm_add_offset,
        combine_median=combine_median,
        combine_sigma=combine_sigma,
        auto_pad=auto_pad,
        subtract_running_average=subtract_running_average,
        align_params=align_params,
        snake_kwargs=dict(
            resources={
                'gpu': gpu
            },
            dryrun=dryrun,
            cores=cores,
            unlock=unlock
        ),
        cluster=cluster,
        verbose=verbose
    )


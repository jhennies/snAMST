
import os
import snakemake
import json
from snake_utils.settings import get_run_info_fp


def run_pre_align(
        source_folder,
        target_folder,
        snake_kwargs=None,
        verbose=False
):

    # Set the run parameters
    run_info = dict(
        source_folder=source_folder,
        target_folder=target_folder,
        verbose=verbose
    )
    with open(get_run_info_fp(), mode='w') as f:
        json.dump(run_info, f, indent=2)

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
    parser.add_argument('source_folder', type=str,
                        help='The input dataset')
    parser.add_argument('target_folder', type=str,
                        help='The output folder')
    parser.add_argument('-n', '--dryrun', action='store_true',
                        help='Trigger dry run')
    parser.add_argument('-c', '--cores', type=int, default=os.cpu_count(),
                        help='Number of cores to use')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    source_folder = args.source_folder
    target_folder = args.target_folder
    dryrun = args.dryrun
    cores = args.cores
    verbose = args.verbose

    run_pre_align(
        source_folder,
        target_folder,
        snake_kwargs=dict(
            dryrun=dryrun,
            cores=cores
        ),
        verbose=verbose
    )


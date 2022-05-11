
import os


def get_run_info_fp():
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    return os.path.join('tmp', 'run_info.json')

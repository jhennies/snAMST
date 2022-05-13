
import os
import json


def get_run_info_fp():
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    return os.path.join('tmp', 'run_info.json')


def get_run_info():
    with open(get_run_info_fp(), mode='r') as f:
        run_info = json.load(f)
    return run_info


def set_run_info(run_info):
    with open(get_run_info_fp(), mode='w') as f:
        json.dump(run_info, f, indent=2)


def get_params_fp(folder=None):
    if folder is not None:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, 'params.json')
    else:
        return get_run_info()['params_fp']


def set_params(params):
    fp = get_params_fp()
    with open(fp, mode='w') as f:
        json.dump(params, f, indent=2)


def get_params(fp=None):
    fp = fp if fp is not None else get_params_fp()
    with open(fp, mode='r') as f:
        params = json.load(f)
    return params

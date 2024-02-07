
import os
import json
import getpass


def get_run_info_fp():
    path = os.path.join(
        '/home', getpass.getuser(), '.tmp'
    )
    if not os.path.exists(path):
        os.mkdir(path)
    return os.path.join(path, 'run_info.json')


def get_run_info():
    with open(get_run_info_fp(), mode='r') as f:
        run_info = json.load(f)
    return run_info


def set_run_info(run_info):
    fp = get_run_info_fp()
    if os.path.exists(fp):
        raise RuntimeError(f'Another instance of the AMST pre-alignment is already running! '
                           f'If this is not the case delete {fp} and start again.')
    with open(fp, mode='w') as f:
        json.dump(run_info, f, indent=2)


def remove_run_info():
    fp = get_run_info_fp()
    os.remove(fp)


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

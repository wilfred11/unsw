import shutil
import numpy as np
import os


def empty_string_to_nan(x):
    if not x.isspace():
        return x
    return np.nan


def unsw_data(test):
    if not test:
        return ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv', 'UNSW-NB15_2.csv']
    else:
        return ['UNSW-NB15_1.csv']


def non_numeric_features():
    return ['proto', 'service', 'state', 'is_ftp_login', 'is_sm_ips_ports', 'attack_cat', 'Label',
            ]


def numeric_features(raw_data):
    return [item for item in raw_data.columns if item not in non_numeric_features()]


def features_to_be_denominalized():
    return ['proto', 'service', 'state']


def non_numeric_features_to_keep():
    return ['is_ftp_login', 'is_sm_ips_ports', 'attack_cat', 'label']


def irrelevant_features():
    return ['Ltime', 'Stime', 'srcip', 'sport', 'dstip', 'dsport']


def cleanup_project_dirs():
    if os.path.exists(figures_dir()):
        shutil.rmtree(figures_dir())
    if os.path.exists(data_dir()):
        shutil.rmtree(data_dir())

    os.makedirs(figures_dir(), exist_ok=True)
    os.makedirs(data_dir(), exist_ok=True)
    os.makedirs(test_classifiers_dir(), exist_ok=True)


def external_data_dir():
    return '../unsw_external_data'


def data_dir():
    return external_data_dir() + '/generated_data'


def figures_dir():
    return data_dir() + '/figs'


def test_classifiers_dir():
    return data_dir() + '/test_classifiers'


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_large_data_to_file(file_path, buffer_size, data):
    with open(file_path, 'w', buffering=buffer_size) as file:
        file.write(data)

import shutil
import numpy as np
import os


def empty_string_to_nan(x):
    if not x.isspace() and not x == '':
        return np.int64(x)
    elif x.isspace() or x == '':
        return 0
    else:
        return np.nan


def unsw_data(test):
    if not test:
        return ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv', 'UNSW-NB15_4.csv']
    else:
        return ['UNSW-NB15_1.csv']


def non_numeric_features():
    return ['proto', 'service', 'state', 'is_ftp_login', 'is_sm_ips_ports', 'attack_cat', 'Label']


def numeric_features(raw_data):
    return [item for item in raw_data.columns if item not in (non_numeric_features() + ip_port_features())]


def features_to_be_denominalized():
    return ['proto', 'service', 'state']


def non_numeric_features_to_keep(include_targets=True):
    if include_targets == True:
        return ['is_ftp_login', 'is_sm_ips_ports', 'attack_cat', 'Label']
    else:
        return ['is_ftp_login', 'is_sm_ips_ports']


def irrelevant_features():
    return ['Ltime', 'Stime', 'srcip', 'sport', 'dstip', 'dsport']


def ip_port_features():
    return ['srcip', 'sport', 'dstip', 'dsport']


def denominalized_and_boolean_features(raw_data, include_targets=False):
    l = raw_data.columns[raw_data.columns.str.startswith('service')].to_list() + raw_data.columns[
        raw_data.columns.str.startswith('proto')].to_list() + raw_data.columns[
            raw_data.columns.str.startswith('state')].to_list() + non_numeric_features_to_keep(include_targets)
    return l


def keep_numeric_columns(raw_data, exclude_targets=True):
    l = denominalized_and_boolean_features(raw_data, exclude_targets)
    cols = [col for col in raw_data.columns if col not in l]
    raw_data_numeric = raw_data[cols]
    return raw_data_numeric


def cleanup_project_dirs():
    if os.path.exists(figures_dir()):
        shutil.rmtree(figures_dir())
    if os.path.exists(data_dir()):
        shutil.rmtree(data_dir())

    os.makedirs(figures_dir(), exist_ok=True)
    os.makedirs(data_dir(), exist_ok=True)
    os.makedirs(test_classifiers_dir(), exist_ok=True)
    os.makedirs(test_classifiers_figs_dir(), exist_ok=True)
    os.makedirs(read_prepare_dir(), exist_ok=True)
    os.makedirs(read_prepare_figs_dir(), exist_ok=True)
    os.makedirs(feature_reduction_dir(), exist_ok=True)
    os.makedirs(feature_reduction_dir() + '/figs', exist_ok=True)


def external_data_dir():
    return '../unsw_external_data'


def data_dir():
    return external_data_dir() + '/generated_data'


def figures_dir():
    return data_dir() + '/figs'


def test_classifiers_dir():
    return data_dir() + '/test_classifiers'


def read_prepare_dir():
    return data_dir() + '/read_prepare'


def read_prepare_figs_dir():
    return read_prepare_dir() + '/figs'


def test_classifiers_figs_dir():
    return test_classifiers_dir() + '/figs'


def feature_reduction_dir():
    return data_dir() + '/feature_reduction'


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_large_data_to_file(file_path, buffer_size, data):
    with open(file_path, 'w', buffering=buffer_size) as file:
        file.write(data)

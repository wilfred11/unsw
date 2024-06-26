import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import empty_string_to_nan, irrelevant_features, external_data_dir, figures_dir, data_dir, \
    read_prepare_dir, dataset_dir, correct_col_names


def clean_data(raw_data):
    print('cleaning data')
    raw_data['attack_cat'] = raw_data['attack_cat'].replace(np.nan, 'Normal')
    raw_data['attack_cat'] = raw_data['attack_cat'].str.strip()
    raw_data['attack_cat'] = raw_data['attack_cat'].replace('Backdoor', 'Backdoors')
    raw_data.is_ftp_login = raw_data.is_ftp_login.astype('bool')
    raw_data.is_sm_ips_ports = raw_data.is_sm_ips_ports.astype('bool')
    raw_data.drop(columns=irrelevant_features(), axis=1, inplace=True)
    return raw_data


def clean_data_1(raw_data):
    print('cleaning prepared data')
    raw_data.is_ftp_login = raw_data.is_ftp_login.astype('bool')
    raw_data.is_sm_ips_ports = raw_data.is_sm_ips_ports.astype('bool')
    #print(raw_data.columns)
    #print(len(raw_data.columns))
    # raw_data = raw_data.drop('rate', axis=1)
    raw_data.drop(columns=['id', 'rate'], axis=1, inplace=True)
    return raw_data


def info(raw_data):
    print("shape:" + str(raw_data.shape))
    print("\n")
    print(str(sys.getsizeof(raw_data) / 1024) + " kb")
    print("\n")
    raw_data.info(verbose=1)
    print("\n")
    rd_attacks = raw_data[raw_data['attack_cat'] != 'Normal']
    raw_data.attack_cat.value_counts(normalize=False).to_csv(read_prepare_dir() + '/' + 'attack_cat_counts.csv')
    with open("attack_cat_counts.txt", "w") as text_file:
        text_file.write(raw_data.attack_cat.value_counts(normalize=False).to_frame().to_latex())
    print("\n")
    plt.figure(figsize=(12, 8))
    sns.barplot(x=rd_attacks.attack_cat.value_counts().index, y=rd_attacks.attack_cat.value_counts()).set(title='')
    plt.savefig(figures_dir() + '/' + 'attack_counts.png')
    print("number of attacks:" + str(len(rd_attacks)))
    print("\n")
    print('number of duplicates')
    print(raw_data.duplicated(subset=None, keep='first').sum())


def read_prepared_data(unsw_prepared_data):
    print("reading train and test data")
    raw_data_list = []

    for filename in unsw_prepared_data:
        raw_data_part = pd.read_csv(dataset_dir() + "/" + filename, sep=",", header=0,
                                    dtype={'sport': 'string', 'dsport': 'string', 'attack_cat': 'string'},
                                    converters={'ct_ftp_cmd': empty_string_to_nan, 'is_ftp_login': empty_string_to_nan,
                                                'ct_flw_http_mthd': empty_string_to_nan},
                                    encoding='utf8')
        raw_data_list.append(raw_data_part)

    raw_data = pd.concat(raw_data_list)
    raw_data.drop_duplicates(inplace=True, subset=None, keep='first')
    clean_data_1(raw_data)
    return raw_data


def read_data(unsw_data, test):
    print("reading data")
    raw_data_list = []
    column_names = pd.read_csv(dataset_dir() + "/" + 'UNSW-NB15_features.csv', encoding='ISO-8859-1')
    column_names['Name'] = column_names['Name'].str.strip()
    column_names['Name'] = column_names['Name'].str.lower()
    column_names['Name'] = column_names['Name'].apply(
        lambda x: correct_col_names()[x] if x in correct_col_names() else x)

    for filename in unsw_data(test):
        raw_data_part = pd.read_csv(dataset_dir() + "/" + filename, sep=",", header=None,
                                    names=column_names['Name'],
                                    dtype={'sport': 'string', 'dsport': 'string', 'attack_cat': 'string'},
                                    converters={'ct_ftp_cmd': empty_string_to_nan, 'is_ftp_login': empty_string_to_nan,
                                                'ct_flw_http_mthd': empty_string_to_nan},
                                    encoding='ISO-8859-1')
        raw_data_list.append(raw_data_part)

    raw_data = pd.concat(raw_data_list)

    clean_data(raw_data)
    raw_data.drop_duplicates(inplace=True, subset=None, keep='first')
    return raw_data

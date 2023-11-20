import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import empty_string_to_nan, irrelevant_features, external_data_dir, figures_dir, data_dir, \
    read_prepare_dir


def clean_data(raw_data):
    raw_data['attack_cat'] = raw_data['attack_cat'].replace(np.nan, 'Normal')
    raw_data['attack_cat'] = raw_data['attack_cat'].str.strip()
    # raw_data.ct_ftp_cmd = raw_data.ct_ftp_cmd.fillna(0)
    # raw_data['ct_flw_http_mthd'] = raw_data['ct_flw_http_mthd'].replace('', 0)
    # raw_data['is_ftp_login'] = raw_data['is_ftp_login'].replace('', 0)
    raw_data['attack_cat'] = raw_data['attack_cat'].replace('Backdoor', 'Backdoors')
    # for feat in irrelevant_features():
    #    raw_data.drop(feat, axis=1, inplace=True)
    return raw_data


def add_column_names(raw_data):
    columns = pd.read_csv(external_data_dir() + "/" + 'UNSW-NB15_features.csv', encoding='ISO-8859-1')
    print(columns.Name.head())
    columns['Name'] = columns['Name'].str.strip()
    raw_data.columns = columns['Name']


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
    # rd_attacks.attack_cat.value_counts(normalize=False).plot(kind='bar', figsize=(10, 8))
    plt.figure(figsize=(12, 8))
    # plt.grid()
    sns.barplot(x=rd_attacks.attack_cat.value_counts().index, y=rd_attacks.attack_cat.value_counts()).set(title='')
    # sns.set(rc={"figure.figsize": (10, 8)})
    # print(rd_attacks.attack_cat.value_counts(normalize=False))
    # d=rd_attacks.attack_cat.value_counts(normalize=False)
    # sns.barplot(data=d)
    plt.savefig(figures_dir() + '/' + 'attack_counts.png')
    print("number of attacks:" + str(len(rd_attacks)))
    print("\n")
    print('number of duplicates')
    print(raw_data.duplicated(subset=None, keep='first').sum())


def read_data(unsw_data, test):
    raw_data = pd.DataFrame()
    raw_data_list = []

    column_names = pd.read_csv(external_data_dir() + "/" + 'UNSW-NB15_features.csv', encoding='ISO-8859-1')
    # print(columns.Name.head())
    column_names['Name'] = column_names['Name'].str.strip()

    for filename in unsw_data(test):
        raw_data_part = pd.read_csv(external_data_dir() + "/" + filename, sep=",", header=None,
                                    names=column_names['Name'],
                                    dtype={'sport': 'string', 'dsport': 'string', 'attack_cat': 'string'},
                                    converters={'ct_ftp_cmd': empty_string_to_nan, 'is_ftp_login': empty_string_to_nan,
                                                'ct_flw_http_mthd': empty_string_to_nan},
                                    encoding='ISO-8859-1')
        print(filename)
        print('1:', raw_data_part.columns[1])
        print('3:', raw_data_part.columns[3])
        print('47:', raw_data_part.columns[47])
        print('37:', raw_data_part.columns[37])
        print('38:', raw_data_part.columns[38])
        print('39:', raw_data_part.columns[39])
        raw_data_list.append(raw_data_part)

    raw_data = pd.concat(raw_data_list)
    print(raw_data.head(5))
    print('number of duplicates')
    # print(raw_data.duplicated(subset=None, keep='first').sum())
    raw_data.drop_duplicates(inplace=True, subset=None, keep='first')
    print(raw_data.shape)
    print(raw_data.head())

    # print('number of duplicates')
    # print(raw_data.duplicated(subset=None, keep='first').sum())

    if test:
        raw_data = raw_data.sample(frac=0.75)

    # add_column_names(raw_data)

    # print('number of duplicates')
    # print(raw_data.duplicated(subset=None, keep='first').sum())
    print('cleaning data')
    clean_data(raw_data)

    print('number of duplicates')
    print(raw_data.duplicated(subset=None, keep='first').sum())

    print(raw_data.shape)
    print(raw_data.head())

    print("data read, column names added, data cleaned")
    return raw_data

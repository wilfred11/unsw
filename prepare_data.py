import pandas as pd
from sklearn.preprocessing import StandardScaler
from functions import numeric_features
from functions import non_numeric_features
from sklearn.utils import resample


def standardize(raw_data):
    print('standardizing data')
    scaler = StandardScaler()
    raw_data_numeric_std = pd.DataFrame(data=scaler.fit_transform(raw_data[numeric_features(raw_data)]),
                                        columns=numeric_features(raw_data))
    raw_data_std = pd.merge(raw_data_numeric_std, raw_data[non_numeric_features()], left_index=True, right_index=True)
    return raw_data_std


def denominalize(raw_data):
    print('denominalizing data')
    raw_data = pd.get_dummies(raw_data, columns=["proto"])
    raw_data = pd.get_dummies(raw_data, columns=["service"])
    raw_data = pd.get_dummies(raw_data, columns=["state"])
    return raw_data


def filter_on_attack_cat(raw_data, attack_cat="Normal"):
    result = raw_data[raw_data['attack_cat'] == attack_cat]
    return result


def prepare_data_for_specific_attack_cat(raw_data, attack_cat, test):
    if test:
        half_max = 75
    else:
        half_max = 100000

    raw_data_tmp = raw_data.copy()
    if attack_cat!='Normal':
        raw_data_tmp.loc[raw_data_tmp['attack_cat'] != attack_cat, 'attack_cat'] = "Normal"

    number_of_attack_cat = len(raw_data_tmp[raw_data_tmp['attack_cat'] == attack_cat])
    number_of_normal = len(raw_data_tmp[raw_data_tmp['attack_cat'] == "Normal"])

    if attack_cat != 'Normal':
        X_attack = raw_data_tmp[raw_data_tmp.attack_cat == attack_cat]
    else:
        X_attack = raw_data_tmp[raw_data_tmp.attack_cat != attack_cat]

    X_normal = raw_data_tmp[raw_data_tmp.attack_cat == "Normal"]

    X_normal_max = resample(X_normal, replace=False, n_samples=half_max, random_state=0)

    if number_of_attack_cat <= half_max:
        X_attack_max = resample(X_attack, replace=False, n_samples=half_max, random_state=0)
    else:
        X_attack_max = resample(X_attack, replace=True, n_samples=half_max, random_state=0)

    X_complete = pd.concat([X_attack_max, X_normal_max])
    raw_data_tmp = None
    return X_complete


def remove_target_columns(raw_data):
    X = raw_data.drop('attack_cat', axis=1).copy()
    X = X.drop('Label', axis=1)
    # raw_data.to_csv('check.csv')
    # print(X.head())
    y = raw_data.Label.copy()
    return X, y


def remove_columns(data, columns):
    for col in columns:
        data = data.drop(col, axis=1).copy()
    return data

# def split_data(raw_data):

import mpu
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from functions import numeric_features, features_to_be_denominalized, feature_reduction_dir
from functions import non_numeric_features
from sklearn.utils import resample


def standardize(raw_data):
    print('standardizing data')
    # https://towardsdatascience.com/methods-for-normality-test-with-application-in-python-bb91b49ed0f5

    scaler = StandardScaler(with_std=True, with_mean=True)
    raw_data_numeric_std = pd.DataFrame(data=scaler.fit_transform(raw_data[numeric_features(raw_data)]),
                                        columns=numeric_features(raw_data))
    raw_data_numeric_std.reset_index(inplace=True)
    print('mean')
    print(raw_data_numeric_std.mean(axis=0))
    print('std')
    print(raw_data_numeric_std.std(axis=0))

    raw_data.reset_index(inplace=True)
    raw_data = pd.concat([raw_data[non_numeric_features()], raw_data_numeric_std], axis=1)
    return raw_data


def min_max(raw_data):
    print('standardizing data')
    # https://towardsdatascience.com/methods-for-normality-test-with-application-in-python-bb91b49ed0f5

    scaler = MinMaxScaler()
    raw_data_numeric_std = pd.DataFrame(data=scaler.fit_transform(raw_data[numeric_features(raw_data)]),
                                        columns=numeric_features(raw_data))
    raw_data_numeric_std.reset_index(inplace=True)
    print('min')
    print(raw_data_numeric_std.min(axis=0))
    print('max')
    print(raw_data_numeric_std.max(axis=0))

    raw_data.reset_index(inplace=True)
    raw_data = pd.concat([raw_data[non_numeric_features()], raw_data_numeric_std], axis=1)
    return raw_data


def denominalize(raw_data):
    print('denominalizing data')
    raw_data = pd.get_dummies(raw_data, columns=["proto"])
    raw_data = pd.get_dummies(raw_data, columns=["service"])
    raw_data = pd.get_dummies(raw_data, columns=["state"])
    return raw_data


def filter_on_attack_cat(raw_data, attack_cat="Normal"):
    result = raw_data[raw_data['attack_cat'] == attack_cat]
    return result


def prepare_data_for_umap(raw_data):
    raw_data_tmp = raw_data.copy()
    normal = raw_data_tmp[raw_data_tmp['attack_cat'] == "Normal"]
    X_attack = pd.DataFrame()
    # attacks = dict()
    for attack_cat in raw_data_tmp.attack_cat.unique():
        if attack_cat != 'Normal':
            tmp = raw_data_tmp[raw_data_tmp['attack_cat'] == attack_cat]
            print('tmp:', len(tmp))
            X_attack = pd.concat([X_attack, tmp])

    X_normal = resample(normal, replace=False, n_samples=int(len(X_attack)), random_state=0)
    X_complete = pd.concat([X_attack, X_normal])
    return X_complete


def prepare_data_for_normal(raw_data, size):
    raw_data_tmp = raw_data.copy()
    X_complete = pd.DataFrame()
    for attack_cat in raw_data_tmp.attack_cat.unique():
        tmp_attack_cat = raw_data_tmp[raw_data_tmp['attack_cat'] == attack_cat]
        if len(tmp_attack_cat) >0:
            if len(tmp_attack_cat) < size / len(raw_data_tmp.attack_cat.unique()):
                X_attack_cat = resample(tmp_attack_cat, replace=True, n_samples=int(size / len(raw_data_tmp.attack_cat.unique())), random_state=0)
            else:
                X_attack_cat = resample(tmp_attack_cat, replace=False, n_samples=int(size / len(raw_data_tmp.attack_cat.unique())), random_state=0)

        X_complete = pd.concat([X_complete, X_attack_cat])
    return X_complete


def prepare_data_for_specific_attack_cat(raw_data, attack_cat, size, exclude_other_attacks=True, reduce_cats=False,
                                         denominalize=False):
    raw_data_tmp = raw_data.copy()
    number_of_attack_cat = 0
    number_of_normal = 0
    if attack_cat != 'Normal':
        if exclude_other_attacks:
            raw_data_tmp.drop(raw_data_tmp[~raw_data_tmp['attack_cat'].isin([attack_cat, 'Normal'])].index,
                              inplace=True)
        else:
            raw_data_tmp.loc[raw_data_tmp['attack_cat'] != attack_cat, 'Label'] = 0
            raw_data_tmp.loc[raw_data_tmp['attack_cat'] != attack_cat, 'attack_cat'] = "Normal"

        number_of_attack_cat = len(raw_data_tmp[raw_data_tmp['attack_cat'] == attack_cat])
        number_of_normal = len(raw_data_tmp[raw_data_tmp['attack_cat'] == "Normal"])
    if attack_cat == 'Normal':
        # raw_data_tmp.loc[raw_data_tmp['attack_cat'] != attack_cat, 'attack_cat'] = "Normal"
        # Todo evenly spread attacks
        number_of_attack_cat = len(raw_data_tmp[raw_data_tmp['attack_cat'] != 'Normal'])
        number_of_normal = len(raw_data_tmp[raw_data_tmp['attack_cat'] == "Normal"])

    if attack_cat != 'Normal':
        X_attack = raw_data_tmp[raw_data_tmp.attack_cat == attack_cat]
    else:
        X_attack = raw_data_tmp[raw_data_tmp.attack_cat != attack_cat]

    X_normal = raw_data_tmp[raw_data_tmp.attack_cat == "Normal"]

    X_normal_max = resample(X_normal, replace=False, n_samples=int(size / 2), random_state=0)

    if number_of_attack_cat < size / 2:
        X_attack_max = resample(X_attack, replace=True, n_samples=int(size / 2), random_state=0)
    else:
        X_attack_max = resample(X_attack, replace=False, n_samples=int(size / 2), random_state=0)

    X_complete = pd.concat([X_attack_max, X_normal_max])
    print(X_complete.head())
    if reduce_cats:
        handle_categorical_data(X_complete)
        X_complete = reduce_categories(X_complete)
    if denominalize:
        X_complete = denominalize(X_complete)

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


def create_column_name(row):
    if len(row['feat_agg']) == 1:
        return row['feat_agg'][0]
    elif len(row['feat_agg']) == 2:
        return row['feat_agg'][0] + '--' + row['feat_agg'][1]
    elif len(row['feat_agg']) > 2:
        return row['feat_agg'][0] + '--' + row['feat_agg'][1] + '...'


def handle_categorical_data(raw_data):
    for feature in features_to_be_denominalized():
        ct = pd.crosstab(raw_data[feature], raw_data['attack_cat'], normalize='index').round(2)
        with open(feature_reduction_dir() + '/' + feature + "-attack-ct.txt", "w") as text_file:
            text_file.write(ct.to_latex())
        ct.to_excel(feature_reduction_dir() + '/' + feature + '-attack-ct.xlsx')
        ct = ct.multiply(10)
        ct = ct.apply(np.ceil)
        ct.to_excel(feature_reduction_dir() + '/' + feature + '-attack-ct-mp.xlsx')
        l = ct.columns.to_list()
        ct = ct.reset_index()
        result = ct.groupby(l, as_index=False).agg({feature: lambda x: list(x)})
        result.to_excel(feature_reduction_dir() + '/' + feature + '-attack-gb.xlsx')
        df = pd.DataFrame()
        df = pd.DataFrame({'feat': result[feature], 'feat_agg': result[feature]})
        df = df.explode('feat')
        df['column_name'] = df.apply(lambda x: create_column_name(x), axis=1)
        df.to_excel(feature_reduction_dir() + '/' + feature + '-attack-aggregated.xlsx')
        feat_dict = dict(zip(df.feat, df.column_name))
        mpu.io.write(feature_reduction_dir() + '/' + feature + '_cat_dict.pickle', feat_dict)


def reduce_categories(raw_data):
    for feature in features_to_be_denominalized():
        feat_dict = mpu.io.read(feature_reduction_dir() + '/' + feature + '_cat_dict.pickle')
        raw_data = raw_data.replace({feature: feat_dict})
    return raw_data
# def split_data(raw_data):

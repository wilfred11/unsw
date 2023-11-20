import pandas as pd
import seaborn as sns
import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy import arange

from create_figures import create_results_plot_all, correlated_features, pairplot, my_umap
from functions import unsw_data, cleanup_project_dirs, external_data_dir, feature_reduction_dir, numeric_features, \
    read_prepare_dir, read_prepare_figs_dir
from read_data import read_data, info
from inspect_data import inspect_for_empty_or_na_columns
from prepare_data import standardize, denominalize
from project import test_classifiers, handle_categorical_data, reduce_categories, reduce_features_lasso, \
    train_reduce_test
from wakepy import keep
from scipy import stats


def feature_props(column_values, column_props_global, column_name, column_description, size):
    column_values_sampled = column_values.sample(size, random_state=0)
    new_row1 = {'name': column_name, 'description': column_description,
                'max_value': column_values.max(),
                'min_value': column_values.min(),
                'kurtosis': scipy.stats.kurtosis(column_values_sampled, bias=True),
                'skewness': scipy.stats.skew(column_values_sampled, bias=True),
                'normal': scipy.stats.normaltest(column_values_sampled)[1] > 0.5,
                'nunique_values': column_values.nunique()}

    column_props_global = pd.concat([column_props_global, pd.DataFrame([new_row1])], ignore_index=True)
    column_props_global.to_excel(read_prepare_dir() + '/' + 'column-props.xlsx')

    plt.figure(figsize=(8, 5))
    # sns.histplot(data=column_values_sampled.to_frame(), x=column_name)
    # plt.savefig(read_prepare_figs_dir() + '/' + 'hist-' + column_name + '.png')

    hist_values, bin_edges = np.histogram(column_values_sampled)
    plt.bar(x=bin_edges[:-1], height=hist_values / len(column_values_sampled), width=np.diff(bin_edges), align='edge')
    # y = arange(0, hist_values.max(), hist_values.max()/5)
    # plt.yticks(y)
    plt.savefig(read_prepare_figs_dir() + '/' + column_name + '-hist.png')
    plt.close('all')

    with open(read_prepare_figs_dir() + '/' + column_name + '-tab.txt', 'w', encoding='utf-8') as text_file:
        nr = pd.DataFrame(new_row1.items())
        text_file.write(nr.to_latex(header=False, index=False))
    return column_props_global

    # plt.show()


test = False
# execute=1: read data, handle categorical data, standardize data, denominalize data, correlated features, save prepared data
# execute=2 test classifiers
# execute=3 lasso


execute = 6

sns.set_style("darkgrid")

if execute == 2:
    with keep.running() as k:
        raw_data = pd.read_csv(external_data_dir() + '/' + 'raw_data_prepared.csv', index_col=None)

        # test_classifiers(raw_data, test)
        test_classifiers(raw_data, test, ['dt', 'svm'], 1000)

        # TODO find optimal features
        # TODO train model with optimal features
        # TODO check which attack_data is classified as normal
        # TODO make classifier Neural network

        # print(raw_data.attack_cat.unique())
        # crosstab_service_to_attack_cat(raw_data)

        # pd.reset_option('^display.', silent=True)

        # n_neighbors=optimize_lof_parameters(raw_data)

        # outliers(raw_data, n_neighbors)

elif execute == 1:
    with keep.running() as k:
        cleanup_project_dirs()
        # TODO drop duplicate rows
        # TODO drop constant columns
        raw_data = pd.DataFrame()

        raw_data = read_data(unsw_data, test)

        info(raw_data)

        inspect_for_empty_or_na_columns(raw_data)

        handle_categorical_data(raw_data)
        raw_data = reduce_categories(raw_data)
        # print(raw_data.shape)
        # print(raw_data.head())
        # TODO handle outliers
        raw_data = standardize(raw_data)

        # print(raw_data.shape)
        # print(raw_data.head())

        proto = raw_data.proto.copy()
        service = raw_data.service.copy()
        state = raw_data.state.copy()
        is_ftp_login = raw_data.is_ftp_login.copy()
        is_sm_ips_ports = raw_data.is_sm_ips_ports.copy()
        raw_data = raw_data.drop('proto', axis=1)
        raw_data = raw_data.drop('service', axis=1)
        raw_data = raw_data.drop('state', axis=1)
        raw_data = raw_data.drop('is_sm_ips_ports', axis=1)
        raw_data = raw_data.drop('is_ftp_login', axis=1)

        pairplot(raw_data, 'Normal', 1000)

        raw_data['service'] = service
        raw_data['proto'] = proto
        raw_data['state'] = state
        raw_data['is_ftp_login'] = is_ftp_login
        raw_data['is_sm_ips_ports'] = is_sm_ips_ports

        raw_data = denominalize(raw_data)
        # print(raw_data.shape)
        # print(raw_data.head())

        attack_cat = raw_data.attack_cat.copy()
        Label = raw_data.Label.copy()
        raw_data = raw_data.drop('attack_cat', axis=1)
        raw_data = raw_data.drop('Label', axis=1)

        correlated_features(raw_data)

        raw_data['attack_cat'] = attack_cat
        raw_data['Label'] = Label

        raw_data.to_csv(external_data_dir() + '/' + 'raw_data_prepared.csv', index=False)
        # reduce_features(raw_data, 'Normal')
        # https://medium.com/analytics-vidhya/feature-selection-using-scikit-learn-5b4362e0c19b
        # https: // thepythoncode.com / article / dimensionality - reduction - using - feature - extraction - sklearn

elif execute == 3:
    with keep.running() as k:
        raw_data = pd.read_csv(external_data_dir() + '/' + 'raw_data_prepared.csv', index_col=0)
        print(raw_data.shape)
        print(raw_data.head())
        # raw_data.head().to_csv(external_data_dir() + '/' + 'raw_data_prepared1.csv', index=False)
        reduce_features_lasso(raw_data)

elif execute == 4:
    with keep.running() as k:
        raw_data = pd.read_csv(external_data_dir() + '/' + 'raw_data_prepared.csv', index_col=0)
        train_reduce_test(raw_data, ['svm'], ['Normal'], 100000)


elif execute == 5:
    with keep.running() as k:
        raw_data = pd.read_csv(external_data_dir() + '/' + 'raw_data_prepared.csv', index_col=0)
        my_umap(raw_data)

elif execute == 6:
    # https://www.kaggle.com/code/khairulislam/unsw-nb15- eda
    column_props = pd.DataFrame()
    cleanup_project_dirs()
    raw_data = read_data(unsw_data, test)

    columns_info = pd.read_csv(external_data_dir() + "/" + 'UNSW-NB15_features.csv', encoding='ISO-8859-1')
    columns_info['Name'] = columns_info['Name'].str.strip()

    name_desc_dict = dict(zip(columns_info.Name, columns_info.Description))

    # raw_data.columns = columns['Name']

    c = numeric_features(raw_data)
    rd = raw_data[c].copy()
    i = 0
    for feature_name, feature_values in rd.items():
        print(feature_name)
        print('#', i)

        column_props = feature_props(feature_values, column_props, feature_name, name_desc_dict[feature_name], 10000)
        i = i + 1

        # feature_props(rd['ct_src_dport_ltm'], column_props, 'ct_src_dport_ltm', 'column_description', 1000)

    # column_props.to_excel(read_prepare_dir() + '/' + 'column_props.xlsx')
else:
    corr_mat = pd.read_pickle(feature_reduction_dir() + "/feature_corrs.pkl")
    features_corr = pd.DataFrame({'column': [], 'other_column': [], 'correlation': []})

    columns = corr_mat.columns
    for i in range(corr_mat.shape[0]):
        for j in range(i + 1, corr_mat.shape[0]):
            if corr_mat.iloc[i, j] >= 0.95 or corr_mat.iloc[i, j] <= -0.95:
                print(f"{columns[i]:20s} {columns[j]:20s} {corr_mat.iloc[i, j]}")
                new_row = {'column': columns[i],
                           'other_column': columns[j],
                           'correlation': corr_mat.iloc[i, j]
                           }

                features_corr = pd.concat([features_corr, pd.DataFrame([new_row])], ignore_index=True)
    print(features_corr)
    features_corr.to_excel(external_data_dir() + '/' + 'features_corr.xlsx', index=False)
    create_results_plot_all()

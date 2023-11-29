import pandas as pd
import seaborn as sns
from pandas import read_csv

from create_figures import create_results_plot_all, correlated_features, pairplot, my_umap
from functions import unsw_data, cleanup_project_dirs, external_data_dir, feature_reduction_dir, numeric_features, \
    read_prepare_dir, read_prepare_figs_dir
from read_data import read_data, info
from inspect_data import numeric_feature_inspection, inspect_for_empty_or_na_columns
from prepare_data import standardize, denominalize, min_max, handle_categorical_data, reduce_categories, \
    prepare_data_for_umap, get_balanced_dataset, remove_low_variance_columns
from project import test_classifiers, train_reduce_test, test_classifiers_basic, \
    reduce_features_lasso_balanced
from wakepy import keep

test = False
# execute=1: read data, handle categorical data, standardize data, denominalize data, low variance columns, save prepared data
# execute=2 test model cv balanced
# execute=3 correlated, remove features, save adapted dataset
# execute=4 lasso, remove features, save adapted dataset


execute = 2

sns.set_style("darkgrid")

if execute == 6:
    with keep.running() as k:
        raw_data = pd.read_csv(external_data_dir() + '/' + 'raw_data_prepared.csv', index_col=None)

        # test_classifiers(raw_data, test)
        test_classifiers(raw_data, test, ['dt', 'svm'], 1000)

        create_results_plot_all()

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

        raw_data = pd.DataFrame()

        raw_data = read_data(unsw_data, test)

        info(raw_data)

        inspect_for_empty_or_na_columns(raw_data)

        numeric_feature_inspection(raw_data)

        # TODO handle outliers
        raw_data = min_max(raw_data)

        raw_data.to_csv(external_data_dir() + '/' + 'raw_data_std.csv', index=False)

        raw_data_balanced = get_balanced_dataset(raw_data, 2000000)

        handle_categorical_data(raw_data_balanced)
        raw_data = reduce_categories(raw_data)

        raw_data = denominalize(raw_data)

        raw_data = remove_low_variance_columns(raw_data)

        print(raw_data.shape)
        print(raw_data.head())


        raw_data.to_csv(external_data_dir() + '/' + 'raw_data_std_denom_var.csv', index=False)

        # create_results_plot_all()
        # reduce_features(raw_data, 'Normal')
        # https://medium.com/analytics-vidhya/feature-selection-using-scikit-learn-5b4362e0c19b
        # https: // thepythoncode.com / article / dimensionality - reduction - using - feature - extraction - sklearn

elif execute == 2:
    with keep.running() as k:
        cleanup_project_dirs()
        raw_data = read_csv(external_data_dir() + '/' + 'raw_data_std_denom_var.csv')
        print(raw_data.shape)
        print(raw_data.head(5))
        print(raw_data.columns)
        test_classifiers_basic(raw_data, ['dt'], 2000000, scoring=False, cm=True)
        # test_classifiers(raw_data, test, ['dt', 'svm'], 1000, ['Normal'], False)


elif execute == 3:
    with keep.running() as k:
        raw_data = read_csv(external_data_dir() + '/' + 'raw_data_std_denom_var.csv')
        raw_data = raw_data.drop('attack_cat', axis=1)
        raw_data = raw_data.drop('Label', axis=1)
        correlated_features(raw_data)


elif execute == 4:
    with keep.running() as k:
        #https://tahera-firdose.medium.com/lasso-regression-a-comprehensive-guide-to-feature-selection-and-regularization-2c6a20b61e23
        raw_data = read_csv(external_data_dir() + '/' + 'raw_data_std_denom_var.csv')
        raw_data = raw_data.drop('Stime', axis=1)
        raw_data = raw_data.drop('Ltime', axis=1)
        raw_data = raw_data.drop('dloss', axis=1)
        raw_data = raw_data.drop('Dpkts', axis=1)
        raw_data = raw_data.drop('swin', axis=1)
        raw_data = raw_data.drop('index', axis=1)
        raw_data.reset_index(drop=True, inplace=True)
        print(raw_data.columns)
        raw_data.is_ftp_login = raw_data.is_ftp_login.astype('bool')
        raw_data.is_sm_ips_ports = raw_data.is_sm_ips_ports.astype('bool')

        raw_data_bool = raw_data.select_dtypes(include='bool')
        raw_data_numeric = raw_data.select_dtypes(exclude='bool')

        result = raw_data.dtypes
        pd.set_option("display.max_rows", 70)
        print(result)
        print(raw_data.shape)
        print(raw_data.head(5))

        pairplot(raw_data_numeric, 'Normal', 2500)

        print('afte pp')
        print(raw_data.shape)
        print(raw_data.head(5))
        raw_data = pd.concat([raw_data_bool, raw_data_numeric],  axis=1)
        reduce_features_lasso_balanced(raw_data)



elif execute == 5:
    with keep.running() as k:
        raw_data = read_csv(external_data_dir() + '/' + 'raw_data_std_denom_var.csv')
        #raw_data = raw_data.drop('Stime', axis=1)
        #raw_data = raw_data.drop('Ltime', axis=1)
        #raw_data = raw_data.drop('dloss', axis=1)
        #raw_data = raw_data.drop('Dpkts', axis=1)
        #raw_data = raw_data.drop('swin', axis=1)
        #raw_data = raw_data.drop('Spkts', axis=1)
        raw_data.to_csv(external_data_dir() + '/' + 'raw_data_std_denom_var.csv', index=False)

elif execute == 6:
    with keep.running() as k:
        raw_data = pd.read_csv(external_data_dir() + '/' + 'raw_data_prepared.csv', index_col=0)
        train_reduce_test(raw_data, ['svm'], ['Normal'], 100000)


elif execute == 7:
    with keep.running() as k:
        raw_data = pd.read_csv(external_data_dir() + '/' + 'raw_data_std.csv')
        print(raw_data.head())
        raw_data = prepare_data_for_umap(raw_data)
        handle_categorical_data(raw_data)
        raw_data = reduce_categories(raw_data)
        raw_data = denominalize(raw_data)
        my_umap(raw_data)


else:
    pass
    # cats = raw_data.attack_cat.unique().to_list()
    # cats = ['Normal']

    # test_classifiers(raw_data, test, ['dt', 'svm'], 1000, cats, False)

    # handle_categorical_data(raw_data)
    # raw_data = reduce_categories(raw_data)

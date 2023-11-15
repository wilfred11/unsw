import pandas as pd
import seaborn as sns
from create_figures import create_results_plot_all, correlated_features, pairplot, my_umap
from functions import unsw_data, cleanup_project_dirs, external_data_dir, feature_reduction_dir
from read_data import read_data, info
from inspect_data import inspect_for_empty_or_na_columns
from prepare_data import standardize, denominalize
from project import test_classifiers, handle_categorical_data, reduce_categories, reduce_features_lasso, \
    train_reduce_test
from wakepy import keep

test = False
# execute=1: read data, handle categorical data, standardize data, denominalize data, correlated features, save prepared data
# execute=2 test classifiers
# execute=3 lasso


execute = 5

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
        #print(raw_data.shape)
        #print(raw_data.head())
        # TODO handle outliers
        raw_data = standardize(raw_data)

        #print(raw_data.shape)
        #print(raw_data.head())

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
        #print(raw_data.shape)
        #print(raw_data.head())

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

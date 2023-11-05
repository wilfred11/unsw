import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from create_figures import create_results_plot, create_results_plot_all, correlated_features, pairplot
from functions import unsw_data, cleanup_project_dirs, data_dir, keep_numeric_columns, feature_reduction_dir
from read_data import read_data, info
from inspect_data import inspect_for_empty_or_na_columns
from prepare_data import standardize, denominalize
from project import test_classifiers, reduce_features, lasso, handle_categorical_data
# from feature_engine.selection import DropConstantFeatures
from wakepy import keep

# LOKY_MAX_CPU_COUNT = str(len(psutil.Process().cpu_affinity()))

test = False

execute = 2

sns.set_style("darkgrid")

if execute == 1:
    with keep.running() as k:
        cleanup_project_dirs()

        raw_data = pd.DataFrame()

        raw_data = read_data(unsw_data, test)

        info(raw_data)

        inspect_for_empty_or_na_columns(raw_data)

        raw_data = standardize(raw_data)

        raw_data = denominalize(raw_data)

        # handle outliers

        # print(raw_data.head())
        print('data shape', raw_data.shape)
        raw_data.to_csv(data_dir() + '/' + 'raw_data_prepared.csv')

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

elif execute == 2:
    with keep.running() as k:
        cleanup_project_dirs()

        raw_data = pd.DataFrame()

        raw_data = read_data(unsw_data, test)
        handle_categorical_data(raw_data)

        """
        raw_data = pd.read_csv(data_dir() + "/" + 'raw_data_prepared.csv', index_col=0)
        raw_data_nobools = keep_numeric_columns(raw_data, exclude_targets=False)
        print(raw_data_nobools.shape)
        print(raw_data_nobools.head())
        #correlated_features(raw_data_nobools)

        #lasso(raw_data, 'Normal')
        pairplot(raw_data_nobools, 'Normal')
        """
        # reduce_features(raw_data, 'Normal')
        # https://medium.com/analytics-vidhya/feature-selection-using-scikit-learn-5b4362e0c19b
        # https: // thepythoncode.com / article / dimensionality - reduction - using - feature - extraction - sklearn


else:
    create_results_plot_all()

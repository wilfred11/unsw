import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from create_figures import create_results_plot, create_results_plot_all
from functions import unsw_data, cleanup_project_dirs, data_dir
from read_data import read_data, info
from inspect_data import inspect_for_empty_or_na_columns
from prepare_data import standardize, denominalize
from project import test_classifiers, reduce_features
from wakepy import keep

# LOKY_MAX_CPU_COUNT = str(len(psutil.Process().cpu_affinity()))

test = True

execute = 1

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

        #test_classifiers(raw_data, test)
        test_classifiers(raw_data, test, ['dt', 'svm'])


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
        raw_data = pd.read_csv(data_dir() + "/" + 'raw_data_prepared.csv')
        #reduce_features(raw_data, test, 'Normal')
        #https://medium.com/analytics-vidhya/feature-selection-using-scikit-learn-5b4362e0c19b
        #https: // thepythoncode.com / article / dimensionality - reduction - using - feature - extraction - sklearn


else:
    create_results_plot_all()

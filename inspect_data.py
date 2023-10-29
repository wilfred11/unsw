import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import Birch
from matplotlib import pyplot
from functions import numeric_features, data_dir


def crosstab_service_to_attack_cat(raw_data):
    raw_data.proto.value_counts()
    ct = pd.crosstab(raw_data['service'], raw_data['attack_cat'])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(ct)


def outliers(raw_data, n_neighbors):
    X_before = raw_data[numeric_features(raw_data)]
    print(X_before.head())
    print(numeric_features(raw_data))
    y_before = raw_data['Label']
    Xy_before = pd.merge(X_before, y_before, left_index=True, right_index=True)
    print(Xy_before.head())
    clf = LocalOutlierFactor(novelty=False, n_neighbors=n_neighbors, contamination=0.01)
    clf.fit_predict(X_before)
    X_before = pd.DataFrame(X_before)
    X_before["negative_outlier_factor"] = clf.negative_outlier_factor_
    record_count = len(X_before)
    outlier_count = len(X_before[(X_before.negative_outlier_factor < -1.5) | (X_before.negative_outlier_factor > -0.5)])
    print("\n")
    print("number of outliers:" + str(outlier_count))
    print("number of records:" + str(record_count))
    print("outlier fraction:" + str(outlier_count / record_count))


def optimize_lof_parameters(raw_data):
    X_before = raw_data[numeric_features(raw_data)]
    y_before = raw_data['Label']

    n = 30  # Max number of neighbours you want to consider
    param_grid = {'n_neighbors': np.arange(10, 30)}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy", verbose=2)

    X_train, X_test, y_train, y_test = train_test_split(X_before, y_before, stratify=y_before, random_state=42)

    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(str(grid.best_params_['n_neighbors']))
    return grid.best_params_['n_neighbors']
    # model = grid.best_estimator_
    # y_pred = model.fit(X_train, y_train).predict(X_test)


def find_clustering(raw_data):
    # define the model
    model = Birch(threshold=1.5, n_clusters=None, branching_factor=50)
    # fit the model
    model.fit(raw_data)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    pyplot.show()


def inspect_for_empty_or_na_columns(raw_data):
    data_na = raw_data.isna().sum()
    data_empty = raw_data.eq('').sum()
    data_na.to_csv(data_dir()+'/'+ 'columns_na_count.csv')
    data_empty.to_csv(data_dir() + '/' + 'columns_empty_count.csv')

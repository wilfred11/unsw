import numpy
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from functions import data_dir, test_classifiers_dir
from model_functions import classify, grid_search, reduce_features_by_clf
from prepare_data import prepare_data_for_specific_attack_cat, remove_target_columns


def test_classifiers(raw_data, test):
    cats = list(raw_data.attack_cat.unique().to_numpy())
    # cats.remove('Normal')
    print(cats)
    kinds = ['svm', 'lr', 'knn']
    scoring_values = ['fit_time', 'score_time', 'test_accuracy', 'test_precision', 'test_recall']
    results_pre = pd.DataFrame()
    results_post = pd.DataFrame()
    params_svm_ = pd.DataFrame()
    params_knn_ = pd.DataFrame()
    params_lr_ = pd.DataFrame()

    for attack_cat in cats:
        print('**************************')
        attack_cat_data = prepare_data_for_specific_attack_cat(raw_data, attack_cat, test)
        X, y = remove_target_columns(attack_cat_data)

        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        lr_classifier = LogisticRegression(max_iter=400)
        svm_classifier = svm.SVC()

        scores_svm_ = classify(svm_classifier, X, y, 'svm', attack_cat)

        scores_lr_ = classify(lr_classifier, X, y, 'lr', attack_cat)
        scores_knn_ = classify(knn_classifier, X, y, 'knn', attack_cat)

        scores_pre = pd.concat([scores_svm_, scores_lr_, scores_knn_], axis=1).T

        scores_pre_indexed = scores_pre.set_index(
            [pd.Series([attack_cat, attack_cat, attack_cat]).repeat(6), pd.Series(kinds).repeat(6),
             scores_pre.index.to_series()])
        results_pre = pd.concat([results_pre, scores_pre_indexed])

        svm_param_grid = [{'C': [0.5, 1, 10, 100], 'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}]
        opt_svm_params = grid_search(svm_classifier, svm_param_grid, X, y, 'svm', attack_cat)
        params_svm = pd.DataFrame(opt_svm_params[0], index=[0])
        params_svm_indexed = params_svm.set_index([pd.Series([attack_cat])])
        params_svm_indexed.index.name = 'attack_cat'
        params_svm_ = pd.concat([params_svm_, params_svm_indexed])


        #opt_svm_params[1].to_csv(test_classifiers_dir() + '/' + attack_cat + '_svm_' + 'optimal_params.csv')
        svm_classifier = svm.SVC(C=opt_svm_params[0]['C'], gamma=opt_svm_params[0]['gamma'],
                                 kernel=opt_svm_params[0]['kernel'])

        k_range = list(range(1, 31))
        knn_param_grid = dict(n_neighbors=k_range)
        opt_knn_params = grid_search(knn_classifier, knn_param_grid, X, y, 'knn', attack_cat)

        params_knn = pd.DataFrame(opt_knn_params[0], index=[0])
        params_knn_indexed = params_knn.set_index([pd.Series([attack_cat])])
        params_knn_indexed.index.name = 'attack_cat'
        params_knn_ = pd.concat([params_knn_, params_knn_indexed])

        #opt_knn_params[1].to_csv(test_classifiers_dir() + '/' + attack_cat + '_knn_' + 'optimal_params.csv')

        knn_classifier = KNeighborsClassifier(n_neighbors=opt_knn_params[0]['n_neighbors'])

        lr_param_grid = {"C": np.logspace(0.1, 3, 30), "penalty": ["l1", "l2"], 'max_iter': [800, 1000, 1200],
                         'solver': ['saga'], 'dual': [False], 'tol': [1e-3]}
        opt_lr_params = grid_search(lr_classifier, lr_param_grid, X, y, 'lr', attack_cat)

        params_lr = pd.DataFrame(opt_lr_params[0], index=[0])
        params_lr_indexed = params_lr.set_index([pd.Series([attack_cat])])
        params_lr_indexed.index.name = 'attack_cat'
        params_lr_ = pd.concat([params_lr_, params_lr_indexed])
        #opt_lr_params[1].to_csv(test_classifiers_dir() + '/' + attack_cat + '_lr_' + 'optimal_params.csv')

        lr_classifier = LogisticRegression(C=opt_lr_params[0]['C'], penalty=opt_lr_params[0]['penalty'],
                                           max_iter=opt_lr_params[0]['max_iter'], solver=opt_lr_params[0]['solver'],
                                           dual=opt_lr_params[0]['dual'], tol=opt_lr_params[0]['tol'])

        scores_svm_post = classify(svm_classifier, X, y, 'svm', attack_cat, True)
        scores_lr_post = classify(lr_classifier, X, y, 'lr', attack_cat, True)
        scores_knn_post = classify(knn_classifier, X, y, 'knn', attack_cat, True)

        scores_post = pd.concat([scores_svm_post, scores_lr_post, scores_knn_post], axis=1).T
        # print(test1)
        scores_post_indexed = scores_post.set_index(
            [pd.Series([attack_cat, attack_cat, attack_cat]).repeat(6), pd.Series(kinds).repeat(6),
             scores_post.index.to_series()])
        results_post = pd.concat([results_post, scores_post_indexed])
        print(results_post)

    results_pre.to_pickle(test_classifiers_dir() + '/' + 'clf_results_pre.pkl')
    results_pre.to_excel(test_classifiers_dir() + '/' + 'clf_results_pre.xlsx')
    results_post.to_pickle(test_classifiers_dir() + '/' + 'clf_results_post.pkl')
    results_post.to_excel(test_classifiers_dir() + '/' + 'clf_results_post.xlsx')
    params_lr_.to_excel(test_classifiers_dir()+ '/lr_params.xlsx')
    params_knn_.to_excel(test_classifiers_dir() + '/knn_params.xlsx')
    params_svm_.to_excel(test_classifiers_dir() + '/svm_params.xlsx')
    params_lr_.to_pickle(test_classifiers_dir() + '/lr_params.pkl')
    params_knn_.to_pickle(test_classifiers_dir() + '/knn_params.pkl')
    params_svm_.to_pickle(test_classifiers_dir() + '/svm_params.pkl')


def reduce_features(raw_data, test):
    cats = raw_data.attack_cat.unique()
    # cats.remove('Normal')
    print(cats)
    kinds = ['svm', 'lr', 'knn']
    scoring_values = ['fit_time', 'score_time', 'test_accuracy', 'test_precision', 'test_recall']
    results_pre = pd.DataFrame()
    for attack_cat in cats:
        print('**************************')
        attack_cat_data = prepare_data_for_specific_attack_cat(raw_data, attack_cat, test)
        X, y = remove_target_columns(attack_cat_data)

        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        lr_classifier = LogisticRegression(max_iter=400)
        # svm_classifier = svm.SVC()

        reduce_features_by_clf(lr_classifier, X, y, 'knn', attack_cat)

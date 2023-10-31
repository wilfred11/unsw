import numpy
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from functions import data_dir, test_classifiers_dir
from model_functions import classify, grid_search, reduce_features_shap
from prepare_data import prepare_data_for_specific_attack_cat, remove_target_columns


def test_classifiers(raw_data, test, kinds):
    cats = list(raw_data.attack_cat.unique().to_numpy())
    scores_pre = pd.DataFrame()
    scores_post = pd.DataFrame()
    optimal_params = {}
    for kind in kinds:
        optimal_params[kind] = pd.DataFrame()

    print(optimal_params)
    for attack_cat in cats:
        print('**************************')
        attack_cat_data = prepare_data_for_specific_attack_cat(raw_data, attack_cat, test)
        X, y = remove_target_columns(attack_cat_data)
        for kind in kinds:
            clf = get_classifier(kind)
            print('scoring ' + kind + ' ' + attack_cat)
            scores = classify(clf, X, y)
            scores = scores.T
            scores = scores.set_index([pd.Series(attack_cat).repeat(len(scores)), pd.Series(kind).repeat(len(scores)),
                                       scores.index.to_series()])
            scores_pre = pd.concat([scores_pre, scores])

            pg = get_params(kind)
            print('optimizing ' + kind + ' ' + attack_cat)
            opt_params = grid_search(clf, pg, X, y)
            params = pd.DataFrame(opt_params[0], index=[0])
            params_indexed = params.set_index([pd.Series(attack_cat)])
            optimal_params[kind] = pd.concat([optimal_params[kind], params_indexed])

            clf_opt = get_classifier(kind, opt_params[0])
            print('scoring ' + kind + ' ' + attack_cat)
            scores_opt = classify(clf_opt, X, y)
            scores_opt = scores_opt.T
            scores_opt = scores_opt.set_index(
                [pd.Series(attack_cat).repeat(len(scores_opt)), pd.Series(kind).repeat(len(scores_opt)),
                 scores_opt.index.to_series()])
            scores_post = pd.concat([scores_post, scores_opt])

    for kind in kinds:
        optimal_params[kind].to_excel(test_classifiers_dir() + '/' + kind + '_params.xlsx')
        optimal_params[kind].to_pickle(test_classifiers_dir() + '/' + kind + '_params.pkl')
    scores_pre.to_pickle(test_classifiers_dir() + '/' + 'clf_results_pre.pkl')
    scores_pre.to_excel(test_classifiers_dir() + '/' + 'clf_results_pre.xlsx')
    scores_post.to_pickle(test_classifiers_dir() + '/' + 'clf_results_post.pkl')
    scores_post.to_excel(test_classifiers_dir() + '/' + 'clf_results_post.xlsx')


def get_classifier(kind, params=None):
    match kind:
        case 'lr':
            if params is None:
                return LogisticRegression(max_iter=400)
            else:
                return LogisticRegression(C=params['C'], penalty=params['penalty'],
                                          max_iter=params['max_iter'],
                                          solver=params['solver'],
                                          dual=params['dual'], tol=params['tol'])

        case 'svm':
            if params is None:
                return svm.SVC()
            else:
                return svm.SVC(C=params['C'], gamma=params['gamma'],
                               kernel=params['kernel'])

        case 'knn':
            if params is None:
                return KNeighborsClassifier(n_neighbors=3)
            else:
                return KNeighborsClassifier(n_neighbors=params['n_neighbors'])
        case 'dt':
            if params is None:
                return DecisionTreeClassifier()
            else:
                return DecisionTreeClassifier(criterion=params['criterion'],
                                              max_depth=params['max_depth'])

        case _:
            return


def get_params(kind):
    match kind:
        case 'dt':
            return {'criterion': ['gini', 'entropy'],
                    'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]}
        case 'svm':
            return {'C': [0.5, 1, 10, 100], 'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
        case 'lr':
            return {"C": np.logspace(0.1, 3, 30), "penalty": ["l1", "l2"], 'max_iter': [800, 1000, 1200],
                    'solver': ['saga'], 'dual': [False], 'tol': [1e-3]}
        case 'knn':
            k_range = list(range(1, 31))
            return dict(n_neighbors=k_range)
        case _:
            return


def reduce_features(raw_data, test, attack_cat):
    attack_cat_data = prepare_data_for_specific_attack_cat(raw_data, attack_cat, test)
    X, y = remove_target_columns(attack_cat_data)
    clf = get_classifier('dt')
    params = get_params('dt')
    reduce_features_shap(X, y, clf, params)


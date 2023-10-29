import pandas as pd
from numpy import mean, std
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, chi2


def classify(classifier, X, y, kind, attack_cat, after_opt=False):
    print('scoring ' + kind + ' ' + attack_cat)
    scoring = ['accuracy', 'precision', 'recall']
    cv = KFold(n_splits=5,  shuffle=False)
    # scores = cross_val_score(classifier, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    scores = cross_validate(estimator=classifier, X=X, y=y, cv=cv, scoring=scoring, n_jobs=-1)

    # print(kind + ' accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    scores = pd.DataFrame(scores)

    scores['test_F1'] = (2 * scores.test_precision * scores.test_recall) / (scores.test_precision + scores.test_recall)

    if after_opt:
        prefix = 'opt_'
    else:
        prefix = ''
    return scores


def grid_search(classifier, param_grid, X, y, kind, attack_cat):
    print('optimizing ' + kind + ' ' + attack_cat)
    optimal_params = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
    optimal_params.fit(X, y)
    opt = pd.DataFrame(optimal_params.best_params_.items())
    opt.columns = ['param', 'value']
    return optimal_params.best_params_, opt


def reduce_features_by_clf(classifier, X, y, kind, attack_cat):
    print('reducing features ' + kind + ' ' + attack_cat)
    classifier=classifier.fit(X, y)
    cv = KFold(n_splits=10, shuffle=False)
    rfecv = RFECV(estimator=classifier, step=1, cv=cv, scoring='accuracy', min_features_to_select=20, n_jobs=-1)
    rfecv = rfecv.fit(X, y)
    print('Optimal number of features :', rfecv.n_features_)
    print('Best features :', X.columns[rfecv.support_])

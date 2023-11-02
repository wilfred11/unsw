import pandas as pd
# from probatus.feature_elimination import ShapRFECV
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
# from probatus.feature_elimination import ShapRFECV
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, chi2


def classify(classifier, X, y):
    scoring = ['accuracy', 'precision', 'recall']
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    # scores = cross_val_score(classifier, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    scores = cross_validate(estimator=classifier, X=X, y=y, cv=cv, scoring=scoring, n_jobs=-1)
    scores = pd.DataFrame(scores)
    scores['test_F1'] = (2 * scores.test_precision * scores.test_recall) / (scores.test_precision + scores.test_recall)
    return scores


def grid_search(classifier, param_grid, X, y):
    optimal_params = GridSearchCV(classifier, param_grid, cv=5, scoring='f1_micro', verbose=0, n_jobs=-1)
    optimal_params.fit(X, y)
    opt = pd.DataFrame(optimal_params.best_params_.items())
    opt.columns = ['param', 'value']
    return optimal_params.best_params_, opt


def reduce_features_shap(X, y, clf, params):
    """
    #clf.fit(X, y)
    cv = StratifiedKFold(n_splits=10, shuffle=False)
    grid_search = GridSearchCV(clf, param_grid=params, cv=5, scoring='roc_auc', verbose=0, n_jobs=-1)
    shap_rfecv = ShapRFECV(grid_search, step=0.2, cv=cv, scoring='roc_auc', n_jobs=-1)
    shap_rfecv.fit_compute(X,y)

    shap_rfecv.plot()
    shap_rfecv.get_reduced_features_set()
    #print('Optimal number of features :', rfecv.n_features_)
    #print('Best features :', X.columns[rfecv.support_])
    """


def reduce_features_rfecv(classifier, X, y):
    # print('reducing features ' + kind + ' ' + attack_cat)
    #classifier.fit(X, y)
    cv = StratifiedKFold(n_splits=10, shuffle=False)
    rfecv = RFECV(estimator=classifier, step=1, cv=cv, scoring='accuracy', min_features_to_select=20, n_jobs=-1)
    rfecv = rfecv.fit(X, y)
    #print('Optimal number of features :', rfecv.n_features_)
    print('Best features :', X.columns[rfecv.support_])

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import randint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from probatus.feature_elimination import ShapRFECV
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold, StratifiedGroupKFold, \
    StratifiedShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
# from probatus.feature_elimination import ShapRFECV
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, chi2

from functions import test_classifiers_dir, test_classifiers_figs_dir


# Todo https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

def classify(classifier, X, y):
    scoring = ['accuracy', 'precision', 'recall']
    cv = KFold(n_splits=5)
    # scores = cross_val_score(classifier, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # scores = cross_validate(estimator=classifier, X=X, y=y, cv=cv, scoring=scoring, n_jobs=-1)
    scores = cross_validate(estimator=classifier, X=X, y=y, cv=cv,
                            scoring=('precision_macro', 'recall_macro', 'f1_macro'), n_jobs=-1)
    scores = pd.DataFrame(scores)
    # scores['test_F1'] = (2 * scores.test_precision * scores.test_recall) / (scores.test_precision + scores.test_recall)
    return scores


def evaluate_model1(clf, X, y):
    np.random.seed(42)
    X=pd.DataFrame(np.random.normal(size=(10000, 3)))
    y = pd.DataFrame(randint(0, 10, 10000))
    print(y.value_counts())
    k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    predicted_targets = np.array([])
    actual_targets = np.array([])
    # ca =np.zeros((10, 10))
    # https://www.appsloveworld.com/scikit-learn/37/confusion-matrix-and-classification-report-of-stratifiedkfold?expand_article=1
    for train_ix, test_ix in k_fold.split(X, y):
        train_x, test_x = X.loc[X.index[train_ix]], X.loc[X.index[test_ix]]
        train_y, test_y = y.loc[y.index[train_ix]], y.loc[y.index[test_ix]]
        print(test_y.value_counts())
        # classifier = clf.fit(train_x, train_y)
        # predicted_labels = classifier.predict(test_x)

        # print(confusion_matrix(test_y, predicted_labels, labels=clf.classes_))
        # cm = confusion_matrix(test_y, predicted_labels)
        # ca = np.add(ca, cm)

        # predicted_targets = np.append(predicted_targets, predicted_labels)
        # actual_targets = np.append(actual_targets, test_y)

    # disp = ConfusionMatrixDisplay(confusion_matrix=ca, display_labels=clf.classes_)
    # disp.plot()
    # plt.figure(figsize=(15, 15))
    # plt.grid(False)
    # plt.show()
    return 5, 4
    # return predicted_targets, actual_targets


def evaluate_model(clf, X, y):
    k_fold = StratifiedKFold(n_splits=10, shuffle=True)
    predicted_targets = np.array([])
    actual_targets = np.array([])
    ca =np.zeros((y.nunique(), y.nunique()))
    for train_ix, test_ix in k_fold.split(X,y):
        train_x, test_x = X.loc[X.index[train_ix]], X.loc[X.index[test_ix]]
        train_y, test_y = y.loc[y.index[train_ix]], y.loc[y.index[test_ix]]
        print(test_y.value_counts())
        classifier = clf.fit(train_x, train_y)
        predicted_labels = classifier.predict(test_x)
        cm = confusion_matrix(test_y, predicted_labels)
        ca = np.add(ca, cm)

        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, test_y)

    plt.grid(False)
    plt.figure(figsize=(15, 15))
    ax = plt.subplot(1, 1, 1)
    ax.grid(False)
    disp = ConfusionMatrixDisplay(confusion_matrix=ca.astype(int), display_labels=clf.classes_)
    disp.plot(values_format='', ax=ax,  cmap='Blues')
    plt.xticks(rotation=45)
    #plt.grid(False)
    #plt.show()
    plt.savefig(test_classifiers_figs_dir() + '/' + str(y.nunique()) + '-class-confusion-map.png')


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
    # classifier.fit(X, y)
    cv = StratifiedKFold(n_splits=10, shuffle=False)
    rfecv = RFECV(estimator=classifier, step=1, cv=cv, scoring='accuracy', min_features_to_select=20, n_jobs=-1)
    rfecv = rfecv.fit(X, y)
    # print('Optimal number of features :', rfecv.n_features_)
    print('Best features :', X.columns[rfecv.support_])

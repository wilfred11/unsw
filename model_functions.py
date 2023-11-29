import mpu
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import randint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
# from probatus.feature_elimination import ShapRFECV
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.model_selection import GridSearchCV
# from probatus.feature_elimination import ShapRFECV
from sklearn.feature_selection import RFECV
from functions import test_classifiers_dir, test_classifiers_figs_dir, external_data_dir, feature_reduction_dir


# Todo https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

def evaluate_model_scoring(classifier, X, y, cv):
    scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    scores = cross_validate(estimator=classifier, X=X, y=y, cv=cv,
                            scoring=scoring, n_jobs=-1)
    scores = pd.DataFrame(scores)

    # scores['test_F1'] = (2 * scores.test_precision * scores.test_recall) / (scores.test_precision + scores.test_recall)
    return scores


def evaluate_model_cm(clf, X, y, cv):
    predicted_labels_accum = np.array([])
    actual_labels_accum = np.array([])
    score = np.zeros(cv.get_n_splits([X, y]))
    i = 0
    ca = np.zeros((y.nunique(), y.nunique()))
    for train_ix, test_ix in cv.split(X, y):
        train_x, test_x = X.loc[X.index[train_ix]], X.loc[X.index[test_ix]]
        train_y, test_y = y.loc[y.index[train_ix]], y.loc[y.index[test_ix]]
        #print(test_y.value_counts())
        classifier = clf.fit(train_x, train_y)
        predicted_labels = classifier.predict(test_x)
        cm = confusion_matrix(test_y, predicted_labels)
        ca = np.add(ca, cm)
        predicted_labels_accum = np.append(predicted_labels_accum, predicted_labels)
        actual_labels_accum = np.append(actual_labels_accum, test_y)
        score[i] = accuracy_score(test_y, predicted_labels)
        i = i + 1

    score_dict = dict(enumerate(score))

    score_dict['metric'] = 'accuracy'
    score_dict['type'] = 'balanced'
    scores=pd.DataFrame(score_dict, index=[0])
    scores.set_index(['type', 'metric'], inplace=True)
    print(scores)
    scores.to_excel(test_classifiers_dir() + '/' + 'cv_acc_score_.xlsx')
    print(classification_report(actual_labels_accum, predicted_labels_accum, target_names=clf.classes_))
    cr = classification_report(actual_labels_accum, predicted_labels_accum, target_names=clf.classes_, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    cr_df.to_excel(test_classifiers_dir() + '/' + 'clf_cr.xlsx')
    # print(df)
    plt.grid(False)
    plt.figure(figsize=(15, 15))
    ax = plt.subplot(1, 1, 1)
    ax.grid(False)
    disp = ConfusionMatrixDisplay(confusion_matrix=ca.astype(int), display_labels=clf.classes_)
    disp.plot(values_format='', ax=ax, cmap='Blues')
    plt.xticks(rotation=45)
    # plt.grid(False)
    # plt.show()
    plt.savefig(test_classifiers_figs_dir() + '/' + str(y.nunique()) + '-class-confusion-map.png')
    mpu.io.write(feature_reduction_dir() + '/' + 'conf-mat-agg.pickle', ca)


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

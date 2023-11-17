import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpu
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif

from functions import test_classifiers_dir, feature_reduction_dir, features_to_be_denominalized, external_data_dir
from model_functions import classify, grid_search, reduce_features_rfecv
from prepare_data import prepare_data_for_specific_attack_cat, remove_target_columns


# https://www.rasgoml.com/feature-engineering-tutorials/feature-selection-using-mutual-information-in-scikit-learn

def test_classifiers(raw_data, test, kinds, size):
    cats = list(raw_data.attack_cat.unique())
    scores_pre = pd.DataFrame()
    scores_post = pd.DataFrame()
    optimal_params = {}
    for kind in kinds:
        optimal_params[kind] = pd.DataFrame()

    print(optimal_params)
    for attack_cat in cats:
        print('**************************')
        attack_cat_data = prepare_data_for_specific_attack_cat(raw_data, attack_cat, size)
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
        case 'lasso':
            if params is None:
                return Lasso()
            else:
                return Lasso(params)
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
        case 'lasso':
            return {'alpha': (np.logspace(-8, 8, 100))}
        case _:
            return


def create_column_name(row):
    if len(row['feat_agg']) == 1:
        return row['feat_agg'][0]
    elif len(row['feat_agg']) == 2:
        return row['feat_agg'][0] + '--' + row['feat_agg'][1]
    elif len(row['feat_agg']) > 2:
        return row['feat_agg'][0] + '--' + row['feat_agg'][1] + '...'


def handle_categorical_data(raw_data):
    for feature in features_to_be_denominalized():
        ct = pd.crosstab(raw_data[feature], raw_data['attack_cat'], normalize='index').round(2)
        with open(feature_reduction_dir() + '/' + feature + "-attack-ct.txt", "w") as text_file:
            text_file.write(ct.to_latex())
        ct.to_excel(feature_reduction_dir() + '/' + feature + '-attack-ct.xlsx')
        ct = ct.multiply(10)
        ct = ct.apply(np.ceil)
        ct.to_excel(feature_reduction_dir() + '/' + feature + '-attack-ct-mp.xlsx')
        l = ct.columns.to_list()
        ct = ct.reset_index()
        result = ct.groupby(l, as_index=False).agg({feature: lambda x: list(x)})
        result.to_excel(feature_reduction_dir() + '/' + feature + '-attack-gb.xlsx')
        df = pd.DataFrame()
        df = pd.DataFrame({'feat': result[feature], 'feat_agg': result[feature]})
        df = df.explode('feat')
        df['column_name'] = df.apply(lambda x: create_column_name(x), axis=1)
        df.to_excel(feature_reduction_dir() + '/' + feature + '-attack-aggregated.xlsx')
        feat_dict = dict(zip(df.feat, df.column_name))
        mpu.io.write(feature_reduction_dir() + '/' + feature + '_cat_dict.pickle', feat_dict)


def reduce_categories(raw_data):
    for feature in features_to_be_denominalized():
        feat_dict = mpu.io.read(feature_reduction_dir() + '/' + feature + '_cat_dict.pickle')
        raw_data = raw_data.replace({feature: feat_dict})
    return raw_data


def reduce_features(raw_data, attack_cat):
    attack_cat_data = prepare_data_for_specific_attack_cat(raw_data, attack_cat, 1000)
    X, y = remove_target_columns(attack_cat_data)
    # clf = get_classifier('dt')
    # params = get_params('dt')
    # reduce_features_shap(X, y, clf, params)
    # clf = get_classifier('svm')
    lr_params = read_params('dt')
    lr_params.index.name = 'attack_cat'
    # print(svm_params)
    params = lr_params.query("attack_cat =='" + attack_cat + "'")
    print(params.to_dict('records'))
    classifier = get_classifier('dt', params.to_dict('records')[0])
    # print(params)
    reduce_features_rfecv(classifier, X, y)


def condition(x):
    return x > 0.0000001 or x < -0.0000001


def reduce_features_lasso(raw_data):
    print(raw_data.shape)
    """
    https://www.datacamp.com/tutorial/tutorial-lasso-ridge-regression?utm_source=google&utm_medium=paid_search&utm_campaignid=19589720818&utm_adgroupid=157156373751&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=676354848902&utm_targetid=dsa-2218886984100&utm_loc_interest_ms=&utm_loc_physical_ms=1001071&utm_content=&utm_campaign=230119_1-sea~dsa~tofu_2-b2c_3-eu_4-prc_5-na_6-na_7-le_8-pdsh-go_9-na_10-na_11-na-oct23&gclid=CjwKCAjw7oeqBhBwEiwALyHLM6I2VIxuXzBANx3jIYuJcKTrj0bCip6PCFS0GDmdnnftoJCZyGrINBoC13MQAvD_BwE
    https://www.shedloadofcode.com/blog/eight-ways-to-perform-feature-selection-with-scikit-learn
    """

    cats = list(raw_data.attack_cat.unique())
    features_to_be_removed = pd.DataFrame({'attack_cat': [], 'features_to_be_removed': []})
    common_unselected_features = raw_data.columns.to_list()
    for attack_cat in cats:
        attack_cat_data = prepare_data_for_specific_attack_cat(raw_data, attack_cat, 500000)
        X, y = remove_target_columns(attack_cat_data)

        lasso = Lasso(alpha=0.0001, max_iter=5000)
        lasso.fit(X, y)

        # Get the non-zero feature coefficients
        nonzero_coefs = lasso.coef_
        print(nonzero_coefs)
        print(len(nonzero_coefs))
        # selected_indices = nonzero_coefs != 0
        selected_indices = [idx for idx, element in enumerate(nonzero_coefs) if condition(element)]
        unselected_indices = [idx for idx, element in enumerate(nonzero_coefs) if not condition(element)]
        print(selected_indices)
        # print(unselected_indices)
        selected_features = X.columns[selected_indices]
        unselected_features = X.columns[unselected_indices]

        common_unselected_features = list(set(unselected_features).intersection(common_unselected_features))
        new_row = {'attack_cat': attack_cat,
                   'features_to_be_removed': unselected_features.to_list()
                   }

        features_to_be_removed = pd.concat([features_to_be_removed, pd.DataFrame([new_row])], ignore_index=True)

        nonzero_coefs = nonzero_coefs[selected_indices]

        # Plot the feature coefficients
        plt.figure(figsize=(70, 100))
        plt.barh(range(len(nonzero_coefs)), nonzero_coefs, tick_label=selected_features)
        plt.xlabel('Coefficient Values')
        plt.ylabel('Features')
        plt.title('L1 Regularisation (Lasso): Feature Coefficients')
        # plt.show()
        plt.savefig(feature_reduction_dir() + '/figs/feature-coef.png')
        print("Selected Features:")
        print(selected_features)
    print(features_to_be_removed)
    print(common_unselected_features)
    features_to_be_removed.to_csv(feature_reduction_dir() + '/' + 'features_to_be_removed_lasso.csv', index=False)
    cuf = pd.DataFrame(common_unselected_features, columns=['unselected features'])
    cuf.to_csv(feature_reduction_dir() + '/' + 'common_unselected_features.csv', index=False)


def train_reduce_test(raw_data, kinds, cats, size):
    #unused
    scores_pre = pd.DataFrame()
    scores_post = pd.DataFrame()
    optimal_params = {}
    for kind in kinds:
        optimal_params[kind] = pd.DataFrame()

        for attack_cat in cats:
            attack_cat_data = prepare_data_for_specific_attack_cat(raw_data, attack_cat, 1000)
            X, y = remove_target_columns(attack_cat_data)
            classifier = get_classifier(kind, params=None)
            train_X, test_X, train_y, test_y = train_test_split(X, y, stratify=True, random_state=0)
            classifier.fit(train_X, train_y)
            classifier.predict(test_X)


def generate_accuracy_and_heatmap(model, x, y):
    cm = confusion_matrix(y, model.predict(x))
    sns.heatmap(cm, annot=True, fmt="d")
    ac = accuracy_score(y, model.predict(x))
    f_score = f1_score(y, model.predict(x))
    print('Accuracy is: ', ac)
    print('F1 score is: ', f_score)
    print("\n")
    return 1


def read_params(kind):
    results = pd.read_pickle(test_classifiers_dir() + "/" + kind + '_params.pkl', compression='infer')
    return results

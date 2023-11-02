import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from functions import test_classifiers_dir, figures_dir


def read_results():
    results = pd.read_pickle(test_classifiers_dir() + "/" + 'clf_results_post.pkl', compression='infer')
    results.index.names = ['attack_cat', 'clf', 'score']
    return results


def create_results_plot_all():
    results = read_results()
    print(results.index.get_level_values('attack_cat').unique().to_list())
    for attack_cat in results.index.get_level_values('attack_cat').unique().to_list():
        create_results_plot(results, attack_cat)


def create_results_plot(results, attack_cat):
    results = results.query("attack_cat =='" + attack_cat + "'")
    results = results.query("score == 'test_F1'")
    results = results.droplevel(0, axis=0)
    results = results.droplevel(1, axis=0)
    results = results.T
    plt.figure(figsize=(12, 7))
    plt.xlabel('testrun#')
    plt.ylabel('F1 score')
    plt.xticks(results.index.to_list())
    sns.lineplot(data=results).set_title(
        'Resultaten Normal<>' + (attack_cat if attack_cat != 'Normal' else 'alle aanvallen'))
    plt.legend(title='')
    plt.savefig(figures_dir() + '/' + attack_cat + '_results.png')


def correlated_features(raw_data):
    corr_mat = raw_data.corr(method='pearson')
    plt.figure(figsize=(25, 25))
    sns.heatmap(corr_mat, square=True, annot=True, annot_kws={'size': 8})
    plt.savefig(figures_dir() + '/heatmap_features.png')

    columns = corr_mat.columns
    for i in range(corr_mat.shape[0]):
        for j in range(i + 1, corr_mat.shape[0]):
            if corr_mat.iloc[i, j] >= 0.9:
                print(f"{columns[i]:20s} {columns[j]:20s} {corr_mat.iloc[i, j]}")
    #plt.show()
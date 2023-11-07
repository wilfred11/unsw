import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from functions import test_classifiers_dir, figures_dir, feature_reduction_dir
from prepare_data import prepare_data_for_specific_attack_cat
from math import ceil


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
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_mat, square=True, annot=True, annot_kws={'fontsize': 6}, fmt='.2f')
    plt.savefig(feature_reduction_dir() + "/figs/feature_heatmap_hidpi.png", dpi=200)

    columns = corr_mat.columns
    for i in range(corr_mat.shape[0]):
        for j in range(i + 1, corr_mat.shape[0]):
            if corr_mat.iloc[i, j] >= 0.9 or corr_mat.iloc[i, j] <= -0.9:
                print(f"{columns[i]:20s} {columns[j]:20s} {corr_mat.iloc[i, j]}")


def pairplot(raw_data, attack_cat, size):
    """
    https://www.datacamp.com/tutorial/tutorial-lasso-ridge-regression?utm_source=google&utm_medium=paid_search&utm_campaignid=19589720818&utm_adgroupid=157156373751&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=676354848902&utm_targetid=dsa-2218886984100&utm_loc_interest_ms=&utm_loc_physical_ms=1001071&utm_content=&utm_campaign=230119_1-sea~dsa~tofu_2-b2c_3-eu_4-prc_5-na_6-na_7-le_8-pdsh-go_9-na_10-na_11-na-oct23&gclid=CjwKCAjw7oeqBhBwEiwALyHLM6I2VIxuXzBANx3jIYuJcKTrj0bCip6PCFS0GDmdnnftoJCZyGrINBoC13MQAvD_BwE
    https://www.shedloadofcode.com/blog/eight-ways-to-perform-feature-selection-with-scikit-learn
    """
    attack_cat_data = prepare_data_for_specific_attack_cat(raw_data, attack_cat, size)
    # attack_cat_data.drop(['Label'], inplace=True, axis=1)
    attack_cat_data.drop(['attack_cat'], inplace=True, axis=1)
    cols = attack_cat_data.columns.to_list()
    print(range(ceil(len(attack_cat_data.columns) / 4)))
    for i in range(ceil(len(attack_cat_data.columns) / 4)):
        if len(cols) == 5:
            cols_to_be_shown = cols[:5]
        elif len(cols) >= 4:
            cols_to_be_shown = cols[:4]
        elif len(cols) < 4 and len(cols) > 0:
            cols_to_be_shown = cols
        elif len(cols) == 0:
            return

        cols = [item for item in cols if item not in cols_to_be_shown]
        sns.pairplot(attack_cat_data[cols_to_be_shown])
        plt.savefig(feature_reduction_dir() + '/figs/columns_pairplot_' + str(i) + '.png')

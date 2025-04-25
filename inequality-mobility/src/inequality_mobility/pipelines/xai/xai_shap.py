import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
import seaborn as sns
import shap

PATH_FIG = "/home/yoshraf/projects/master-analysis-inequality-mobility/inequality-mobility/docs/plots/xai/"


def create_tree_shap(X, tree):
    explainer_tree = shap.Explainer(tree)
    return explainer_tree(X)


def create_agno_shap(X, model):
    explainer = shap.Explainer(model.predict_proba, X)
    return explainer(X)


def return_main_features(shap_v, y_class):
    dict_important = {}
    for fe in shap_v.feature_names:
        dict_important[fe] = abs(shap_v[:, fe, y_class].values).mean()
    df = pd.DataFrame(
        dict_important.items(), columns=['features', 'shap_values']
    ).sort_values(by='shap_values', ascending=False)
    return df


def return_df_barplot(shap_v, map_class, nfeatures=10):
    df = pd.concat(
        [return_main_features(shap_v, v).rename(
            columns={'features': 'features', 'shap_values': f'shap_values_{k}'}
        ) for k, v in map_class.items()], axis=1)
    # Drop duplicates columns
    df = df.loc[:, ~df.columns.duplicated()].copy()
    # Total
    df = df.assign(total=df.iloc[:, 1:].sum(axis=1))
    df = df.reset_index().drop(
        columns="index").set_index("features")
    return df.sort_values('total').tail(nfeatures)


def plot_main_features(df_barplot, map_class, normalize=None, save_plot=None):
    # Setting
    fields = list(df_barplot.columns)[:-1]
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 10))
    # plot bars
    left = len(df_barplot) * [0]
    if normalize:
        for name in fields:
            plt.barh(df_barplot.index,
                     df_barplot[name] / df_barplot["total"], left=left)
            left = left + df_barplot[name] / df_barplot["total"]
    else:
        for name in fields:
            plt.barh(df_barplot.index, df_barplot[name], left=left)
            left = left + df_barplot[name]

    # Set title, legend, labels
    plt.title('Principais Variáveis por Modo\n', loc='left')
    plt.legend(map_class.keys(), bbox_to_anchor=(
        [0.55, 1, 0, 0]), ncol=4, frameon=False)
    plt.xlabel('Média do módulo do SHAP Value')
    # Remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # Adjust limits and draw grid lines
    plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray', linestyle='dashed')
    # Save plot
    if save_plot:
        PATH_FIG_FI = PATH_FIG + "shap_feature_importance/"
        if normalize:
            plt.savefig(
                PATH_FIG_FI + f"fi_normalize_{save_plot}",  dpi=fig.dpi, bbox_inches='tight')
        else:
            plt.savefig(PATH_FIG_FI + f"fi_{save_plot}",
                        dpi=fig.dpi, bbox_inches='tight')
    plt.show()


def plot_several_beeswarm(shap_v, map_class, save_plot=None):
    for mod, mod_id in map_class.items():
        print(mod)
        fig, ax = plt.subplots(1)
        shap.plots.beeswarm(shap_v[:, :, mod_id], max_display=6, show=False)
        ax.set_title(mod)
        ax.set_xlabel('Shap Value')
        # Save plot
        if save_plot:
            PATH_FIG_BEE = PATH_FIG + "shap_beeswarm/"
            mode_save = mod.replace('/','')
            plt.savefig(PATH_FIG_BEE +
                        f"bee_{mode_save}_{save_plot}",  dpi=fig.dpi, bbox_inches='tight')
        plt.show()


def get_main_features(df_bar, y_class, n=5):
    return (df_bar[f"shap_values_{y_class}"].sort_values(ascending=False) * 100)[:n].index


def plot_scatter(df_bar, shap_v, map_class, save_plot=None):
    for mode, mode_idx in map_class.items():
        print(mode)
        top5 = get_main_features(df_bar, mode)
        for c in top5:
            z = 0.8
            fig, ax = plt.subplots(1, figsize=(15 * z, 8 * z))
            shap.plots.scatter(shap_v[:, c, mode_idx], ax=ax, title=mode,
                               show=False, hist=False)
            if save_plot:
                PATH_FIG_SCT = PATH_FIG + "shap_scatter/"
                mode_save = mode.replace('/','')
                plt.savefig(PATH_FIG_SCT +
                            f"scatter_{mode_save}_{c}_{save_plot}",  dpi=fig.dpi, bbox_inches='tight')
            plt.show()

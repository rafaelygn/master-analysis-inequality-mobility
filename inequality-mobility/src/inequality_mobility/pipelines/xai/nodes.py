import pandas as pd
from matplotlib import pyplot as plt

from .xai_pdp import LogOddsEstimator, generate_pdp_display, plot_pdp_main
from .xai_shap import (create_agno_shap, create_tree_shap, get_main_features,
                       plot_main_features, plot_scatter, plot_several_beeswarm,
                       return_df_barplot)


def node_create_shap(X, clf_rf, clf_logit, samples=500):
    clf_rf, _, _ = clf_rf
    # clf_logit, _, _ = clf_logit
    df_sample = X.sample(samples, random_state=42)
    print('Tree')
    shap_v_rf = create_tree_shap(df_sample, clf_rf)
    print('Logit')
    shap_v_lg = create_agno_shap(df_sample, clf_logit)
    return shap_v_rf, shap_v_lg, df_sample


def shap_analysis(shap_v, map_class, save_plot):
    df_bar = return_df_barplot(shap_v, map_class, 10)
    plot_main_features(
        df_bar,
        map_class, save_plot=save_plot)
    # Normalized
    plot_main_features(
        df_bar,
        map_class,
        True, save_plot)
    # Beeswarm
    plot_several_beeswarm(shap_v, map_class, save_plot)
    # Scatter
    plot_scatter(df_bar, shap_v, map_class, save_plot)
    return None


def node_feature_importance(shap_v_rf, shap_v_lg, map_class):
    # ---------
    # RF
    # ---------
    shap_analysis(shap_v_rf, map_class, 'RF_OD23')
    # ---------
    # Logit
    # ---------
    shap_analysis(shap_v_lg, map_class, 'logit_OD23')
    return None


def node_pdp(clf_rf, clf_logit, shap_v, map_class):
    clf_rf = clf_rf[0]
    clf_rf_odds = LogOddsEstimator(clf_rf)
    clf_logit_odds = LogOddsEstimator(clf_logit)
    X = pd.DataFrame(shap_v.data, columns=shap_v.feature_names) 
    inv_map = {v: k for k, v in map_class.items()}
    # Get Top 5
    df_bar = return_df_barplot(shap_v, map_class, 10)
    for mode, mode_idx in map_class.items():
        print(mode)
        top5 = list(get_main_features(df_bar, mode))
        display_rf, display_logit, display_rf_odds, display_logit_odds = generate_pdp_display(
            clf_rf, clf_rf_odds, clf_logit, clf_logit_odds, X, top5, mode_idx
        )
        # Dummy Plot
        fig, (ax1, ax2)  = plt.subplots(2, len(top5))
        display_rf.plot(ax=ax1, line_kw={"label": "Random Forest"})
        display_logit.plot(ax=ax1, line_kw={"label": "Multinomial Logit", "color": "red"})
        # Real Plot
        plot_pdp_main(
            top5, display_rf, display_rf_odds, display_logit, clf_logit_odds, X, 
            mode_idx, inv_map, 
            # dsp_lg_odds=display_logit_odds
        )

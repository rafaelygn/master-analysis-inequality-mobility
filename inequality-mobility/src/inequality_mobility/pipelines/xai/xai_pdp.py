'''
This script is based on 
https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_partial_dependence_visualization_api.html#sphx-glr-auto-examples-miscellaneous-plot-partial-dependence-visualization-api-py
'''

import numpy as np
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay


from sklearn.base import BaseEstimator

PATH_FIG = "/home/yoshraf/projects/master-analysis-inequality-mobility/inequality-mobility/docs/plots/xai/"

class LogOddsEstimator(BaseEstimator):

    def __init__(self, estimator):
        self._estimator_type = "classifier"
        self.estimator = estimator
        self.classes_ = estimator.classes_

    def fit(self, X):
        pass

    def __sklearn_is_fitted__(self):
        return True

    def predict_proba(self, X):
        '''
        Actually, this function returns a log odds, not a probability. 

        However, the PartialDependenceDisplay requires the name function 'predict_proba'. 
        To avoid division by zero, we have added 1%."
        '''
        return np.log((self.estimator.predict_proba(X) + 1e-2) / (1 - self.estimator.predict_proba(X) + 1e-2))

    def aux_plot_function_logit(self, X, y_class, feature, p=(.0, .9)):
        self.coefs = self.estimator.coef_[y_class]
        self.coefs = dict(zip(list(X.columns), self.coefs))

        y_mean = self.predict_proba(X)[:, y_class].mean()
        x_mean = X[feature].mean()
        x = X[feature].sort_values()
        x = x[int(x.shape[0] * p[0]):int(x.shape[0] * p[1])]
        def y(x): return self.coefs[feature] * x + \
            (y_mean - self.coefs[feature] * x_mean)
        return x, y, x_mean, y_mean


def generate_pdp_display(clf_rf, clf_rf_odds, clf_logit, clf_logit_odds, X, top_features, y_class):

    display_rf_odds = PartialDependenceDisplay.from_estimator(
        clf_rf_odds,
        X,
        features=top_features,
        target=y_class,
        response_method='predict_proba',
        kind='average',
        percentiles=(.1, .9),
        n_cols=len(top_features)
    )

    display_logit = PartialDependenceDisplay.from_estimator(
        clf_logit,
        X,
        features=top_features,
        target=y_class,
        response_method='predict_proba',
        kind='average',
        percentiles=(.1, .9),
        n_cols=len(top_features)
    )

    display_rf = PartialDependenceDisplay.from_estimator(
        clf_rf,
        X,
        features=top_features,
        target=y_class,
        response_method='predict_proba',
        kind='average',
        percentiles=(.1, .9),
        n_cols=len(top_features)
    )

    display_logit_odds = PartialDependenceDisplay.from_estimator(
        clf_logit_odds,
        X,
        features=top_features,
        target=y_class,
        response_method='predict_proba',
        kind='average',
        percentiles=(.1, .9),
        n_cols=len(top_features)
    )
    return display_rf, display_logit, display_rf_odds, display_logit_odds


def plot_pdp_main(features, dsp_ml, dsp_ml_logodds, dsp_lg, logit, X, y_class, inv_map, dsp_lg_odds=None, f=1.4, f_lim=1.00, prop=(16, 6)):
    n_feats = len(features)

    fig, (ax1, ax2) = plt.subplots(2, n_feats, figsize=(prop[0] * f, prop[1] * f))

    dsp_ml.plot(ax=ax1, line_kw={"label": "Random Forest"})
    dsp_lg.plot(ax=ax1, line_kw={"label": "Multinomial Logit", "color": "red"})
    dsp_ml_logodds.plot(ax=ax2, line_kw={"label": "Random Forest"})

    if dsp_lg_odds != None:
        dsp_lg_odds.plot(ax=ax2, line_kw={
                         "label": "Multinomial Logit (Real)", "color": "pink"})


    ylim = dsp_ml_logodds.axes_[0].get_ylim()
    for i, ax in enumerate(ax2):
        x0, y0, x_mean0, y_mean0 = logit.aux_plot_function_logit(
            X, y_class, features[i], p=(0, .9))
        # Ajust ylim
        ylim = min(y0(x0).min() * f_lim,  ylim[0]), max(y0(x0).max() * f_lim, ylim[1])
        ax.plot(x0, y0(x0), color='red', label="Multinomial Logit")
        ax.axhline(y_mean0, linestyle='--', color='grey', alpha=.3)
        ax.axvline(x_mean0, linestyle='--', color='grey', alpha=.3)
        ax.legend()
        # Update xlim ax2
        ax.set_xlim(x0.min(), x0.max())
        ax1[i].set_xlim(x0.min(), x0.max())

    # Update ylim ax1
    for i, ax in enumerate(ax1):
        ylim_ml = dsp_ml.axes_[0].get_ylim()
        ylim_logit = dsp_lg.axes_[0].get_ylim()
        ax.set_ylim(
            min(ylim_ml[0] * f_lim, ylim_logit[0] * f_lim),
            max(ylim_ml[1] * f_lim, ylim_logit[1] * f_lim)
        )
        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel("Partial dependence\n(Probability)")
        else:
            ax.set_ylabel("")

    # Update ylim ax2
    for i, ax in enumerate(ax2):
        ax.set_ylim(ylim)
        if i == 0:
            ax.set_ylabel("Partial dependence\n(Logodds)")
        else:
            ax.set_ylabel("")

    fig.suptitle(inv_map[y_class], fontsize=16)
    PATH_FIG_PDP = PATH_FIG + "pdp/"
    mode_save = inv_map[y_class].replace('/','')
    plt.savefig(PATH_FIG_PDP +
                f"pdp_{mode_save}",  dpi=fig.dpi, bbox_inches='tight')

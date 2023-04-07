from kedro.pipeline import Pipeline, node

from .nodes import node_create_shap, node_feature_importance, node_pdp


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=node_create_shap,
                inputs=["X_train_dc", "clf_rf", "clf_logit"],
                outputs=["shap_v_rf", "shap_v_logit"],
                name="create_shap",
                tags="create_shap",
            ),
            node(
                func=node_feature_importance,
                inputs=["shap_v_rf", "shap_v_logit", "params:modeling.map_class"],
                outputs=None,
                name="feature_importance",
                tags="feature_importance",
            ),
            node(
                func=node_pdp,
                inputs=["clf_rf", "clf_logit", "shap_v_rf", "params:modeling.map_class"],
                outputs=None,
                name="pdp",
                tags="pdp",
            ),
        ]
    )

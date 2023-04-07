from kedro.pipeline import Pipeline, node

from .nodes import node_logit, node_lgbm, node_tree, node_rf, node_svm, node_xgb, node_ann


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=node_logit,
                inputs=["X_train_dc", "X_test_dc", "y_train_dc",
                        "y_test_dc", "params:modeling.map_class"],
                outputs="clf_logit",
                name="node_logit",
                tags="node_logit",
            ),
            node(
                func=node_lgbm,
                inputs=["X_train_dc", "X_test_dc", "y_train_dc",
                        "y_test_dc", "params:modeling.map_class"],
                outputs="clf_lgbm",
                name="node_lgbm",
                tags="node_lgbm",
            ),
            node(
                func=node_tree,
                inputs=["X_train_dc", "X_test_dc", "y_train_dc",
                        "y_test_dc", "params:modeling.map_class"],
                outputs="clf_tree",
                name="node_tree",
                tags="node_tree",
            ),
            node(
                func=node_rf,
                inputs=["X_train_dc", "X_test_dc", "y_train_dc",
                        "y_test_dc", "params:modeling.map_class"],
                outputs="clf_rf",
                name="node_rf",
                tags="node_rf",
            ),
            node(
                func=node_svm,
                inputs=["X_train_dc", "X_test_dc", "y_train_dc",
                        "y_test_dc", "params:modeling.map_class"],
                outputs="clf_svm",
                name="node_svm",
                tags="node_svm",
            ),
            node(
                func=node_xgb,
                inputs=["X_train_dc", "X_test_dc", "y_train_dc",
                        "y_test_dc", "params:modeling.map_class"],
                outputs="clf_xgb",
                name="node_xgb",
                tags="node_xgb",
            ),
            node(
                func=node_ann,
                inputs=["X_train_dc", "X_test_dc", "y_train_dc",
                        "y_test_dc", "params:modeling.map_class"],
                outputs="clf_ann",
                name="node_ann",
                tags="node_ann",
            ),
        ]
    )

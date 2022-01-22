from kedro.pipeline import Pipeline, node

from .nodes import node_create_dis_cho

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=node_create_dis_cho,
                inputs=["dataset_final", "params:model_input.discrete_choice"],
                outputs=["X_train_dc", "X_test_dc", "y_train_dc", "y_test_dc"],
                name="model_in_dc",
                tags="model_in_dc",
            ),
        ]
    )
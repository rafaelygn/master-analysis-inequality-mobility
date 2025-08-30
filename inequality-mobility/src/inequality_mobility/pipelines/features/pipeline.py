from kedro.pipeline import Pipeline, node

from .nodes import (
    # node_features_gis_ilumina,
    node_features_gis_eng,
    # node_features_gis_cota,
    node_features_others,
    # node_features_acc,
    node_diff_07,
    node_diff_17,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # node(
            #     func=node_features_gis_ilumina,
            #     inputs=["od2017_filtered", "ilumina"],
            #     outputs="dataset_ilumina",
            #     name="features_ilumina",
            #     tags="features_ilumina"
            # ),
            node(
                func=node_features_gis_eng,
                inputs=["od2023_filtered", "metro_final", "trem",
                        "terminal", "ciclovia"],
                outputs="dataset_eng",
                name="features_eng",
                tags="features_eng",
            ),
            # node(
            #     func=node_features_gis_cota,
            #     inputs=["dataset_eng", "ponto_cotado"],
            #     outputs="dataset_cota",
            #     name="features_cota",
            #     tags="features_cota"
            # ),
            node(
                func=node_features_others,
                inputs=["dataset_eng", "params:features.od2023_socio"],
                outputs="dataset_final",
                name="features_others",
                tags="features_others"
            ),
            # node(
            #     func=node_features_acc,
            #     inputs=["dataset_socio", "acc_joined"],
            #     outputs="dataset_final",
            #     name="features_acc",
            #     tags="features_acc"
            # ),
            # node(
            #     func=node_diff_07,
            #     inputs=["od2007_filtered", "metro",  "parameters"],
            #     outputs="dataset_07_diff",
            #     name="node_diff_07",
            #     tags="node_diff_07"
            # ),
            # node(
            #     func=node_diff_17,
            #     inputs=["od2017_filtered", "metro",  "parameters"],
            #     outputs="dataset_17_diff",
            #     name="node_diff_17",
            #     tags="node_diff_17"
            # ),

        ]
    )

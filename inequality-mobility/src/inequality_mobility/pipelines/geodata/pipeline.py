from kedro.pipeline import Pipeline, node

from .nodes import node_read_gis, node_prep_od2023, node_prep_od2017, node_read_prep_od2007, node_prep_metro

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=node_read_gis,
                inputs="parameters",
                outputs=[
                    "od2023", "od2017",
                    "metro", "trem", "ciclovia", 
                    # "ilumina",
                    # "ponto_onibus",
                    # "ponto_cotado",
                    "terminal",
                    # "acc_emp_ti",
                    # "acc_emp_tp",
                    # "acc_laz_ti",
                    # "acc_laz_tp",
                    # "censo_demo"
                    ],
                name="convert_to_geojson",
                tags="convert_to_geojson"
            ),
            node(
                func=node_prep_metro,
                inputs=["metro", "inauguracao", "parameters"],
                outputs="metro_final",
                name="prep_metro",
                tags="prep_metro",
            ),
            node(
                func=node_prep_od2023,
                inputs=["od2023", "parameters"],
                outputs="od2023_filtered",
                name="prep_od2023",
                tags="prep_od2023",
            ),
            node(
                func=node_prep_od2017,
                inputs=["od2017", "parameters"],
                outputs="od2017_filtered",
                name="prep_od2017",
                tags="prep_od2017",
            ),
            node(
                func=node_read_prep_od2007,
                inputs="parameters",
                outputs="od2007_filtered",
                name="read_prep_od2007",
                tags="read_prep_od2007",
            ),
            # node(
            #     func=node_join_access,
            #     inputs=["acc_emp_ti", "acc_emp_tp", "acc_laz_ti", "acc_laz_tp"],
            #     outputs="acc_joined",
            #     name="node_join_access",
            #     tags="node_join_access",
            # ),
            # node(
            #     func=node_prep_censo,
            #     inputs=["censo_demo", "params:features.censo_demo"],
            #     outputs="censo_demo_final",
            #     name="node_prep_censo",
            #     tags="node_prep_censo",
            # ),
        ]
    )
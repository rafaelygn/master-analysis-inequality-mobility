from typing import Dict, Tuple, Any

import geopandas as gpd
import pandas as pd
import kedro
from shapely.ops import unary_union
from datetime import datetime


from .utils import DICT_GIS_17, create_gis_point, get_counting, geo_prep_to_join
from .cota_knn import predict_knn


# ------------------------------------
# Aux Functions
# ------------------------------------

def prep_ilumina(gdf_ilumin: gpd.GeoDataFrame, sampling: int = 0) -> Tuple:
    # Sampling
    if sampling:
        gdf_ilumin = gdf_ilumin.sample(frac=sampling, random_state=42)
    # Filtro por tipo
    gdf_ilumin_led = gdf_ilumin[gdf_ilumin["il_tipilum"] == "LED"]
    gdf_ilumin_std = gdf_ilumin[gdf_ilumin["il_tipilum"] == "VAPOR DE SODIO"]
    print(f"Loading light pole sample. Rows loaded: {gdf_ilumin_led.shape[0]}")
    print(f"Loading std pole sample. Rows loaded: {gdf_ilumin_std.shape[0]}")
    return gdf_ilumin_led, gdf_ilumin_std


def prep_ciclo(gdf_ciclo: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf_ciclo["rc_inaugur"] = pd.to_datetime(gdf_ciclo["rc_inaugur"])
    gdf_ciclo = gdf_ciclo[gdf_ciclo["rc_inaugur"] <= datetime(2018, 1, 1)]
    v_bools = (gdf_ciclo["rc_ext_t"] >= 3000) | (gdf_ciclo["rc_ext_c"] >= 3000)
    gdf_ciclo = gdf_ciclo[v_bools]
    return gdf_ciclo


def prep_subway(gdf_metro: gpd.GeoDataFrame, gdf_metro_inau: pd.DataFrame) -> gpd.GeoDataFrame:
    map_lines = {
        1: "AZUL",
        2: "VERDE",
        3: "VERMELHA",
        4: "AMARELA",
        5: "LILAS",
        15: "PRATA"
    }
    gdf_metro_inau["Inauguração"] = pd.to_datetime(gdf_metro_inau["Inauguração"])
    gdf_metro_inau["Upper"] = gdf_metro_inau["Nome"].str.upper()
    gdf_metro_inau_inter = gdf_metro_inau[(gdf_metro_inau["Inauguração"].dt.year < 2017)]
    gdf_metro_inau_inter["Linha_Descr"] = gdf_metro_inau_inter["Linha"].map(map_lines)
    df_metro_inau = pd.merge(gdf_metro_inau_inter, gdf_metro, 
        left_on=["Upper", "Linha_Descr"],
        right_on=["emt_nome", "emt_linha"],
        how="inner")
    return gpd.GeoDataFrame(df_metro_inau)



def feature_diff_gis(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf["dist_od"] = gdf.apply(
        lambda x: x["loc_origem"].distance(x["loc_destino"]), axis=1)
    gdf["diff_cota_od"] = gdf["loc_origem_cota"] - gdf["loc_destino_cota"]
    return gdf


def feature_bens_per_capita(gdf: gpd.GeoDataFrame, cols: list, ref: str) -> gpd.GeoDataFrame:
    for c in cols:
        gdf[f"per {c}"] = gdf[c] / gdf[ref]
    return gdf


def feature_time(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf["Hora Saída"] = gdf["Hora Saída"] + gdf["Minuto Saída"] / 60
    gdf["Entre 21-23"] = gdf["Hora Saída"].between(
        21, 23, inclusive=True).astype(int)
    gdf["Entre 23-04"] = gdf["Hora Saída"].between(23, 24, inclusive=True).astype(
        int) + gdf["Hora Saída"].between(0, 4, inclusive=True).astype(int)
    return gdf


# ------------------------------------
# Sub-Main Functions
# ------------------------------------

def create_gis_features_ilumina(gpd_raw: gpd.GeoDataFrame, gdf_ilumin: gpd.GeoDataFrame, dict_gis: Dict) -> gpd.GeoDataFrame:
    gdf_ilumin_led, gdf_ilumin_std = prep_ilumina(gdf_ilumin)
    for k in dict_gis.keys():
        print(f"Create infrastructure feature for {k}")
        print("Light Pole (Led)")
        gpd_raw[k + "_count_ilum_led"] = get_counting(
            gpd_raw, gdf_ilumin_led, k, 150, "il_placa", ["Identifica pessoa"])
        print("Light Pole (Std)")
        gpd_raw[k + "_count_ilum_std"] = get_counting(
            gpd_raw, gdf_ilumin_std, k, 150, "il_placa", ["Identifica pessoa"])
    return gpd_raw


def create_gis_features_cota(gpd_raw: gpd.GeoDataFrame, gdf_cota: gpd.GeoDataFrame, dict_gis: dict) -> gpd.GeoDataFrame:
    for k in dict_gis.keys():
        print("Elevation")
        gpd_raw[k + "_cota"] = predict_knn(gpd_raw, gdf_cota, k, 5)
    return gpd_raw


def features_gis_acc(gdf: gpd.GeoDataFrame, geocol: str, gdf_acc: gpd.GeoDataFrame, r_cols: list, prefix: str = '_') -> gpd.GeoDataFrame:
    gdf = geo_prep_to_join(gdf, geocol)
    cols = list(gdf.columns)
    gdf_aux = gpd.sjoin(gdf, gdf_acc, how="left", op='intersects')
    dict_geo = {"geometry": geocol}
    dict_pref = dict(zip(r_cols, [geocol + prefix + c for c in r_cols]))
    dict_geo.update(dict_pref)
    return gdf_aux[cols + r_cols].rename(columns=dict_geo)


def create_gis_features_eng(
    gpd_raw: gpd.GeoDataFrame,
    gdf_metro: gpd.GeoDataFrame,
    gdf_trem: gpd.GeoDataFrame,
    gdf_term: gpd.GeoDataFrame,
    gdf_ponto: gpd.GeoDataFrame,
    gdf_ciclo: gpd.GeoDataFrame,
    dict_gis: dict
) -> gpd.GeoDataFrame:

    for k in dict_gis.keys():
        print(f"Create infrastructure feature for {k}")
        # Ciclovia
        print("Bycle Path")
        gpd_raw[k + "_dist_ciclo"] = gpd_raw[k].apply(
            lambda x: gdf_ciclo.distance(x))
        # Metro
        print("Subway")
        gpd_raw[f"{k}_dist_metro"] = gpd_raw[k].apply(
            lambda x: gdf_metro.distance(x))
        # Trem
        print("Train")
        gpd_raw[f"{k}_dist_trem"] = gpd_raw[k].apply(
            lambda x: gdf_trem.distance(x))
        # Terminal
        print("Bus Terminal")
        gpd_raw[f"{k}_dist_term"] = gpd_raw[k].apply(
            lambda x: gdf_term.distance(x))
        # Parada
        print("Bus Stop")
        gpd_raw[k + "_count_parada"] = get_counting(
            gpd_raw, gdf_ponto, k, 300, "pt_nome", ["Identifica pessoa"])

    return gpd_raw

# ------------------------------------
# Main  Functions: Nodes
# ------------------------------------


def node_features_gis_ilumina(od2017_filtered: pd.DataFrame, ilumina: gpd.GeoDataFrame) -> pd.DataFrame:
    print("Create main location gis points")
    gdf_gis_final = create_gis_point(od2017_filtered, DICT_GIS_17)
    print("Creating infrastructure features")
    gdf_gis_final = create_gis_features_ilumina(gdf_gis_final, ilumina, DICT_GIS_17)
    # Drop Geometry Columns
    gdf_gis_final = gdf_gis_final.drop(columns=DICT_GIS_17.keys())
    return gdf_gis_final


def node_features_gis_cota(od2017_eng: pd.DataFrame, gdf_cota: gpd.GeoDataFrame) -> pd.DataFrame:
    print("Create main location gis points")
    gdf_gis_final = create_gis_point(od2017_eng, DICT_GIS_17)
    print("Creating infrastructure features")
    gdf_gis_final = create_gis_features_cota(gdf_gis_final, gdf_cota, DICT_GIS_17)
    # Drop Geometry Columns
    gdf_gis_final = gdf_gis_final.drop(columns=DICT_GIS_17.keys())
    return gdf_gis_final


def node_features_others(od2017_cota: pd.DataFrame, params: Dict) -> pd.DataFrame:
    bens = params["bens"]
    per = params["per"]
    gdf_gis_final = create_gis_point(od2017_cota, DICT_GIS_17)
    print("Creating socioeconomic features")
    gdf_gis_final = feature_bens_per_capita(od2017_cota, bens, per)
    print("Creating diff features")
    gdf_gis_final = feature_diff_gis(gdf_gis_final)
    print("Creating trip time features")
    gdf_gis_final = feature_time(gdf_gis_final)
    print("Deciclividade")
    gdf_gis_final["declividade"] = gdf_gis_final["diff_cota_od"] / \
        gdf_gis_final["dist_od"] * 100
    # Drop Geometry Columns
    gdf_gis_final = gdf_gis_final.drop(columns=DICT_GIS_17.keys())
    return gdf_gis_final


def node_features_acc(df_socio: pd.DataFrame, gdf_acc_joined: gpd.GeoDataFrame) -> pd.DataFrame:
    print("Create main location gis points")
    gdf_gis_final = create_gis_point(df_socio, DICT_GIS_17)
    for k in DICT_GIS_17.keys():
        r_cols = ["A_L_TI_", "A_L_TP_", "A_E_60M_TI_", "A_E_60M_TP_"]
        gdf_gis_final = features_gis_acc(
            gdf_gis_final, k, gdf_acc_joined, r_cols
        )
    # Drop Geometry Columns
    gdf_gis_final = gdf_gis_final.drop(columns=DICT_GIS_17.keys())
    return gdf_gis_final


def node_features_gis_eng(
    od2017_ilumina: pd.DataFrame,
    gdf_metro: gpd.GeoDataFrame,
    gdf_metro_inau: pd.DataFrame,
    gdf_trem: gpd.GeoDataFrame,
    gdf_term: gpd.GeoDataFrame,
    gdf_ciclo: gpd.GeoDataFrame,
    gdf_ponto: gpd.GeoDataFrame
) -> pd.DataFrame:

    print("Prep Ciclo")
    gdf_ciclo = prep_ciclo(gdf_ciclo)
    print("prep Subway")
    gdf_metro = prep_subway(gdf_metro, gdf_metro_inau)
    print("Create main location gis points")
    gdf_gis_final = create_gis_point(od2017_ilumina, DICT_GIS_17)
    print("Creating infrastructure features")
    gdf_gis_final = create_gis_features_eng(
        gpd_raw=gdf_gis_final,
        gdf_metro=unary_union(gdf_metro["geometry"]),
        gdf_trem=unary_union(gdf_trem["geometry"]),
        gdf_term=unary_union(gdf_term["geometry"]),
        gdf_ponto=gdf_ponto,
        gdf_ciclo=unary_union(gdf_ciclo["geometry"]),
        dict_gis=DICT_GIS_17)
    # Drop Geometry Columns
    gdf_gis_final = gdf_gis_final.drop(columns=DICT_GIS_17.keys())
    return gdf_gis_final


def node_diff_07(
    od2007: pd.DataFrame,
    gdf_metro: gpd.GeoDataFrame,
    params: Dict
) -> pd.DataFrame:

    GEO_DOM = ["Coordenada X Domicílio", "Coordenada Y Domicílio"]
    DICT_GIS = {"loc_domicilio": GEO_DOM}
    k = "loc_domicilio"

    # Clusters
    c_perif = params["features"]["diff-in-diffs"]["od2007"]["zone"]["cluster_perif"]
    c_center = params["features"]["diff-in-diffs"]["od2007"]["zone"]["cluster_center"]
    subways = params["features"]["diff-in-diffs"]["subway"]
    # Distance
    dist_min = params["features"]["diff-in-diffs"]["distance"]["treatment"]
    dist_max = params["features"]["diff-in-diffs"]["distance"]["treatment-max"]
    # Socio
    bens = ['Quantidade Automóvel', "Renda Familiar Mensal"]
    per = "Número de Moradores no Domicílio"

    # Filtrar os domicilios segundo zona
    print(od2007.columns)
    od07_perif = od2007[od2007["Zona de Domicílio"].isin(c_perif)]
    od07_center = od2007[od2007["Zona de Domicílio"].isin(c_center)]
    od07_perif["cluster"] = "Periferia"
    od07_center["cluster"] = "Centro"
    gdf_gis_final = pd.concat([od07_perif, od07_center])
    print("Create main location gis points")
    gdf_gis_final = create_gis_point(gdf_gis_final, DICT_GIS)
    # Filtrar Metros Desejados
    gdf_metro = gdf_metro[gdf_metro["emt_nome"].isin(subways)]
    geo_metro = unary_union(gdf_metro["geometry"])
    # Calcular distância até metrô
    print("Distance to Subway")
    gdf_gis_final[f"{k}_dist_metro"] = gdf_gis_final[k].apply(lambda x: geo_metro.distance(x))
    # Filtrar metros até certa distância
    gdf_gis_final = gdf_gis_final[gdf_gis_final[f"{k}_dist_metro"] < dist_max]
    gdf_gis_final["T"] = gdf_gis_final[f"{k}_dist_metro"].apply(lambda x: 1 if x <= dist_min else 0)
    # Calcular diff-socio
    gdf_gis_final = feature_bens_per_capita(gdf_gis_final, bens, per)
    # Drop Geometry Columns
    gdf_gis_final = gdf_gis_final.drop(columns=DICT_GIS.keys())
    return gdf_gis_final


def node_diff_17(
    od2017: pd.DataFrame,
    gdf_metro: gpd.GeoDataFrame,
    params: Dict
) -> pd.DataFrame:

    GEO_DOM = ["Coordenada X domicílio", "Coordenada Y domicílio"]
    k = "loc_domicilio"
    DICT_GIS = {k: GEO_DOM}


    # Clusters
    c_perif = params["features"]["diff-in-diffs"]["od2017"]["zone"]["cluster_perif"]
    c_center = params["features"]["diff-in-diffs"]["od2017"]["zone"]["cluster_center"]
    subways = params["features"]["diff-in-diffs"]["subway"]
    # Distance
    dist_min = params["features"]["diff-in-diffs"]["distance"]["treatment"]
    dist_max = params["features"]["diff-in-diffs"]["distance"]["treatment-max"]
    # Socio
    bens = ['Quantidade de automóveis', "Renda familiar mensal"]
    per = "Total de moradores na família"

    # Filtrar os domicilios segundo zona
    print(od2017.columns)
    od17_perif = od2017[od2017["Zona de domicílio"].isin(c_perif)]
    od17_center = od2017[od2017["Zona de domicílio"].isin(c_center)]
    od17_perif["cluster"] = "Periferia"
    od17_center["cluster"] = "Centro"
    gdf_gis_final = pd.concat([od17_perif, od17_center])
    print("Create main location gis points")
    gdf_gis_final = create_gis_point(gdf_gis_final, DICT_GIS)
    # Filtrar Metros Desejados
    gdf_metro = gdf_metro[gdf_metro["emt_nome"].isin(subways)]
    geo_metro = unary_union(gdf_metro["geometry"])
    # Calcular distância até metrô
    print("Distance to Subway")
    gdf_gis_final[f"{k}_dist_metro"] = gdf_gis_final[k].apply(lambda x: geo_metro.distance(x))
    # Filtrar metros até certa distância
    gdf_gis_final = gdf_gis_final[gdf_gis_final[f"{k}_dist_metro"] < dist_max]
    # Calcular diff-socio
    gdf_gis_final = feature_bens_per_capita(gdf_gis_final, bens, per)
    # Drop Geometry Columns
    gdf_gis_final = gdf_gis_final.drop(columns=DICT_GIS.keys())
    return gdf_gis_final

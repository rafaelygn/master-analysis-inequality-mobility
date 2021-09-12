import time

import geopandas as gpd
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from shapely.ops import unary_union


from src.cota_knn import predict_knn
from src.utils import (geo_prep_to_join, get_counting, get_distance,
                       nearest_point_value, prep_ciclo_shp)

PATH_PROJECT = "/home/yoshraf/projects/master-analysis-inequality-mobility/"

GEO_DOM = ["Coordenada X domicílio", "Coordenada Y domicílio"]
GEO_ORI = ["Coordenada X Origem", "Coordenada Y Origem"]
GEO_DES = ["Coordenada X Destino", "Coordenada Y Destino"]

DICT_GIS = {
    "loc_domicilio": GEO_DOM,
    "loc_origem": GEO_ORI,
    "loc_destino": GEO_DES
}
BENS = ['Quantidade de automóveis', "Quantidade de motocicletas",
        "Quantidade de bicicletas", "Renda familiar mensal"]

PER = "Total de moradores na família"

df = pd.read_parquet(f"{PATH_PROJECT}data/processed/OD_2017.parquet")
df_muni = gpd.read_file(
    f"{PATH_PROJECT}data/gis/od2017/Zonas_2017_region.shp")


def create_gis_point(df_raw: pd.DataFrame, dict_gis: dict) -> pd.DataFrame:
    df_raw.dropna(subset=GEO_DOM, inplace=True)
    # df_raw = df_raw.sample(1000, random_state=42)
    df_raw[GEO_DOM + GEO_ORI + GEO_DES] = df_raw[GEO_DOM + GEO_ORI + GEO_DES].fillna(100)
    # df_raw.dropna(subset=GEO_DOM + GEO_ORI + GEO_DES, inplace=True)
    for k, v in dict_gis.items():
        df_raw[k] = gpd.points_from_xy(df_raw[v[0]], df_raw[v[1]])
        # df_raw.drop(columns=v, inplace=True)
    return gpd.GeoDataFrame(df_raw)


def filter_rows(gdf: GeoDataFrame, geocol: str, municipio: str) -> GeoDataFrame:
    gdf = geo_prep_to_join(gdf, geocol)
    cols = gdf.columns
    df_muni_f = df_muni[df_muni["NomeMunici"] == municipio]
    return gpd.sjoin(gdf, df_muni_f, how="inner", op='intersects')[cols].rename(columns={"geometry": geocol})


def filter_sp(gdf: GeoDataFrame, dict_gis: dict) -> GeoDataFrame:
    # for k in dict_gis.keys():
    for k in ["loc_domicilio"]:
        gdf = filter_rows(gdf, k, "São Paulo")
    return gdf


def read_gis_files():
    # Leitura dos arquivos
    gdf_metro = gpd.read_file(
        f"{PATH_PROJECT}data/gis/metro/SAD69-96_SHP_estacaometro_point.shp")
    gdf_trem = gpd.read_file(
        f"{PATH_PROJECT}data/gis/trem/SAD69-96_SHP_estacaotrem_point.shp")
    gdf_term = gpd.read_file(
        f"{PATH_PROJECT}data/gis/onibus_terminal/sad6996_terminal_onibus.shp")
    gdf_parada = gpd.read_file(
        f"{PATH_PROJECT}data/gis/onibus_pontos/SAD69-96_SHP_pontoonibus.shp")
    gdf_ciclo = gpd.read_file(
        f"{PATH_PROJECT}data/gis/ciclovia/sad6996_ciclovia.shp")
    mulipoint_metro = unary_union(gdf_metro["geometry"])
    mulipoint_trem = unary_union(gdf_trem["geometry"])
    mulipoint_term = unary_union(gdf_term["geometry"])
    multiline_ciclo = prep_ciclo_shp(gdf_ciclo, 1.5)
    gdf_ilumin = gpd.read_file(
        f"{PATH_PROJECT}/data/gis/iluminacao/SAD69-96_SHP_iluminacaopublica.shp")
    # Tratamento da iluminação
    gdf_ilumin_led = gdf_ilumin[gdf_ilumin["il_tipo"] == "LED"]
    gdf_ilumin_std = gdf_ilumin[gdf_ilumin["il_tipo"] != "LED"]
    # gdf_ilumin_std = gdf_ilumin_std.sample(frac=.15, random_state=42)
    print(f"Loading light pole sample. Rows loaded: {gdf_ilumin_std.shape[0]}")
    return mulipoint_metro, mulipoint_trem, mulipoint_term, gdf_parada, gdf_ilumin_led, gdf_ilumin_std, multiline_ciclo


def create_distance_feature(gpd_raw: GeoDataFrame, gdf_ref: GeoDataFrame, geo_ref: str, col_ref: str) -> GeoDataFrame:
    gpd_new = gpd.GeoDataFrame(gpd_raw, geometry=geo_ref)
    gpd_new["geo_aux"] = gpd_new[geo_ref].apply(
        lambda row: nearest_point_value(row, gdf_ref))
    gpd_new[f"{geo_ref}_dist_{col_ref}"] = get_distance(
        gpd_new[geo_ref], gpd_new["geo_aux"])
    gpd_new.drop(columns=["geo_aux"], inplace=True)
    return gpd_new


def create_gis_features(gpd_raw: GeoDataFrame, dict_gis: dict) -> GeoDataFrame:
    print("Loading infrastructure shapefiles...")
    mulipoint_metro, mulipoint_trem, mulipoint_term, gdf_parada, gdf_ilumin_led, gdf_ilumin_std, multiline_ciclo = read_gis_files()
    for k in dict_gis.keys():
        print(f"Create infrastructure feature for {k}")
        # Cota
        print("Elevation")
        gpd_raw[k + "_cota"] = predict_knn(gpd_raw, k, 3)
        # Parada
        print("Bus Stop")
        gpd_raw[k + "_count_parada"] = get_counting(
            gpd_raw, gdf_parada, k, 300, "pt_nome", ["Identifica pessoa"])
        # Iluminação LED e padrão
        print("Light Pole (Led)")
        gpd_raw[k + "_count_ilum_led"] = get_counting(
            gpd_raw, gdf_ilumin_led, k, 150, "il_unidade", ["Identifica pessoa"])
        print("Light Pole (Std)")
        gpd_raw[k + "_count_ilum_std"] = get_counting(
            gpd_raw, gdf_ilumin_std, k, 150, "il_unidade", ["Identifica pessoa"])
        # # Metro
        print("Subway")
        # gpd_raw = create_distance_feature(gpd_raw, gdf_metro, k, "metro")
        gpd_raw[f"{k}_dist_metro"] = gpd_raw[k].apply(
            lambda x: mulipoint_metro.distance(x))
        # Trem
        print("Train")
        # gpd_raw = create_distance_feature(gpd_raw, gdf_trem, k, "trem")
        gpd_raw[f"{k}_dist_trem"] = gpd_raw[k].apply(
            lambda x: mulipoint_trem.distance(x))
        # # Terminal
        print("Bus Terminal")
        # gpd_raw = create_distance_feature(gpd_raw, gdf_term, k, "trem")
        gpd_raw[f"{k}_dist_term"] = gpd_raw[k].apply(
            lambda x: mulipoint_term.distance(x))
        # Ciclovia
        print("Bycle Path")
        gpd_raw[k + "_dist_ciclo"] = gpd_raw[k].apply(
            lambda x: multiline_ciclo.distance(x))
    return gpd_raw


def feature_diff_gis(gdf: GeoDataFrame) -> GeoDataFrame:
    gdf["dist_od"] = get_distance(gdf["loc_origem"], gdf["loc_destino"])
    gdf["diff_cota_od"] = gdf["loc_origem_cota"] - gdf["loc_destino_cota"]
    return gdf


def feature_bens_per_capita(gdf: GeoDataFrame, cols: list, ref: str) -> GeoDataFrame:
    for c in cols:
        gdf[f"per {c}"] = gdf[c] / gdf[ref]
    return gdf


def feature_time(gdf: GeoDataFrame) -> GeoDataFrame:
    gdf["Hora Saída"] = gdf["Hora Saída"] + gdf["Minuto Saída"] / 60
    gdf["Entre 21-23"] = gdf["Hora Saída"].between(
        21, 23, inclusive=True).astype(int)
    gdf["Entre 23-04"] = gdf["Hora Saída"].between(23, 24, inclusive=True).astype(
        int) + gdf["Hora Saída"].between(0, 4, inclusive=True).astype(int)
    return gdf


def features_gis_acc(gdf: GeoDataFrame, geocol: str, gdf_acc: GeoDataFrame, r_cols: list, prefix: str) -> GeoDataFrame:
    gdf = geo_prep_to_join(gdf, geocol)
    cols = list(gdf.columns)
    gdf_aux = gpd.sjoin(gdf, gdf_acc, how="left", op='intersects')
    dict_geo = {"geometry": geocol}
    dict_pref = dict(zip(r_cols, [geocol + prefix + c for c in r_cols]))
    dict_geo.update(dict_pref)
    return gdf_aux[cols + r_cols].rename(columns=dict_geo)


def main_features_gis_acc(gdf: GeoDataFrame, dict_gis: dict) -> GeoDataFrame:
    # Read Acc Shapefiles
    gdf_acc_employ_ti = gpd.read_file(
        f"{PATH_PROJECT}data/gis/acessibilidade/Acessibilidade_Empregos/Zonas_OD2017_Acc_60Min_Empregos_TI_Qua_7AM.shp")
    gdf_acc_employ_tp = gpd.read_file(
        f"{PATH_PROJECT}data/gis/acessibilidade/Acessibilidade_Empregos/Zonas_OD2017_Acc_60Min_Empregos_TP_Qua_7AM.shp")
    gdf_acc_leisure_ti = gpd.read_file(
        f"{PATH_PROJECT}data/gis/acessibilidade/Acessibilidade_Lazer/Zonas_OD2017_Acc_30Min_Lazer_TI_Dom_10AM.shp")
    gdf_acc_leisure_tp = gpd.read_file(
        f"{PATH_PROJECT}data/gis/acessibilidade/Acessibilidade_Lazer/Zonas_OD2017_Acc_30Min_Lazer_TP_Dom_10AM.shp")
    # For each localization Point
    # Intersection with accessibility shapefile
    for k in dict_gis.keys():
        print(f"Intersection with {k}")
        gdf = features_gis_acc(
            gdf, k, gdf_acc_employ_ti, ["A_E_60M"], "_ACC_TI_")
        gdf = features_gis_acc(
            gdf, k, gdf_acc_employ_tp, ["A_E_60M"], "_ACC_TP_")
        gdf = features_gis_acc(
            gdf, k, gdf_acc_leisure_ti, ["A_L_TI_"], "_ACC_TI_")
        gdf = features_gis_acc(
            gdf, k, gdf_acc_leisure_tp, ["A_L_TP_"], "_ACC_TI_")
    return gdf


def main(sample=False):
    start = time.time()
    print("Create main location gis points")
    gdf_gis_final = create_gis_point(df, DICT_GIS)
    print(gdf_gis_final.shape)
    if sample:
        print("Sampling...")
        gdf_gis_final = gdf_gis_final.sample(40_000)
    print("Filtering by São Paulo City")
    gdf_gis_final = filter_sp(gdf_gis_final, DICT_GIS)
    print("Creating socioeconomic features")
    gdf_gis_final = feature_bens_per_capita(gdf_gis_final, BENS, PER)
    print(f"Rows: {gdf_gis_final.shape[0]}")
    print("Creating infrastructure features")
    gdf_gis_final = create_gis_features(gdf_gis_final, DICT_GIS)
    print("Creating accessibility features")
    gdf_gis_final = main_features_gis_acc(gdf_gis_final, DICT_GIS)
    print("Creating diff features")
    gdf_gis_final = feature_diff_gis(gdf_gis_final)
    print("Creating trip time features")
    gdf_gis_final = feature_time(gdf_gis_final)
    print("Salvando o resultado em parquet")
    gdf_gis_final.to_parquet(f"{PATH_PROJECT}data/processed/dataset.parquet")
    print("Everything is ok!")
    end = time.time()
    print(f"Time to run main funcion: {end - start}")
    return None


# Execute main function
if __name__ == "__main__":
    main(False)


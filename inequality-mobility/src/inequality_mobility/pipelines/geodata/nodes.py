from typing import Any, Dict, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pyreadstat
import scipy.cluster.hierarchy as shc
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering


def convert_label(df: pd.DataFrame, meta: Any, columns: bool = True, labels: bool = True) -> pd.DataFrame:
    """Descodifica colunas e descodifica o valores da coluna
    """
    df_raw = df.copy()
    if labels:
        for k, v in meta.variable_value_labels.items():
            df_raw[k] = df_raw[k].map(v)
    if columns:
        df_raw.columns = meta.column_labels
    return df_raw


def read_shp(zipfile: str, crs: str = None, to_crs: str = None) -> gpd.GeoDataFrame:
    """Le Arquivos shapefiles zipados do GeoSampa
    """
    file = "zip://"+zipfile
    gdf = gpd.read_file(file)
    # Set CRS
    if crs:
        gdf.crs = crs
    if to_crs:
        # Convert CRS
        gdf.to_crs(to_crs, inplace=True)
    return gdf


def read_od2017(file: str) -> pd.DataFrame:
    """Le base de OD2017 e transforma labels
    """
    df_sav, meta = pyreadstat.read_sav(file)
    df_od17 = convert_label(df_sav, meta)
    return df_od17


def od17_filter_cols(df_od17: pd.DataFrame, parameters: Dict, type: str) -> pd.DataFrame:
    features = parameters["features"][type]["features"]
    if type == "od2017":
        df_od17_sp = df_od17[df_od17["Município do domicílio"] == "São Paulo"]
    elif type == "od2007":
        df_od17_sp = df_od17[df_od17["Município do Domicílio"] == "São Paulo"]
    else:
        print(f"Type {type} inválido")
        raise Exception
    return df_od17_sp[features]


def node_prep_od2017(df_od2017: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    df_final = od17_filter_cols(df_od2017, parameters, "od2017")
    return df_final


def node_prep_censo(gdf_censo_prj: gpd.GeoDataFrame, parameters: Dict) -> gpd.GeoDataFrame:
    per = parameters["features"]
    # Teste
    # A soma das porcentagens tem que ser igual a 1
    assert(gdf_censo_prj[per].sum(axis=1).min() >= .99)
    # Select features to cluster
    df_to_clus = gdf_censo_prj[per]
    # Plot Dendogram
    plt.figure(figsize=(10, 7))  
    plt.title("Dendrograms")  
    dend = shc.dendrogram(shc.linkage(df_to_clus.sample(2000, random_state=42), method='ward'))
    # Fit HC
    cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    gdf_censo_prj["cluster"] = cluster.fit_predict(df_to_clus)
    # gdf_censo_prj["cluster"] = gdf_censo_prj["cluster"].astype("category")
    # Plot Spatial
    gdf_censo_prj.plot(column="cluster", figsize = (10,15), legend=True)
    df_cluster = gdf_censo_prj[per + ["cluster"]].groupby(["cluster"]).mean(["alta_branc"])
    print(df_cluster)
    return gdf_censo_prj[["ID",  "NM_DISTRIT", "cluster"] + per + ["total", "geometry"]]


def node_read_gis(parameters: Dict) -> Tuple:
    to_crs = "EPSG:22523"
    crs_sad69 = "EPSG:5533"
    # GeoSampaData
    df_metro = read_shp(parameters["geodata"]["metro"], crs_sad69, to_crs)
    df_trem = read_shp(parameters["geodata"]["trem"], crs_sad69, to_crs)
    df_ciclovia = read_shp(parameters["geodata"]["ciclovia"], crs_sad69, to_crs)
    df_ilumina = read_shp(parameters["geodata"]["ilumina"], crs_sad69, to_crs)
    df_ponto_onibus = read_shp(parameters["geodata"]["ponto_onibus"], crs_sad69, to_crs)
    df_ponto_cotado = read_shp(parameters["geodata"]["ponto_cotado"], crs_sad69, to_crs)
    df_terminal = read_shp(parameters["geodata"]["terminal"], crs_sad69, to_crs)
    # Externals
    # External - Acessibilidade
    df_emp_ti = read_shp(parameters["geodata"]["acess_empregos_ti"], to_crs)
    df_emp_tp = read_shp(parameters["geodata"]["acess_empregos_tp"], to_crs)
    df_laz_ti = read_shp(parameters["geodata"]["acess_lazer_ti"], to_crs)
    df_laz_tp = read_shp(parameters["geodata"]["acess_lazer_tp"], to_crs)
    # External - Censo Demográfico
    df_censo_demo = read_shp(parameters["geodata"]["censo_demo"], to_crs = to_crs)
    # OD2017
    df_od17 = read_od2017(parameters["geodata"]["od2017"])
    
    return (
        df_od17,
        df_metro, df_trem, df_ciclovia, df_ilumina, df_ponto_onibus, df_ponto_cotado, df_terminal,
        df_emp_ti, df_emp_tp, df_laz_ti, df_laz_tp,
        df_censo_demo
        )

def node_join_access(
    gdf_acc_emp_ti: gpd.GeoDataFrame, 
    gdf_acc_emp_tp: gpd.GeoDataFrame,
    gdf_acc_laz_ti: gpd.GeoDataFrame,
    gdf_acc_laz_tp: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
    gdf_laz = pd.merge(
    gdf_acc_laz_ti, gdf_acc_laz_tp[["NumerZn", "A_L_TP_"]], on=["NumerZn"]
    )
    gdf_emp = pd.merge(
        gdf_acc_emp_ti, gdf_acc_emp_tp[["NumerZn", "A_E_60M"]], on=["NumerZn"], suffixes = ("_TI_", "_TP_")
    )
    gdf_acc = pd.merge(
        gdf_laz, gdf_emp[["NumerZn", "A_E_60M_TI_", "A_E_60M_TP_"]], on=["NumerZn"], suffixes = ("_TI_", "_TP_")
    )
    return gdf_acc

def node_read_prep_od2007(parameters: Dict) -> pd.DataFrame:
    df_od07 = read_od2017(parameters["geodata"]["od2007"])
    df_od07 = od17_filter_cols(df_od07, parameters, "od2007")
    return df_od07

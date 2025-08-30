import geopandas as gpd
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyreadstat
# import scipy.cluster.hierarchy as shc
# from sklearn.cluster import AgglomerativeClustering
from unidecode import unidecode
from shapely.geometry import Point

import re
from typing import Any, Dict, Tuple

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


def normalize_columns(df):
    normalize_columns = []
    for col in df.columns:
        # remove acentos e põe minúsculo
        col = unidecode(col).lower()
        # substitui símbolos por _
        col = re.sub(r"[^\w]", "_", col)
        # colapsa múltiplos _ em um só
        col = re.sub(r"_+", "_", col)
        # remove _ do início/fim
        col = col.strip("_")
        normalize_columns.append(col)
    df.columns = normalize_columns
    return df

def deduplicate_columns_names(df: pd.DataFrame) -> pd.DataFrame:
    # Detecta duplicadas
    duplicated_cols = df.columns[df.columns.duplicated()]
    if duplicated_cols.empty:
        return df
    else:
        print(f"Colunas duplicadas: {duplicated_cols}")
        # Renomeia duplicadas adicionando um sufixo incremental
        new_columns = []
        counts = {}
        for col in df.columns:
            if col in counts:
                counts[col] += 1
                new_columns.append(f"{col}_{counts[col]}")
            else:
                counts[col] = 0
                new_columns.append(col)
        df.columns = new_columns
        return df

def read_od(file: str) -> pd.DataFrame:
    """Le base de OD e transforma labels
    """
    df_sav, meta = pyreadstat.read_sav(file)
    df_od = convert_label(df_sav, meta)
    df_od = deduplicate_columns_names(df_od)
    return df_od


def od_filter_cols(df_od: pd.DataFrame, parameters: Dict, type: str) -> pd.DataFrame:
    features = parameters["features"][type]["features"]
    if type == "od2023":
        df_od = normalize_columns(df_od)
        df_od_sp = df_od[df_od["municipio_de_domicilio"] == "São Paulo"]
    elif type == "od2017":
        df_od_sp = df_od[df_od["Município do domicílio"] == "São Paulo"]
    elif type == "od2007":
        df_od_sp = df_od[df_od["Município do Domicílio"] == "São Paulo"]
        df_od_sp = df_od_sp[features.keys()]
        df_od_sp.columns = features.values()
        return df_od_sp
    else:
        print(f"Type {type} inválido")
        raise Exception
    return df_od_sp[features]


def node_prep_od2023(df_od2023: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    df_final = od_filter_cols(df_od2023, parameters, "od2023")
    return df_final


def node_prep_od2017(df_od2017: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    df_final = od_filter_cols(df_od2017, parameters, "od2017")
    return df_final

def node_read_prep_od2007(parameters: Dict) -> pd.DataFrame:
    df_od07 = read_od(parameters["geodata"]["od2007"])
    df_od07 = od_filter_cols(df_od07, parameters, "od2007")
    return df_od07

def node_prep_metro(gdf_metro: gpd.GeoDataFrame, df_inauguracao: gpd.GeoDataFrame, parameters: Dict) -> gpd.GeoDataFrame:
    EPSG = parameters["geral"]["EPSG"]
    # Metro
    # Reserva a geometria original
    crs_original = gdf_metro.crs
    # Adiciona manualmente as linhas jardim colonial e  vila sonia
    nova_linha_jcolonial = {
        'emt_empres': 'METRO',
        'emt_situac': 'OPERANDO',
        'emt_linha': 'PRATA',
        'emt_nome': 'JARDIM COLONIAL',
        'geometry': Point(350115, 7389400)
    }
    nova_linha_vsonia = {
        'emt_empres': 'METRO',
        'emt_situac': 'OPERANDO',
        'emt_linha': 'AMARELA',
        'emt_nome': 'VILA SÔNIA',
        'geometry': Point(322988, 7389708)
    }
    gdf_metro = gdf_metro.append(nova_linha_jcolonial, ignore_index=True)
    gdf_metro = gdf_metro.append(nova_linha_vsonia, ignore_index=True)
    # Referencia a geometria
    gdf_metro.crs = crs_original
    gdf_metro.to_crs(epsg=EPSG, inplace=True)

    # Inauguração
    # Ajusta nomenclatura da linha 5 para Metro
    df_inauguracao['Construção'] = np.where(df_inauguracao['Linha'] == 5, 'Metrô', df_inauguracao['Construção'])
    # Filtra metro
    df_inauguracao = df_inauguracao[(df_inauguracao['Construção'] == 'Metrô')]
    # Ajusta nomes das estacoes
    df_inauguracao = df_inauguracao.assign(emt_nome=df_inauguracao['Nome'].str.upper())
    map_subway_name = {
        'JARDIM SÃO PAULO': 'AYRTON SENNA-JARDIM SÃO PAULO',
        'SUMARÉ': 'SANTUÁRIO NOSSA SENHORA DE FÁTIMA-SUMARÉ',
        'HIGIENÓPOLIS–MACKENZIE': 'HIGIENÓPOLIS-MACKENZIE',
        'PALMEIRAS–BARRA FUNDA': 'PALMEIRAS-BARRA FUNDA',
        'LIBERDADE': 'JAPÃO-LIBERDADE',
        'TIETÊ': 'PORTUGUESA-TIETÊ',
    }
    df_inauguracao.emt_nome = df_inauguracao.emt_nome.map(map_subway_name).fillna(df_inauguracao.emt_nome)
    # Ajusta nomes das linhas
    map_lines = {
            1: "AZUL",
            2: "VERDE",
            3: "VERMELHA",
            4: "AMARELA",
            5: "LILAS",
            15: "PRATA"
    }
    df_inauguracao["emt_linha"] = df_inauguracao["Linha"].map(map_lines)
    # Filtro por data
    df_inauguracao["Inauguração"] = pd.to_datetime(df_inauguracao["Inauguração"])
    df_inauguracao = df_inauguracao[(df_inauguracao["Inauguração"].dt.year <= 2024)]
    # Join
    df_metro_inau = pd.merge(
        df_inauguracao, gdf_metro, 
        on=["emt_nome", "emt_linha"],
        how="inner"
    )
    return gpd.GeoDataFrame(df_metro_inau)

# def node_prep_censo(gdf_censo_prj: gpd.GeoDataFrame, parameters: Dict) -> gpd.GeoDataFrame:
#     per = parameters["features"]
#     # Teste
#     # A soma das porcentagens tem que ser igual a 1
#     assert(gdf_censo_prj[per].sum(axis=1).min() >= .99)
#     # Select features to cluster
#     df_to_clus = gdf_censo_prj[per]
#     # Plot Dendogram
#     plt.figure(figsize=(10, 7))  
#     plt.title("Dendrograms")  
#     dend = shc.dendrogram(shc.linkage(df_to_clus.sample(2000, random_state=42), method='ward'))
#     # Fit HC
#     cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
#     gdf_censo_prj["cluster"] = cluster.fit_predict(df_to_clus)
#     # gdf_censo_prj["cluster"] = gdf_censo_prj["cluster"].astype("category")
#     # Plot Spatial
#     gdf_censo_prj.plot(column="cluster", figsize = (10,15), legend=True)
#     df_cluster = gdf_censo_prj[per + ["cluster"]].groupby(["cluster"]).mean(["alta_branc"])
#     print(df_cluster)
#     return gdf_censo_prj[["ID",  "NM_DISTRIT", "cluster"] + per + ["total", "geometry"]]


def node_read_gis(parameters: Dict) -> Tuple:
    to_crs = "EPSG:22523"
    crs_sad69 = "EPSG:5533"
    # GeoSampaData
    df_metro = read_shp(parameters["geodata"]["metro"], crs_sad69, to_crs)
    df_trem = read_shp(parameters["geodata"]["trem"], crs_sad69, to_crs)
    df_ciclovia = read_shp(parameters["geodata"]["ciclovia"], crs_sad69, to_crs)
    # df_ilumina = read_shp(parameters["geodata"]["ilumina"], crs_sad69, to_crs)
    # df_ponto_onibus = read_shp(parameters["geodata"]["ponto_onibus"], crs_sad69, to_crs)
    # df_ponto_cotado = read_shp(parameters["geodata"]["ponto_cotado"], crs_sad69, to_crs)
    df_terminal = read_shp(parameters["geodata"]["terminal"], crs_sad69, to_crs)
    # Externals
    # External - Acessibilidade
    # df_emp_ti = read_shp(parameters["geodata"]["acess_empregos_ti"], to_crs)
    # df_emp_tp = read_shp(parameters["geodata"]["acess_empregos_tp"], to_crs)
    # df_laz_ti = read_shp(parameters["geodata"]["acess_lazer_ti"], to_crs)
    # df_laz_tp = read_shp(parameters["geodata"]["acess_lazer_tp"], to_crs)
    # External - Censo Demográfico
    # df_censo_demo = read_shp(parameters["geodata"]["censo_demo"], to_crs = to_crs)
    # OD2017
    df_od17 = read_od(parameters["geodata"]["od2017"])
    # OD2023
    df_od23 = read_od(parameters["geodata"]["od2023"])
    
    return (
        df_od23, df_od17,
        df_metro, df_trem, df_ciclovia, 
        # df_ilumina,
        # df_ponto_onibus, df_ponto_cotado,
        df_terminal,
        # df_emp_ti, df_emp_tp, df_laz_ti, df_laz_tp,
        # df_censo_demo
        )

# def node_join_access(
#     gdf_acc_emp_ti: gpd.GeoDataFrame, 
#     gdf_acc_emp_tp: gpd.GeoDataFrame,
#     gdf_acc_laz_ti: gpd.GeoDataFrame,
#     gdf_acc_laz_tp: gpd.GeoDataFrame
#     ) -> gpd.GeoDataFrame:
#     gdf_laz = pd.merge(
#     gdf_acc_laz_ti, gdf_acc_laz_tp[["NumerZn", "A_L_TP_"]], on=["NumerZn"]
#     )
#     gdf_emp = pd.merge(
#         gdf_acc_emp_ti, gdf_acc_emp_tp[["NumerZn", "A_E_60M"]], on=["NumerZn"], suffixes = ("_TI_", "_TP_")
#     )
#     gdf_acc = pd.merge(
#         gdf_laz, gdf_emp[["NumerZn", "A_E_60M_TI_", "A_E_60M_TP_"]], on=["NumerZn"], suffixes = ("_TI_", "_TP_")
#     )
#     return gdf_acc

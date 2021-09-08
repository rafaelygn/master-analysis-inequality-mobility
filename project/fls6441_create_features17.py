import time

import geopandas as gpd
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from shapely.ops import unary_union

from src.utils import geo_prep_to_join, get_distance

PATH_PROJECT = "/home/yoshraf/projects/mestrado/"

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

ZONA = [
    # Vila Sônia
    "Vila Sônia",
    # Morumbi
    "Morumbi",
    "Jóquei Clube",
    # Butantã
    "Jardim Caxingui",
    "Jardim Bonfiglioli",
    "Rio Pequeno",
    "Butantã",
    "Cidade Universitária",
    # Pinheiros
    "Pinheiros",
    "Jardim Europa",
    "Jardim Paulistano",
    "Alto Pinheiros",
    # Sacomã
    "Alto do Ipiranga",
    "Sacomã",
    "Moinho Velho",
    "Vila Heliópolis",
    "Ipiranga",
    "Vila Carioca",
    "Vila Independência",
    "Vila Sâo José",
    "Tamanduateí",
    "Vila Zelina",
    "Orfanato",
    "Vila Bertioga",
    "Água Rasa",
    "Linhas Corrente",
    "Vila Ema",
    "Vila Formosa"
]

df = pd.read_parquet(f"{PATH_PROJECT}data/processed/OD_2017.parquet")
df_muni = gpd.read_file(
    f"{PATH_PROJECT}data/gis/od2017/Municipios_2017_region.shp")
gdf_neigh = gpd.read_file(
    f"{PATH_PROJECT}data/gis/od2017/Zonas_2017_region.shp")
gdf_metro = gpd.read_file(
    f"{PATH_PROJECT}data/gis/metro/SAD69-96_SHP_estacaometro_point.shp")
df_ina_metro = pd.read_csv(f"{PATH_PROJECT}data/raw/inauguracoes.csv")


# Áreas de interesse
gdf_neigh = gdf_neigh[gdf_neigh["NomeMunici"] == "São Paulo"]
gdf_neigh["Estudo"] = gdf_neigh["NomeZona"].apply(lambda x: 1 if x in ZONA else 0)
# Metros de Interesse
df_ina_metro["Inauguração"] = pd.to_datetime(df_ina_metro["Inauguração"])
df_ina_metro = df_ina_metro[df_ina_metro["Construção"] == "Metrô"]
df_ina_metro[df_ina_metro["Linha"] != 15].sort_values("Inauguração", ascending=False).head(20)

DEL_SUBWAY = [
    "SÃO LUCAS",
    "VILA TOLSTÓI",
    "VILA UNIÃO",
    "CAMILO HADDAD",
    "SÃO MATEUS",
    "FAZENDA DA JUTA",
    "SAPOPEMBA",
    "JARDIM PLANALTO",
    "SÃO PAULO-MORUMBI"
]

df_ina_metro.sort_values("Inauguração", ascending=False).head(20)


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
    gdf_neigh_f = gdf_neigh[gdf_neigh["Estudo"] == 1]
    return gpd.sjoin(gdf, gdf_neigh_f, how="inner", op='intersects')[cols].rename(columns={"geometry": geocol})


def filter_sp(gdf: GeoDataFrame, dict_gis: dict) -> GeoDataFrame:
    # for k in dict_gis.keys():
    for k in ["loc_domicilio"]:
        gdf = filter_rows(gdf, k, "São Paulo")
    return gdf


def read_gis_files():
    # Leitura dos arquivos
    gdf_metro = gpd.read_file(
        f"{PATH_PROJECT}data/gis/metro/SAD69-96_SHP_estacaometro_point.shp")
    # Filtra metros inexistentes
    gdf_metro = gdf_metro[~ gdf_metro["emt_nome"].isin(DEL_SUBWAY)]
    mulipoint_metro = unary_union(gdf_metro["geometry"])
    return mulipoint_metro


def create_gis_features(gpd_raw: GeoDataFrame, dict_gis: dict) -> GeoDataFrame:
    print("Loading infrastructure shapefiles...")
    mulipoint_metro = read_gis_files()
    for k in dict_gis.keys():
        print(f"Create infrastructure feature for {k}")
        # # Metro
        print("Subway")
        gpd_raw[f"{k}_dist_metro"] = gpd_raw[k].apply(
            lambda x: mulipoint_metro.distance(x))
    return gpd_raw


def feature_diff_gis(gdf: GeoDataFrame) -> GeoDataFrame:
    gdf["dist_od"] = get_distance(gdf["loc_origem"], gdf["loc_destino"])
    return gdf


def feature_bens_per_capita(gdf: GeoDataFrame, cols: list, ref: str) -> GeoDataFrame:
    for c in cols:
        gdf[f"per {c}"] = gdf[c] / gdf[ref]
    return gdf


def main():
    start = time.time()
    print("Create main location gis points")
    gdf_gis_final = create_gis_point(df, DICT_GIS)
    print(gdf_gis_final.shape)
    # print("Sampling...")
    # gdf_gis_final = gdf_gis_final.sample(40_000)
    print("Filtering by São Paulo City")
    gdf_gis_final = filter_sp(gdf_gis_final, DICT_GIS)
    print("Creating socioeconomic features")
    gdf_gis_final = feature_bens_per_capita(gdf_gis_final, BENS, PER)
    print(f"Rows: {gdf_gis_final.shape[0]}")
    print("Creating infrastructure features")
    gdf_gis_final = create_gis_features(gdf_gis_final, DICT_GIS)
    print("Creating diff features")
    gdf_gis_final = feature_diff_gis(gdf_gis_final)
    print("Salvando o resultado em parquet")
    gdf_gis_final.to_parquet(f"{PATH_PROJECT}data/processed/dataset17.parquet")
    print("Everything is ok!")
    end = time.time()
    print(f"Time to run main funcion: {end - start}")
    return None


# Execute main function
if __name__ == "__main__":
    main()

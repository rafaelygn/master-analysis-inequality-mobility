import pandas as pd
import geopandas as gpd

from src.cota_knn import predict_knn
from src.utils import nearest_point_value, get_distance, get_counting, prep_ciclo_shp

PATH_PROJECT = "/home/yoshraf/projects/mestrado/"

# Leitura dos arquivos
df = pd.read_parquet(f"{PATH_PROJECT}data/processed/OD_2017.parquet")
df_metro = gpd.read_file(
    f"{PATH_PROJECT}data/gis/metro/SAD69-96_SHP_estacaometro_point.shp")
df_trem = gpd.read_file(
    f"{PATH_PROJECT}data/gis/trem/SAD69-96_SHP_estacaotrem_point.shp")
df_term = gpd.read_file(
    f"{PATH_PROJECT}data/gis/onibus_terminal/sad6996_terminal_onibus.shp")
df_parada = gpd.read_file(
    f"{PATH_PROJECT}data/gis/onibus_pontos/SAD69-96_SHP_pontoonibus.shp")
df_ciclo = gpd.read_file(
    f"{PATH_PROJECT}data/gis/ciclovia/sad6996_ciclovia.shp")
multiline_ciclo = prep_ciclo_shp(df_ciclo, 1.5)
df_ilumin = gpd.read_file(
    f"{PATH_PROJECT}/data/gis/iluminacao/SAD69-96_SHP_iluminacaopublica.shp")
# Tratamento da iluminação
df_ilumin_led = df_ilumin[df_ilumin["il_tipo"] == "LED"]
df_ilumin_std = df_ilumin[df_ilumin["il_tipo"] != "LED"]

# Pegando uma amostra
df_eu = pd.DataFrame.from_dict({"Identifica pessoa": ["eu", "Looh", "Paraisópolis"], "Coordenada X Origem": [
                               326_673.0, 327_596.0, 324_066], "Coordenada Y Origem": [7_397_413.0, 7_392_871.0, 7387396]})
df_sample = df[["Identifica pessoa", "Coordenada X Origem",
                "Coordenada Y Origem"]].sample(1000, random_state=42)
df_sample = pd.concat([df_eu, df_sample])
df_sample.dropna(subset=["Identifica pessoa",
                         "Coordenada X Origem", "Coordenada Y Origem"], inplace=True)
gdf_origin = gpd.GeoDataFrame(df_sample, geometry=gpd.points_from_xy(
    df_sample["Coordenada X Origem"], df_sample["Coordenada Y Origem"]))


# Cota
gdf_origin["cota"] = predict_knn(
    df_sample[["Coordenada X Origem", "Coordenada Y Origem"]], n_neighbors=3)
# Metro
gdf_origin["metro_geo"] = gdf_origin["geometry"].apply(lambda row: nearest_point_value(row, df_metro))
gdf_origin["dist_metro"] = get_distance(gdf_origin["geometry"], gdf_origin["metro_geo"])
gdf_origin["metro_nearest"] = gdf_origin["geometry"].apply(
    lambda row: nearest_point_value(row, df_metro, col_return="emt_nome"))
gdf_origin["count_metro"] = get_counting(
    gdf_origin, df_metro, 2_500, "emt_nome", ["Identifica pessoa"])
# Trem
gdf_origin["trem_geo"] = gdf_origin["geometry"].apply(
    lambda row: nearest_point_value(row, df_trem))
gdf_origin["dist_trem"] = get_distance(
    gdf_origin["geometry"], gdf_origin["trem_geo"])
gdf_origin["trem_nearest"] = gdf_origin["geometry"].apply(
    lambda row: nearest_point_value(row, df_trem, col_return="etr_nome"))
gdf_origin["count_trem"] = get_counting(
    gdf_origin, df_trem, 2_500, "etr_nome", ["Identifica pessoa"])
# Terminal
gdf_origin["term_geo"] = gdf_origin["geometry"].apply(
    lambda row: nearest_point_value(row, df_term))
gdf_origin["dist_term"] = get_distance(
    gdf_origin["geometry"], gdf_origin["term_geo"])
gdf_origin["term_nearest"] = gdf_origin["geometry"].apply(
    lambda row: nearest_point_value(row, df_term, col_return="nm_termina"))
gdf_origin["count_term"] = get_counting(
    gdf_origin, df_term, 2_500, "nm_termina", ["Identifica pessoa"])
# Parada
gdf_origin["count_parada"] = get_counting(
    gdf_origin, df_parada, 500, "pt_nome", ["Identifica pessoa"])
# Iluminação LED e Padrão
gdf_origin["count_ilumin_led"] = get_counting(
    gdf_origin, df_ilumin_led, 500, "il_unidade", ["Identifica pessoa"])
gdf_origin["count_ilumin_std"] = get_counting(
    gdf_origin, df_ilumin_std, 500, "il_unidade", ["Identifica pessoa"])
# Ciclovia
gdf_origin["dist_ciclo"] = gdf_origin["geometry"].apply(lambda x: multiline_ciclo.distance(x))

gdf_origin.head(4)
gdf_origin.describe()
print("Tudo ocorreu bem!")

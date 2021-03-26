import pandas as pd
import geopandas as gpd

from src.cota_knn import predict_knn
from src.utils import nearest_point_value, get_distance, get_counting, prep_ciclo_shp

PATH_PROJECT = "/home/yoshraf/projects/mestrado/"

pd.set_option('display.max_columns', None)

# Leitura dos arquivos
df = pd.read_parquet(f"{PATH_PROJECT}data/processed/OD_2017.parquet")

# Filtro SP
df = df[df["Município do domicílio"] == "São Paulo"]
# Filtro viangens ocorridas dentro de SP
df.head(5)

# Preencher renda individual por zero
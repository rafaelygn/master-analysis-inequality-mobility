import datetime

import geopandas as gpd
import pandas as pd
import yaml

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

PATH_PROJECT = "/home/yoshraf/projects/mestrado/"
# Global Variables
PATH_R_DATARAW = "data/raw/"
PATH_R_DATAPRC = "data/processed/"
PATH_R_CONF = "project/"
YML = "config.yml"

pd.options.display.max_columns = None

df = pd.read_parquet(f"{PATH_PROJECT}data/processed/dataset.parquet")

# Read config
with open(f"{PATH_PROJECT}{PATH_R_CONF}{YML}", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader, )

# Main variables
cols = config["DATAPREP"]["FLS6641-DISCRETE-CHOICE"]["SELECT_COLUMNS"]
idx = config["DATAPREP"]["FLS6641-DISCRETE-CHOICE"]["IDX"]
mapping = config["DATAPREP"]["FLS6641-DISCRETE-CHOICE"]["ENCODING"]
cat = config["DATAPREP"]["FLS6641-DISCRETE-CHOICE"]["CATEGORICAL"]
label = config["DATAPREP"]["FLS6641-DISCRETE-CHOICE"]["LABEL"]["COLUMN"]
label_values = config["DATAPREP"]["FLS6641-DISCRETE-CHOICE"]["LABEL"]

# Filter, select and Drop Values
dataset = df[cols + [label]].drop_duplicates()

# Filling missing valuies with mean
dataset.fillna(dataset.mean(), inplace=True)
# Mapping
for k, v in mapping.items():
    dataset[k] = dataset[k].map(v)
# Main Mode
dataset = dataset[~ dataset[label].isna()]
dataset[label] = dataset[label].apply(lambda x: 1 if x == "Metrô" else 0)
# Group by person
df_mean = pd.DataFrame(dataset.groupby(["Identifica pessoa"])[label].mean())
df_mean.columns = ["Taxa de Metrô"]
dataset2 = dataset.merge(df_mean, on=["Identifica pessoa"])
dataset2 = dataset2.drop(columns=[label]).drop_duplicates()
# Convert to Datetime
dataset2["Data da entrevista"] = pd.to_datetime(
    dataset2["Data da entrevista"], format="%d%m%Y").values
# GeoJoin
gdf = gpd.GeoDataFrame(dataset2,
                       geometry=gpd.points_from_xy(dataset2["Coordenada X domicílio"],
                                                   dataset2["Coordenada Y domicílio"]))
gdf.head(5)
gdf_neigh = gpd.read_file(
    "/home/yoshraf/projects/mestrado/data/gis/od2017/Distritos_2017_region.shp")
dist = ['Campo Belo', "Santo Amaro", "Itaim Bibi"]
gdf_neigh = gdf_neigh[gdf_neigh["NomeDistri"].isin(dist)]
gdf_f = gpd.sjoin(gdf, gdf_neigh, how="inner", op='intersects')

# --------------------------
# Plot to check our filters
gdf_metro = gpd.read_file(
    f"{PATH_PROJECT}data/gis/metro/SAD69-96_SHP_estacaometro_point.shp")
lst_metro_in = [
    "ALTO DA BOA VISTA",
    "BORBA GATO",
    "BROOKLIN"
]
gdf_metro_f = gdf_metro[gdf_metro["emt_nome"].isin(lst_metro_in)]
gdf_f[f"dist_metro_key"] = gdf_f["geometry"].apply(
    lambda x: gdf_metro_f.distance(x)).min(1)
# Create dist lower than 600, 800, 1000
gdf_f["dist_metro_menor_600"] = gdf_f["dist_metro_key"].apply(lambda x: 1 if x < 600 else 0)
gdf_f["dist_metro_menor_800"] = gdf_f["dist_metro_key"].apply(lambda x: 1 if x < 800 else 0)
gdf_f["dist_metro_menor_1000"] = gdf_f["dist_metro_key"].apply(lambda x: 1 if x < 1000 else 0)
gdf_f["dist_metro_menor_1500"] = gdf_f["dist_metro_key"].apply(lambda x: 1 if x < 1500 else 0)
ax = gdf_metro_f.plot(color='red')
gdf_f.plot(
    ax = ax,
    column="dist_metro_menor_1500",
    alpha = .3)

# Time
gdf_f[gdf_f["Data da entrevista"] < datetime.datetime(2017, 11, 27)].shape
gdf_f[gdf_f["Data da entrevista"] >= datetime.datetime(2017, 11, 27)].shape
# Treatment
gdf_f[(gdf_f["dist_metro_menor_1000"] == 1) &
      (gdf_f["Data da entrevista"] < datetime.datetime(2017, 11, 27))].shape
gdf_f.shape

Tr_b = gdf_f[(gdf_f["dist_metro_menor_1500"] == 1) &
      (gdf_f["Data da entrevista"] < datetime.datetime(2017, 11, 27))]
Tr_a = gdf_f[(gdf_f["dist_metro_menor_1500"] == 1) &
      (gdf_f["Data da entrevista"] > datetime.datetime(2017, 11, 27))]
Cr_b = gdf_f[(gdf_f["dist_metro_menor_1500"] == 0) &
      (gdf_f["Data da entrevista"] < datetime.datetime(2017, 11, 27))]
Cr_a = gdf_f[(gdf_f["dist_metro_menor_1500"] == 0) &
      (gdf_f["Data da entrevista"] > datetime.datetime(2017, 11, 27))]
print(Tr_b["Taxa de Metrô"].mean())
print(Tr_a["Taxa de Metrô"].mean())
print(Cr_b["Taxa de Metrô"].mean())
print(Cr_a["Taxa de Metrô"].mean())
Tr_b.shape

# OneHotCoding
enc = OneHotEncoder(handle_unknown="error", drop="if_binary")
X_cat = pd.DataFrame(enc.fit_transform(
    dataset[cat]).toarray(), columns=enc.get_feature_names(cat))
# All Numeric types
numerics_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
X = pd.concat([dataset.select_dtypes(
    include=numerics_dtypes).reset_index(), X_cat], axis=1).set_index(idx)
# Split into X and y
y = dataset[[label]]
# Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

# Save datasets
X_train.to_parquet(f"{PATH_PROJECT}data/discrete_choice/X_train.parquet")
X_test.to_parquet(f"{PATH_PROJECT}data/discrete_choice/X_test.parquet")
y_train.to_parquet(f"{PATH_PROJECT}data/discrete_choice/y_train.parquet")
y_test.to_parquet(f"{PATH_PROJECT}data/discrete_choice/y_test.parquet")
print("Everything is OK!")

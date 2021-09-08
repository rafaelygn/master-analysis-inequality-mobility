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

df["Motivo no Destino"].value_counts()

# Read config
with open(f"{PATH_PROJECT}{PATH_R_CONF}{YML}", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader, )

# Main variables
cols = config["DATAPREP"]["COUNTING_TRIPS_LEISURE"]["SELECT_COLUMNS"]
idx = config["DATAPREP"]["COUNTING_TRIPS_LEISURE"]["IDX"]
enc_others = config["DATAPREP"]["COUNTING_TRIPS_LEISURE"]["ENCODING"]
cat = config["DATAPREP"]["COUNTING_TRIPS_LEISURE"]["CATEGORICAL"]
label = config["DATAPREP"]["COUNTING_TRIPS_LEISURE"]["LABEL"]
reasons = config["DATAPREP"]["COUNTING_TRIPS_LEISURE"]["REASONS"]

# All Numeric types
numerics_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# Counting interessed trip reasons
df["Total Motivo Interesse"] = df["Motivo no Destino"].apply(lambda x: 1 if x in reasons else 0)
df_aux = df.groupby(cols)["Total Motivo Interesse"].sum().reset_index()
cols += ["Total Motivo Interesse"]
# Filter cols and drop duplicates
dataset = df_aux[cols].drop_duplicates()

# Valid Person ID duplicates
valid_dupli = dataset.groupby(["Identifica pessoa"])[cols[0]].count().max()
if valid_dupli > 1:
    print("Temos duplicação de uma mesma pessoa")
    raise Exception

# Set Index
dataset = dataset.set_index(idx)
# Filling missing valuies with mean
dataset.fillna(dataset.mean(), inplace=True)
# Filtra por Outros
# dataset["Estuda atualmente?"] = dataset["Estuda atualmente?"].apply(
#     lambda x: x if x == "Não" else "Sim")
for k, v in enc_others.items():
    dataset[k] = dataset[k].apply(lambda x: x if x in v else "_outros_")
# OneHotCoding
enc = OneHotEncoder(handle_unknown="error", drop="if_binary")
X_cat = pd.DataFrame(enc.fit_transform(
    dataset[cat]).toarray(), columns=enc.get_feature_names(cat))
X = pd.concat([dataset.select_dtypes(
    include=numerics_dtypes).reset_index(), X_cat], axis=1).set_index(idx)
# Split into X and y
y = X[label]
X.drop(columns=label, inplace=True)
# Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)
# Save datasets
X_train.to_parquet(f"{PATH_PROJECT}data/counting_trips/X_train_leisure.parquet")
X_test.to_parquet(f"{PATH_PROJECT}data/counting_trips/X_test_leisure.parquet")
y_train.to_parquet(f"{PATH_PROJECT}data/counting_trips/y_train_leisure.parquet")
y_test.to_parquet(f"{PATH_PROJECT}data/counting_trips/y_test_leisure.parquet")
print("Everything is OK!")

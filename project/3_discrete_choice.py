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
cols = config["DATAPREP"]["DISCRETE_CHOICE"]["SELECT_COLUMNS"]
idx = config["DATAPREP"]["DISCRETE_CHOICE"]["IDX"]
enc_others = config["DATAPREP"]["DISCRETE_CHOICE"]["ENCODING"]
cat = config["DATAPREP"]["DISCRETE_CHOICE"]["CATEGORICAL"]
label = config["DATAPREP"]["DISCRETE_CHOICE"]["LABEL"]["COLUMN"]
label_values = config["DATAPREP"]["DISCRETE_CHOICE"]["LABEL"]["VALUES"]

# Filter, select and Drop Values
df_filter = df[df[label].isin(label_values)][cols + [label]].drop_duplicates()

# Set Index
dataset = df_filter.set_index(idx)
# Filling missing valuies with mean
dataset.fillna(dataset.mean(), inplace=True)
# Filtra por Outros
# for k, v in enc_others.items():
#     dataset[k] = dataset[k].apply(lambda x: x if x in v else "_outros_")
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

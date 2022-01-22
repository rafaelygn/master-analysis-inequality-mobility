from typing import Dict, Tuple, List

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# All Numeric types
numerics_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

def data_info(dataset: pd.DataFrame) -> pd.DataFrame:
    info = pd.DataFrame()
    info["var"] = dataset.columns
    info["# missing"] = list(dataset.isnull().sum())
    info["% missing"] = info["# missing"] / dataset.shape[0] * 100
    info["types"] = list(dataset.dtypes)
    info["unique values"] = list(
        len(dataset[var].unique()) for var in dataset.columns)
    return info


def encoder_onehot(dataset: pd.DataFrame, cat: List, idx: List) -> pd.DataFrame:
    # OneHotCoding
    enc = OneHotEncoder(handle_unknown="error", drop="if_binary")
    X_cat = pd.DataFrame(enc.fit_transform(
        dataset[cat]).toarray(), columns=enc.get_feature_names(cat))
    X = pd.concat([dataset.select_dtypes(
        include=numerics_dtypes).reset_index(), X_cat], axis=1).set_index(idx)
    return X


def encoder_other(dataset: pd.DataFrame, enc_others: Dict) -> pd.DataFrame:
    for k, v in enc_others.items():
        dataset[k] = dataset[k].apply(lambda x: x if x in v else "_outros_")
    return dataset

def node_create_dis_cho(df: pd.DataFrame, params: Dict) -> Tuple:
    # Main variables from params
    cols = params["columns"]
    idx = params["idx"]
    # enc_others = params["encoding"]
    cat = params["categorical"]
    label = params["label"]["column"]
    label_values = params["label"]["values"]
    # Filter, select and Drop Values
    df_filter = df[df[label].isin(label_values)][cols + [label]].drop_duplicates()
    print(f"Número de linhas:{df_filter.shape[0]}")
    print(f"Profile df:\n{data_info(df_filter)}")
    # Set Index
    dataset = df_filter.set_index(idx)
    # Drop Any NaN
    dataset = dataset.dropna(how="any")
    # Filling missing valuies with mean
    # dataset.fillna(dataset.mean(), inplace=True)
    X = encoder_onehot(dataset, cat, idx)
    # Split into X and y
    y = dataset[[label]]
    # Split into Train and Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)
    return X_train, X_test, y_train, y_test


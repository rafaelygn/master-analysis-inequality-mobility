import geopandas as gpd
import pandas as pd

from pandas.core.frame import DataFrame
from geopandas.geodataframe import GeoDataFrame
from numpy import ndarray
from sklearn.neighbors import KNeighborsRegressor


def create_xy(df_relevo):
    # Tranforma geometry em duas colunas
    df_relevo["x"] = df_relevo["geometry"].apply(lambda pos: pos.x)
    df_relevo["y"] = df_relevo["geometry"].apply(lambda pos: pos.y)
    return df_relevo


def fit_knn(df, n_neighbors: int) -> KNeighborsRegressor:
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(df[["x", "y"]], df["ALTURA"])
    return knn


def decode_to_dataframe(gdp_raw: GeoDataFrame, col: str) -> DataFrame:
    x = gdp_raw[col].apply(lambda pos: pos.x)
    y = gdp_raw[col].apply(lambda pos: pos.y)
    d = {"x": x, "y": y}
    return pd.DataFrame(d)


def predict_knn(gdp_raw: GeoDataFrame, gdp_relevo: GeoDataFrame, col: str, n_neighbors: int) -> ndarray:
    df_relevo = create_xy(gdp_relevo)
    knn = fit_knn(df_relevo, n_neighbors)
    X = decode_to_dataframe(gdp_raw, col)
    return knn.predict(X)

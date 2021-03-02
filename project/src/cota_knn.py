import geopandas as gpd

from pandas.core.frame import DataFrame
from numpy import ndarray
from sklearn.neighbors import KNeighborsRegressor

# Necessário carregar um shp de cotas
df_relevo = gpd.read_file(
    "/home/yoshraf/projects/mestrado/data/gis/ponto_cotado/sad6996_PONTO_COTADO_INTERVIA.shp")
# Tranforma geometry em duas colunas
df_relevo["x"] = df_relevo["geometry"].apply(lambda pos: pos.x)
df_relevo["y"] = df_relevo["geometry"].apply(lambda pos: pos.y)


def fit_knn(df, n_neighbors: int) -> KNeighborsRegressor:
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(df[["x", "y"]], df["ALTURA"])
    return knn


def predict_knn(X: DataFrame, n_neighbors: int) -> ndarray:
    knn = fit_knn(df_relevo, n_neighbors)
    return knn.predict(X)

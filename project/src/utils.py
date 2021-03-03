'''
Funções Uteis
'''

from typing import Any

import geopandas as gpd
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from numpy import ndarray
from pandas.core.frame import DataFrame
from shapely.geometry import Point
from shapely.geometry.multilinestring import MultiLineString
from shapely.ops import nearest_points, unary_union


def convert_label(df: DataFrame, meta: Any, columns: bool = True, labels: bool = True) -> DataFrame:
    """Descodifica colunas e descodifica o valores da coluna
    """
    df_raw = df.copy()
    if labels:
        for k, v in meta.variable_value_labels.items():
            df_raw[k] = df_raw[k].map(v)
    if columns:
        df_raw.columns = meta.column_labels
    return df_raw


def dist_euclidian(coord_x0: float, coord_y0: float, coord_x1: float, coord_y1: float) -> float:
    """Calcula distância euclidiana
    """
    delta_x = coord_x1 - coord_x0
    delta_y = coord_y1 - coord_y0
    return (delta_x**2 + delta_y**2)**(.5)


def nearest_point_value(point: Point, df_ref: GeoDataFrame, col_pts: str = "geometry", col_return: str = "geometry") -> Any:
    """Retorna a coluna/valor mais próximo de um ponto de um geopandas de referência
    """
    pts_ref = df_ref[col_pts].unary_union
    nearest = df_ref[df_ref.geometry == nearest_points(point, pts_ref)[1]]
    return nearest[col_return].values[0]


def get_distance(p1: Point, p2: Point) -> Point:
    """Calcula distância euclidiana entre dois pontos
    """
    return p1.distance(p2)


def geo_prep_to_join(gdf: GeoDataFrame, col: str) -> GeoDataFrame:
    '''Preparação necessária quando deseja-se realizar uma intersecção com uma coluna 'col' que não se chama 'geometry'

    A coluna Geometry é um atributo importante de GeoDataFrame podendo conter apenas 1 deste atributo.
    Caso tenha um GeoDataFrame com mais de um 'geometry' é necessário deletar o 'geometry' original
    e renomear o outro geometry para 'geometry'
    '''
    return gdf.drop(columns=["geometry"]).rename(columns={col: "geometry"})


def get_counting(gdf: GeoDataFrame, gdf_ref: GeoDataFrame, buffer: int, col_ref: str, idx: list, col_return: str = "count") -> ndarray:
    '''
    Dado um 'gdf' contendo POINTS, cria-se um buffer cuja intersecção com 'gdf_ref'
    realiza-se contagem
    '''
    gdf["buffer_geo"] = gdf.buffer(buffer)
    # Intersecção entre o buffer
    gpd_join = gpd.sjoin(geo_prep_to_join(gdf, "buffer_geo"),
                         gdf_ref, how="left", op='intersects')
    # Os pontos não intersecionados serão substituidos por 'NOTFOUND'
    gpd_join[col_ref] = gpd_join[col_ref].fillna('NOTFOUND')
    # Substituimos o que não foi intersecionado por 0 e caso contrário 1
    gpd_join[col_return] = gpd_join[col_ref].apply(
        lambda x: 1 if x != 'NOTFOUND' else 0)
    # Criação do DataFrame com a contagem
    df_join_aux = pd.DataFrame(gpd_join.groupby(idx)[col_return].sum())
    # Joins entre o DataFrame auxiliar criado com a contagem com 'gdf' original
    return gdf.set_index(idx).merge(df_join_aux, on=idx)[col_return].values


def prep_ciclo_shp(gdf: GeoDataFrame, distance_to_consider: float, col_str: str = "Descriptio") -> MultiLineString:
    '''Preparação do shapefile da ciclovia

    Criamos a coluna 'lenght_km' a partir de uma regex da descrição da ciclovia
    Filtramos considerando uma certo 'lenght_km' que julgamos ideal
    Por fim, criamos uma união de todas as linha e retornamos um MultiLineString.
    '''
    pattern = r'Extensão: ([0-9]{1,2},*[0-9]{1,2}?)'
    gdf["lenght_km"] = gdf[col_str].str.replace("km", ",0").str.extract(pattern)[0].str.replace(",", ".").astype(float)
    gdf = gdf[gdf["lenght_km"] > distance_to_consider]
    return unary_union(gdf["geometry"])

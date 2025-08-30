'''
Funções Uteis
'''

from typing import Dict, Any

import geopandas as gpd
import pandas as pd
from numpy import ndarray
from shapely.geometry import Point
from shapely.geometry.multilinestring import MultiLineString
from shapely.ops import nearest_points, unary_union

GEO_DOM = ["Coordenada X Domicílio", "Coordenada Y Domicílio"]
GEO_ORI = ["Coordenada X Origem", "Coordenada Y Origem"]
GEO_DES = ["Coordenada X Destino", "Coordenada Y Destino"]

GEO_DOM_17 = ["Coordenada X domicílio", "Coordenada Y domicílio"]
GEO_ORI_17 = ["Coordenada X Origem", "Coordenada Y Origem"]
GEO_DES_17 = ["Coordenada X Destino", "Coordenada Y Destino"]

GEO_DOM_23 = ["coordenada_x_do_domicilio", "coordenada_y_do_domicilio"]
GEO_ORI_23 = ["coordenada_x_origem", "coordenada_y_origem"]
GEO_DES_23 = ["coordenada_x_destino", "coordenada_y_destino"]

DICT_GIS_07 = {
    "loc_domicilio": GEO_DOM,
    "loc_origem": GEO_ORI,
    "loc_destino": GEO_DES
}
DICT_GIS_17 = {
    "loc_domicilio": GEO_DOM_17,
    "loc_origem": GEO_ORI_17,
    "loc_destino": GEO_DES_17
}
DICT_GIS_23 = {
    "loc_domicilio": GEO_DOM_23,
    "loc_origem": GEO_ORI_23,
    "loc_destino": GEO_DES_23
}

DICT_GIS = DICT_GIS_23


def create_gis_point(df_raw: pd.DataFrame, dict_gis: Dict) -> gpd.GeoDataFrame:
    df_raw.dropna(subset=dict_gis["loc_domicilio"], inplace=True)
    for k, v in dict_gis.items():
        df_raw[k] = gpd.points_from_xy(df_raw[v[0]], df_raw[v[1]])
    if df_raw.dtypes[df_raw.dtypes == 'geometry'].shape[0] > 1:
        gdf = gpd.GeoDataFrame(df_raw,  geometry='loc_domicilio', crs = "EPSG:22523")
    else:
        gdf = gpd.GeoDataFrame(df_raw, crs = "EPSG:22523")
    return gdf


def nearest_point_value(point: Point, df_ref: gpd.GeoDataFrame, col_pts: str = "geometry", col_return: str = "geometry") -> Any:
    """Retorna a coluna/valor mais próximo de um ponto de um geopandas de referência
    """
    pts_ref = df_ref[col_pts].unary_union
    nearest = df_ref[df_ref.geometry == nearest_points(point, pts_ref)[1]]
    return nearest[col_return].values[0]


def geo_prep_to_join(gdf: gpd.GeoDataFrame, col: str) -> gpd.GeoDataFrame:
    '''Preparação necessária quando deseja-se realizar uma intersecção com uma coluna 'col' que não se chama 'geometry'

    A coluna Geometry é um atributo importante de gpd.GeoDataFrame podendo conter apenas 1 deste atributo.
    Caso tenha um gpd.GeoDataFrame com mais de um 'geometry' é necessário deletar o 'geometry' original
    e renomear o outro geometry para 'geometry'
    '''
    try:
        return gdf.drop(columns=["geometry"]).rename(columns={col: "geometry"})
    except KeyError:
        return gdf.rename(columns={col: "geometry"})


def get_counting(gdf: gpd.GeoDataFrame, gdf_ref: gpd.GeoDataFrame, geocol: str, buffer: int, col_ref: str, idx: list, col_return: str = "count") -> ndarray:
    '''
    Dado um 'gdf' contendo POINTS, cria-se um buffer cuja intersecção com 'gdf_ref'
    realiza-se contagem
    '''
    gdf = geo_prep_to_join(gdf, geocol)
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


def prep_ciclo_shp(gdf: gpd.GeoDataFrame, distance_to_consider: float, col_str: str = "Descriptio") -> MultiLineString:
    '''Preparação do shapefile da ciclovia

    Criamos a coluna 'lenght_km' a partir de uma regex da descrição da ciclovia
    Filtramos considerando uma certo 'lenght_km' que julgamos ideal
    Por fim, criamos uma união de todas as linha e retornamos um MultiLineString.
    '''
    pattern = r'Extensão: ([0-9]{1,2},*[0-9]{1,2}?)'
    gdf["lenght_km"] = gdf[col_str].str.replace("km", ",0").str.extract(pattern)[0].str.replace(",", ".").astype(float)
    gdf = gdf[gdf["lenght_km"] > distance_to_consider]
    return unary_union(gdf["geometry"])

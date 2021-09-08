'''
Autor: Rafael Yoshio Gomes Nomachi
Projeto: Dissertação de Mestrado

------
    Objetivo: Preparação da Base OD2017 afim de retornar
uma base de modelagem
'''

import pandas as pd
import pyreadstat
import yaml
from loguru import logger

from src.utils import convert_label
from pathlib import Path

# Log config
file_name = Path(__file__).stem
logger.add(f"logs/{file_name}.log")
# Visualization
pd.set_option('display.max_columns', None)

# Global Variables
FILE = "OD 97 Zona.sav"
FILE_TO = "OD_1997.parquet"
PATH_R_DATARAW = "data/raw/"
PATH_R_DATAPRC = "data/processed/"
PATH_R_CONF = "project/"
YML = "config.yml"
PATH_ROOT = "/home/yoshraf/projects/mestrado/"

COLS = [
    "Zona de Domicílio",
    "Município do Domicílio",
    "Coordenada X Domicílio",
    "Coordenada Y Domicílio",
    "Coordenada X Origem",
    "Coordenada Y Origem",
    "Coordenada X Destino",
    "Coordenada Y Destino",
    "Identifica Domicílo",
    "Número do Domicílio",
    "Data da Entrevista",
    "Número de Moradores no Domicílio",
    "Quantidade Automóvel",
    "Renda Familiar Mensal",
    "Identifica Pessoa",
    "Situação Familiar",
    "Idade",
    "Gênero",
    "Valor da Renda Individual",
    "Número da Viagem",
    "Total de Viagens internas",
    "Motivo na Origem",
    "Motivo no Destino",
    "Modo Principal"
]

# Read OD2017
df_sav, meta = pyreadstat.read_sav(f"{PATH_ROOT}{PATH_R_DATARAW}{FILE}")
# Read config
with open(f"{PATH_ROOT}{PATH_R_CONF}{YML}", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader, )

df_raw = convert_label(df_sav, meta)

df_raw = df_raw[COLS]
df_raw.to_parquet(f"{PATH_R_DATAPRC}{FILE_TO}")
logger.debug(f"Arquivo {FILE_TO} salvo em {PATH_R_DATAPRC}")
logger.debug("Tudo ocorreu bem!")

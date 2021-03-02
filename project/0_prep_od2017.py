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
FILE = "OD_2017.sav"
FILE_TO = "OD_2017.parquet"
PATH_R_DATARAW = "data/raw/"
PATH_R_DATAPRC = "data/processed/"
PATH_R_CONF = "project/"
YML = "config.yml"

# Read OD2017
df_sav, meta = pyreadstat.read_sav(f"{PATH_R_DATARAW}{FILE}")
# Read config
with open(f"{PATH_R_CONF}{YML}", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader, )

df_raw = convert_label(df_sav, meta)

SELECTED_COLS = config["DATAPREP"]["SELECT_COLUMNS"]
df_raw = df_raw[SELECTED_COLS]
df_raw.to_parquet(f"{PATH_R_DATAPRC}{FILE_TO}")
logger.debug(f"Arquivo {FILE_TO} salvo em {PATH_R_DATAPRC}")
logger.debug("Tudo ocorreu bem!")

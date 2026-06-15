from pathlib import Path
from datetime import date

import pandas as pd
from vnstock.api.quote import Quote
from vnstock import Vnstock

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE = "VCI"

MIN_DATE = "2010-01-01"

START_DATE = MIN_DATE

END_DATE = str(date.today())
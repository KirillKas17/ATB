import pandas as pd
from typing import Any


def load_market_data(filepath: str) -> pd.DataFrame:
    # Проверяем наличие метода read_csv
    if not hasattr(pd, 'read_csv'):
        raise AttributeError("pandas не поддерживает метод read_csv")
    return pd.read_csv(filepath)


def save_market_data(data: pd.DataFrame, filepath: str) -> None:
    # Проверяем наличие метода to_csv
    if not hasattr(data, 'to_csv'):
        raise AttributeError("DataFrame не поддерживает метод to_csv")
    data.to_csv(filepath)

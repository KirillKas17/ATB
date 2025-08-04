import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict


def calculate_imbalance(data: pd.DataFrame) -> pd.Series:
    buy_volume = data[data["side"] == "buy"]["volume"].sum()
    sell_volume = data[data["side"] == "sell"]["volume"].sum()
    total = buy_volume + sell_volume
    if total == 0:
        return pd.Series([0])
    return pd.Series([(buy_volume - sell_volume) / total])


def calculate_volume_profile(data: pd.DataFrame, bins: int = 24) -> Dict[str, Any]:
    # Проверяем наличие необходимых методов pandas
    if not hasattr(pd, 'cut') or not hasattr(data, 'groupby'):
        raise AttributeError("pandas DataFrame не поддерживает необходимые методы")
    
    hist, bin_edges = (
        pd.cut(data["price"], bins, retbins=True, labels=False),
        pd.cut(data["price"], bins, retbins=True)[1],
    )
    volume_profile = data.groupby(hist)["volume"].sum()
    return {
        "histogram": volume_profile.tolist(),
        "bins": bin_edges.tolist(),
    }

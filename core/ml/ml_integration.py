from typing import Any, Dict, Optional, Protocol, Union

import numpy as np
import pandas as pd


class ModelProtocol(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...


class MLIntegration:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model: Optional[ModelProtocol] = None

    def _process_value(self, value: Union[float, np.ndarray, pd.Series]) -> float:
        """
        Обработка значения.

        Args:
            value: Входное значение

        Returns:
            float: Обработанное значение
        """
        if isinstance(value, float):
            return float(value)
        elif isinstance(value, np.ndarray):
            return float(pd.Series(value).iloc[0])
        elif isinstance(value, pd.Series):
            return float(value.iloc[0])
        return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Обучение модели.

        Args:
            X: Признаки
            y: Целевая переменная
        """
        if self.model is not None:
            self.model.fit(X, y)

"""
Доменный сервис для расчета технических индикаторов.
"""

from typing import Dict, List

import pandas as pd

from domain.entities.market import MarketData


class TechnicalIndicatorsService:
    """Сервис для расчета технических индикаторов."""

    def calculate_indicators(
        self, market_data: List[MarketData], indicators: List[str]
    ) -> Dict[str, List[float]]:
        """Расчет технических индикаторов."""
        if not market_data:
            return {}
        # Преобразуем в pandas DataFrame для вычисления индикаторов
        df = pd.DataFrame(
            [
                {
                    "timestamp": data.timestamp,
                    "open": data.open_price,
                    "high": data.high_price,
                    "low": data.low_price,
                    "close": data.close_price,
                    "volume": data.volume,
                }
                for data in market_data
            ]
        )
        result = {}
        for indicator in indicators:
            if indicator == "sma_20":
                result[indicator] = df["close"].rolling(window=20).mean().tolist()
            elif indicator == "sma_50":
                result[indicator] = df["close"].rolling(window=50).mean().tolist()
            elif indicator == "rsi":
                # Простой RSI
                delta = df["close"].diff()
                gain = (delta.where(pd.to_numeric(delta) > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(pd.to_numeric(delta) < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                result[indicator] = (100 - (100 / (1 + rs))).tolist()
            elif indicator == "bollinger_upper":
                sma = df["close"].rolling(window=20).mean()
                std = df["close"].rolling(window=20).std()
                result[indicator] = (sma + (std * 2)).tolist()
            elif indicator == "bollinger_lower":
                sma = df["close"].rolling(window=20).mean()
                std = df["close"].rolling(window=20).std()
                result[indicator] = (sma - (std * 2)).tolist()
        return result

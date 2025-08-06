"""
Сервис для технического анализа.
"""

import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List

from domain.exceptions import DomainError, ExchangeError
from domain.repositories.market_repository import MarketRepository
from domain.services.technical_analysis import TechnicalAnalysisService as DomainTechnicalAnalysisService


class TechnicalAnalysisService:
    """Сервис для технического анализа."""

    def __init__(self, *args, **kwargs) -> Any:
        self.market_repository = market_repository
        self.technical_analysis_service = DomainTechnicalAnalysisService()

    async def get_technical_indicators(
        self, symbol: str, timeframe: str, indicators: List[str], limit: int = 100
    ) -> Dict[str, List[float]]:
        """Получение технических индикаторов."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=limit)
            market_data = await self.market_repository.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )
            if not market_data:
                raise DomainError(f"No market data found for {symbol}")
            # Преобразуем в pandas DataFrame
            df_data = []
            for data in market_data:
                try:
                    df_data.append(
                        {
                            "timestamp": data.timestamp,
                            "open": self._extract_numeric_value(data.open_price),
                            "high": self._extract_numeric_value(data.high_price),
                            "low": self._extract_numeric_value(data.low_price),
                            "close": self._extract_numeric_value(data.close_price),
                            "volume": self._extract_numeric_value(data.volume),
                        }
                    )
                except Exception:
                    continue
            if not df_data:
                return {}  # type: Dict[str, Any]
            df = pd.DataFrame(df_data)
            # Используем domain-сервис для расчёта индикаторов
            analysis_result = self.technical_analysis_service.analyze_market_data(
                market_data, indicators
            )
            # Преобразуем результат в ожидаемый формат
            result = {}
            for indicator_name, indicator_data in analysis_result.indicators.items():
                if hasattr(indicator_data, 'value'):
                    result[indicator_name] = [indicator_data.value]
                elif hasattr(indicator_data, 'values'):
                    result[indicator_name] = indicator_data.values
                elif hasattr(indicator_data, '__float__'):
                    result[indicator_name] = [float(indicator_data)]
                else:
                    result[indicator_name] = [0.0]
            return result
        except Exception as e:
            raise ExchangeError(f"Error getting technical indicators: {str(e)}")

    def _extract_numeric_value(self, value_obj: Any) -> float:
        """Безопасное извлечение числового значения из value object."""
        try:
            if hasattr(value_obj, "amount"):
                return float(value_obj.amount)
            elif hasattr(value_obj, "value"):
                return float(value_obj.value)
            elif hasattr(value_obj, "__float__"):
                return float(value_obj)
            elif isinstance(value_obj, (int, float, Decimal)):
                return float(value_obj)
            else:
                return 0.0
        except (ValueError, TypeError, AttributeError):
            return 0.0

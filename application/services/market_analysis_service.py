"""
Сервис для анализа рынка.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List

from domain.exceptions import DomainError, ExchangeError
from domain.repositories.market_repository import MarketRepository
from domain.services.market_analysis import MarketAnalysisService as DomainMarketAnalysisService


class MarketAnalysisService:
    """Сервис для анализа рынка."""

    def __init__(self, *args, **kwargs) -> Any:
        self.market_repository = market_repository
        self.domain_market_analysis_service = DomainMarketAnalysisService()

    async def get_market_summary(
        self, symbol: str, timeframe: str = "1h"
    ) -> Dict[str, Any]:
        """Получение сводки рынка."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=100)
            latest_data = await self.market_repository.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                limit=100,
            )
            if not latest_data:
                raise DomainError(f"No market data found for {symbol}")
            # Используем domain-сервис для расчёта сводки рынка
            summary = self.domain_market_analysis_service.calculate_market_summary(
                latest_data, symbol, timeframe
            )
            # Преобразуем в ожидаемый формат
            return {
                "symbol": summary.symbol,
                "timeframe": summary.timeframe,
                "current_price": summary.current_price,
                "price_change_24h": summary.price_change_24h,
                "price_change_percent": summary.price_change_percent,
                "volume_24h": summary.volume_24h,
                "high_24h": summary.high_24h,
                "low_24h": summary.low_24h,
                "volatility": summary.volatility,
                "trend_direction": summary.trend_direction,
                "support_levels": summary.support_levels,
                "resistance_levels": summary.resistance_levels,
                "timestamp": summary.timestamp.isoformat(),
            }
        except Exception as e:
            raise ExchangeError(f"Error getting market summary: {str(e)}")

    async def get_volume_profile(
        self, symbol: str, timeframe: str = "1h", days: int = 30
    ) -> Dict[str, Any]:
        """Получение профиля объема."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            market_data = await self.market_repository.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
            )
            if not market_data:
                return {}
            # Собираем данные
            price_volume_data = []
            for data in market_data:
                try:
                    price = self._extract_numeric_value(data.close_price)
                    volume = self._extract_numeric_value(data.volume)
                    price_volume_data.append((price, volume))
                except Exception:
                    continue
            if not price_volume_data:
                return {}
            # Используем domain-сервис для расчёта профиля объёма
            volume_profile_result = (
                self.domain_market_analysis_service.calculate_volume_profile(
                    market_data, symbol, timeframe
                )
            )
            return {
                "symbol": volume_profile_result.symbol,
                "timeframe": volume_profile_result.timeframe,
                "poc_price": volume_profile_result.poc_price,
                "total_volume": volume_profile_result.total_volume,
                "volume_profile": volume_profile_result.volume_profile,
                "price_range": volume_profile_result.price_range,
                "timestamp": volume_profile_result.timestamp.isoformat(),
            }
        except Exception as e:
            raise ExchangeError(f"Error getting volume profile: {str(e)}")

    async def get_market_regime_analysis(
        self, symbol: str, timeframe: str = "1h", days: int = 30
    ) -> Dict[str, Any]:
        """Анализ рыночного режима."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            market_data = await self.market_repository.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
            )
            if not market_data:
                return {}
            # Извлекаем данные
            prices = []
            volumes = []
            for data in market_data:
                try:
                    price = self._extract_numeric_value(data.close_price)
                    volume = self._extract_numeric_value(data.volume)
                    prices.append(price)
                    volumes.append(volume)
                except Exception:
                    continue
            if len(prices) < 20:
                return {}
            # Используем domain-сервис для анализа рыночного режима
            market_regime_result = (
                self.domain_market_analysis_service.calculate_market_regime(
                    market_data, symbol, timeframe
                )
            )
            return {
                "symbol": market_regime_result.symbol,
                "timeframe": market_regime_result.timeframe,
                "regime": market_regime_result.regime,
                "volatility": market_regime_result.volatility,
                "trend_strength": market_regime_result.trend_strength,
                "price_trend": market_regime_result.price_trend,
                "volume_trend": market_regime_result.volume_trend,
                "confidence": market_regime_result.confidence,
                "timestamp": market_regime_result.timestamp.isoformat(),
            }
        except Exception as e:
            raise ExchangeError(f"Error getting market regime analysis: {str(e)}")

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

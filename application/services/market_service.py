"""
Сервис для работы с рынком.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from domain.entities.market import MarketData, MarketState
from domain.exceptions import ExchangeError
from domain.repositories.market_repository import MarketRepository
from .technical_analysis_service import TechnicalAnalysisService


class MarketService:
    """Сервис для работы с рынком."""

    def __init__(
        self,
        market_repository: MarketRepository,
        technical_analysis_service: TechnicalAnalysisService,
    ):
        self.market_repository = market_repository
        self.technical_analysis_service = technical_analysis_service

    async def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[MarketData]:
        """Получение рыночных данных."""
        try:
            return await self.market_repository.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )
        except Exception as e:
            raise ExchangeError(f"Error getting market data: {str(e)}")

    async def get_market_summary(
        self, symbol: str, timeframe: str = "1h"
    ) -> Dict[str, Any]:
        """Получение сводки рынка."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)
            market_data = await self.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                limit=168,  # 7 дней * 24 часа
            )
            if not market_data:
                return {}
            # Анализируем данные
            latest_data = market_data[-1]
            oldest_data = market_data[0]
            # Рассчитываем изменения
            price_change = (
                self._extract_numeric_value(latest_data.close_price)
                - self._extract_numeric_value(oldest_data.close_price)
            )
            price_change_percent = (
                price_change
                / self._extract_numeric_value(oldest_data.close_price)
                * 100
            )
            # Находим максимум и минимум
            high_prices = [
                self._extract_numeric_value(data.high_price) for data in market_data
            ]
            low_prices = [
                self._extract_numeric_value(data.low_price) for data in market_data
            ]
            volumes = [
                self._extract_numeric_value(data.volume) for data in market_data
            ]
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": self._extract_numeric_value(latest_data.close_price),
                "price_change_24h": price_change,
                "price_change_percent": price_change_percent,
                "volume_24h": sum(volumes),
                "high_24h": max(high_prices),
                "low_24h": min(low_prices),
                "volatility": (max(high_prices) - min(low_prices))
                / self._extract_numeric_value(latest_data.close_price)
                * 100,
                "trend_direction": "up" if price_change > 0 else "down",
                "support_levels": [min(low_prices)],
                "resistance_levels": [max(high_prices)],
                "timestamp": latest_data.timestamp.isoformat(),
            }
        except Exception as e:
            raise ExchangeError(f"Error getting market summary: {str(e)}")

    async def get_technical_indicators(
        self, symbol: str, timeframe: str, indicators: List[str]
    ) -> Dict[str, List[float]]:
        """Получение технических индикаторов."""
        try:
            return await self.technical_analysis_service.get_technical_indicators(
                symbol, timeframe, indicators, limit=100
            )
        except Exception as e:
            raise ExchangeError(f"Error getting technical indicators: {str(e)}")

    async def get_market_alerts(
        self, symbol: str, alert_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Получение рыночных алертов."""
        try:
            if alert_types is None:
                alert_types = []
            # В реальной системе здесь была бы логика анализа алертов
            alerts = []
            # Проверяем волатильность
            market_summary = await self.get_market_summary(symbol)
            if market_summary.get("volatility", 0) > 10:
                alerts.append(
                    {
                        "type": "high_volatility",
                        "message": f"High volatility detected: {market_summary['volatility']:.2f}%",
                        "severity": "warning",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            # Проверяем объем
            if market_summary.get("volume_24h", 0) > 1000000:
                alerts.append(
                    {
                        "type": "high_volume",
                        "message": f"High volume detected: {market_summary['volume_24h']:,.0f}",
                        "severity": "info",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            return alerts
        except Exception as e:
            raise ExchangeError(f"Error getting market alerts: {str(e)}")

    async def get_market_depth(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """Получение глубины рынка."""
        try:
            # В реальной системе здесь был бы запрос к бирже
            return {
                "symbol": symbol,
                "bids": [
                    {"price": 100.0 - i * 0.1, "quantity": 1000 - i * 50}
                    for i in range(depth)
                ],
                "asks": [
                    {"price": 100.0 + i * 0.1, "quantity": 1000 - i * 50}
                    for i in range(depth)
                ],
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            raise ExchangeError(f"Error getting market depth: {str(e)}")

    async def get_order_book(self, symbol: str) -> Dict[str, Any]:
        """Получение стакана заявок."""
        try:
            # В реальной системе здесь был бы запрос к бирже
            return {
                "symbol": symbol,
                "bids": [
                    {"price": 100.0 - i * 0.1, "quantity": 1000 - i * 50}
                    for i in range(20)
                ],
                "asks": [
                    {"price": 100.0 + i * 0.1, "quantity": 1000 - i * 50}
                    for i in range(20)
                ],
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            raise ExchangeError(f"Error getting order book: {str(e)}")

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Получение последних сделок."""
        try:
            # В реальной системе здесь был бы запрос к бирже
            trades = []
            base_price = 100.0
            for i in range(limit):
                trades.append(
                    {
                        "id": f"trade_{i}",
                        "price": base_price + (i % 10 - 5) * 0.1,
                        "quantity": 100 + (i % 50),
                        "side": "buy" if i % 2 == 0 else "sell",
                        "timestamp": (
                            datetime.now() - timedelta(minutes=i)
                        ).isoformat(),
                    }
                )
            return trades
        except Exception as e:
            raise ExchangeError(f"Error getting recent trades: {str(e)}")

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

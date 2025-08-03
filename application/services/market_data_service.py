"""
Сервис для работы с рыночными данными.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from domain.entities.market import MarketData, MarketState
from domain.exceptions import ExchangeError
from domain.repositories.market_repository import MarketRepository


class MarketDataService:
    """Сервис для работы с рыночными данными."""

    def __init__(self, market_repository: MarketRepository):
        self.market_repository = market_repository
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 минут

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

    async def save_market_data(
        self, symbol: str, timeframe: str, market_data: MarketData
    ) -> MarketData:
        """Сохранение рыночных данных."""
        try:
            return await self.market_repository.save_market_data(market_data)
        except Exception as e:
            raise ExchangeError(f"Error saving market data: {str(e)}")

    async def save_market_data_batch(
        self, symbol: str, timeframe: str, market_data_list: List[MarketData]
    ) -> List[MarketData]:
        """Сохранение пакета рыночных данных."""
        try:
            return await self.market_repository.save_market_data_batch(market_data_list)
        except Exception as e:
            raise ExchangeError(f"Error saving market data batch: {str(e)}")

    async def get_market_state(self, symbol: str) -> Optional[MarketState]:
        """Получение состояния рынка."""
        try:
            return await self.market_repository.get_market_state(symbol)
        except Exception as e:
            raise ExchangeError(f"Error getting market state: {str(e)}")

    async def update_market_state(
        self, symbol: str, market_state: MarketState
    ) -> MarketState:
        """Обновление состояния рынка."""
        try:
            return await self.market_repository.save_market_state(market_state)
        except Exception as e:
            raise ExchangeError(f"Error updating market state: {str(e)}")

    async def get_volume_profile(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Получение профиля объема."""
        try:
            # В реальной системе здесь был бы запрос к репозиторию
            return {
                "symbol": symbol,
                "poc_price": "51000",
                "total_volume": "10000",
                "volume_profile": {"51000": "5000"}
            }
        except Exception as e:
            raise ExchangeError(f"Error getting volume profile: {str(e)}")

    async def get_market_regime_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Получение анализа рыночного режима."""
        try:
            # В реальной системе здесь был бы запрос к репозиторию
            return {
                "symbol": symbol,
                "regime": "trending",
                "volatility": "20.5",
                "trend_strength": "75.0"
            }
        except Exception as e:
            raise ExchangeError(f"Error getting market regime analysis: {str(e)}")

    async def get_available_symbols(self) -> List[str]:
        """Получение доступных символов."""
        try:
            return await self.market_repository.get_available_symbols()
        except Exception as e:
            raise ExchangeError(f"Error getting available symbols: {str(e)}")

    async def get_available_timeframes(self) -> List[str]:
        """Получение доступных таймфреймов."""
        try:
            return await self.market_repository.get_available_timeframes()
        except Exception as e:
            raise ExchangeError(f"Error getting available timeframes: {str(e)}")

    async def get_price_history(
        self, symbol: str, timeframe: str, days: int = 30
    ) -> List[dict]:
        """Получение истории цен."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            market_data = await self.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
            )
            result = []
            for data in market_data:
                try:
                    result.append(
                        {
                            "timestamp": data.timestamp.isoformat(),
                            "open": str(self._extract_numeric_value(data.open_price)),
                            "high": str(self._extract_numeric_value(data.high_price)),
                            "low": str(self._extract_numeric_value(data.low_price)),
                            "close": str(self._extract_numeric_value(data.close_price)),
                            "volume": str(self._extract_numeric_value(data.volume)),
                        }
                    )
                except Exception:
                    continue
            return result
        except Exception as e:
            raise ExchangeError(f"Error getting price history: {str(e)}")

    async def get_real_time_price(self, symbol: str) -> Optional[float]:
        """Получение цены в реальном времени."""
        try:
            market_data = await self.get_market_data(
                symbol=symbol, timeframe="1m", limit=1
            )
            if market_data:
                return self._extract_numeric_value(market_data[0].close_price)
            return None
        except Exception as e:
            raise ExchangeError(f"Error getting real-time price: {str(e)}")

    async def get_market_depth(self, symbol: str, depth: int = 10) -> dict:
        """Получение глубины рынка."""
        try:
            # В реальной системе здесь был бы запрос к бирже
            return {
                "symbol": symbol,
                "bids": [],
                "asks": [],
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            raise ExchangeError(f"Error getting market depth: {str(e)}")

    async def subscribe_to_updates(self, symbol: str, callback: Callable) -> bool:
        """Подписка на обновления рыночных данных."""
        try:
            # В реальной системе здесь была бы подписка на WebSocket
            return True
        except Exception as e:
            raise ExchangeError(f"Error subscribing to updates: {str(e)}")

    async def cleanup_old_data(
        self, symbol: str, timeframe: str, days_to_keep: int = 365
    ) -> int:
        """Очистка старых данных."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            return await self.market_repository.delete_old_data(
                symbol=symbol, timeframe=timeframe, before_date=cutoff_date
            )
        except Exception as e:
            raise ExchangeError(f"Error cleaning up old data: {str(e)}")

    def _extract_numeric_value(self, value_obj) -> float:
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

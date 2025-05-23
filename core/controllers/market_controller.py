from typing import Any, Dict, List, Optional

from loguru import logger

from ..models import MarketData
from .base import BaseController


class MarketController(BaseController):
    """Контроллер для работы с рынком"""

    def __init__(self, exchange):
        super().__init__()
        self.exchange = exchange
        self.market_data: Dict[str, List[MarketData]] = {}
        self.current_state = None

    async def get_ticker(self, pair: str) -> Dict[str, Any]:
        """
        Получение тикера.

        Args:
            pair: Торговая пара

        Returns:
            Dict[str, Any]: Данные тикера
        """
        try:
            return await self.exchange.fetch_ticker(pair)
        except Exception as e:
            logger.error(f"Error getting ticker for {pair}: {e}")
            raise

    async def get_ohlcv(
        self, pair: str, timeframe: str = "1m", limit: int = 100
    ) -> List[MarketData]:
        """
        Получение OHLCV данных.

        Args:
            pair: Торговая пара
            timeframe: Временной интервал
            limit: Количество свечей

        Returns:
            List[MarketData]: Список свечей
        """
        try:
            # Получение данных
            ohlcv = await self.exchange.fetch_ohlcv(pair, timeframe, limit=limit)

            # Преобразование в объекты
            result = []
            for candle in ohlcv:
                data = {
                    "timestamp": candle[0],
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                    "pair": pair,
                }
                result.append(MarketData.from_dict(data))

            # Сохранение
            self.market_data[pair] = result

            return result

        except Exception as e:
            logger.error(f"Error getting OHLCV for {pair}: {e}")
            raise

    async def get_order_book(self, pair: str, limit: int = 20) -> Dict[str, List]:
        """
        Получение книги ордеров.

        Args:
            pair: Торговая пара
            limit: Количество ордеров

        Returns:
            Dict[str, List]: Книга ордеров
        """
        try:
            return await self.exchange.fetch_order_book(pair, limit)
        except Exception as e:
            logger.error(f"Error getting order book for {pair}: {e}")
            raise

    async def get_trades(self, pair: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Получение сделок.

        Args:
            pair: Торговая пара
            limit: Количество сделок

        Returns:
            List[Dict[str, Any]]: Список сделок
        """
        try:
            return await self.exchange.fetch_trades(pair, limit=limit)
        except Exception as e:
            logger.error(f"Error getting trades for {pair}: {e}")
            raise

    def get_market_data(self, pair: str) -> Optional[List[MarketData]]:
        """
        Получение рыночных данных.

        Args:
            pair: Торговая пара

        Returns:
            Optional[List[MarketData]]: Рыночные данные или None
        """
        return self.market_data.get(pair)

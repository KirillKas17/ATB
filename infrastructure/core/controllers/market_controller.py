"""
Контроллер для работы с рыночными данными.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import asyncio
import logging

from domain.entities.market import MarketData
from application.services.market_service import MarketService
from infrastructure.core.exchange import Exchange

logger = logging.getLogger(__name__)


class MarketController:
    """Контроллер для управления рыночными данными."""

    def __init__(self, exchange: Exchange) -> None:
        """Инициализация контроллера.
        
        Args:
            exchange: Экземпляр биржи для получения данных
        """
        self.exchange = exchange
        self._market_service: Optional[MarketService] = None

    @property
    def market_service(self) -> MarketService:
        """Получение сервиса рынка."""
        if self._market_service is None:
            # Создаем заглушку для MarketService, так как он требует репозиторий
            # В реальной реализации здесь должна быть правильная инициализация
            self._market_service = MarketService(None, None)
        return self._market_service

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Получение тикера.
        
        Args:
            symbol: Торговая пара
            
        Returns:
            Данные тикера
        """
        try:
            ticker_data = self.exchange.get_ticker(symbol)
            if ticker_data:
                return {
                    "symbol": symbol,
                    "last": ticker_data.get("last", 0.0),
                    "bid": ticker_data.get("bid", 0.0),
                    "ask": ticker_data.get("ask", 0.0),
                    "volume": ticker_data.get("volume", 0.0),
                    "high": ticker_data.get("high", 0.0),
                    "low": ticker_data.get("low", 0.0),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Fallback к сервису
                return await self.market_service.get_market_summary(symbol)
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return {}

    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> List[MarketData]:
        """Получение OHLCV данных.
        
        Args:
            symbol: Торговая пара
            timeframe: Временной интервал
            limit: Количество свечей
            
        Returns:
            Список OHLCV данных
        """
        try:
            ohlcv_data = self.exchange.get_ohlcv(symbol, timeframe, limit=limit)
            market_data_list = []
            
            for candle in ohlcv_data:
                from domain.type_definitions import TimestampValue
                market_data = MarketData(
                    timestamp=TimestampValue(datetime.fromtimestamp(candle["timestamp"] / 1000)),
                    open=candle["open"],
                    high=candle["high"],
                    low=candle["low"],
                    close=candle["close"],
                    volume=candle["volume"],
                    symbol=symbol  # type: ignore[arg-type]
                )
                market_data_list.append(market_data)
                
            return market_data_list
        except Exception as e:
            logger.error(f"Error getting OHLCV for {symbol}: {e}")
            return []

    async def get_order_book(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Получение книги ордеров.
        
        Args:
            symbol: Торговая пара
            depth: Глубина стакана
            
        Returns:
            Данные стакана заявок
        """
        try:
            orderbook = self.exchange.get_orderbook(symbol, depth)
            if orderbook:
                return {
                    "symbol": symbol,
                    "bids": orderbook.get("bids", []),
                    "asks": orderbook.get("asks", []),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Fallback к сервису
                return await self.market_service.get_order_book(symbol)
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return {"symbol": symbol, "bids": [], "asks": [], "timestamp": datetime.now().isoformat()}

    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Получение последних сделок.
        
        Args:
            symbol: Торговая пара
            limit: Количество сделок
            
        Returns:
            Список сделок
        """
        try:
            trades = self.exchange.get_trades(symbol, limit)
            return [
                {
                    "id": trade.id,
                    "symbol": symbol,
                    "side": trade.side,
                    "price": trade.price.amount,
                    "amount": getattr(trade, 'quantity', getattr(trade, 'amount', 0)),
                    "timestamp": getattr(trade, 'timestamp', datetime.now()).isoformat()
                }
                for trade in trades
            ]
        except Exception as e:
            logger.error(f"Error getting trades for {symbol}: {e}")
            return []

    async def get_market_data(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> Dict[str, Any]:
        """Получение комплексных рыночных данных.
        
        Args:
            symbol: Торговая пара
            timeframe: Временной интервал
            limit: Количество данных
            
        Returns:
            Комплексные рыночные данные
        """
        try:
            # Получаем все данные параллельно
            ticker_task = asyncio.create_task(self.get_ticker(symbol))
            ohlcv_task = asyncio.create_task(self.get_ohlcv(symbol, timeframe, limit))
            orderbook_task = asyncio.create_task(self.get_order_book(symbol))
            trades_task = asyncio.create_task(self.get_trades(symbol, limit))
            
            ticker, ohlcv, orderbook, trades = await asyncio.gather(
                ticker_task, ohlcv_task, orderbook_task, trades_task,
                return_exceptions=True
            )
            
            return {
                "symbol": symbol,
                "ticker": ticker if not isinstance(ticker, Exception) else {},
                "ohlcv": ohlcv if not isinstance(ohlcv, Exception) else [],
                "orderbook": orderbook if not isinstance(orderbook, Exception) else {},
                "trades": trades if not isinstance(trades, Exception) else [],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {
                "symbol": symbol,
                "ticker": {},
                "ohlcv": [],
                "orderbook": {},
                "trades": [],
                "timestamp": datetime.now().isoformat()
            } 
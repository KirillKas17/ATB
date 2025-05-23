from datetime import datetime
from typing import Any, Dict, List, Optional

import ccxt
import pandas as pd
from loguru import logger

from core.models import Account, MarketData, Order, Position, Trade


class Exchange:
    """Базовый класс для работы с биржей."""

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация биржи.

        Args:
            config: Конфигурация биржи
        """
        self.config = config
        self.exchange = self._initialize_exchange()
        self.client: Optional[Any] = None
        self.markets: Dict[str, Any] = {}
        self.timeframes: Dict[str, str] = {}
        self.symbols: List[str] = []
        self.data_cache: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {"error_count": 0, "last_error": None}
        self.cache_ttl: int = config.get("cache_ttl", 60)  # TTL в секундах

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Инициализация биржи."""
        try:
            # Создание экземпляра биржи
            exchange_class = getattr(ccxt, self.config["exchange"])
            exchange = exchange_class(
                {
                    "apiKey": self.config["api_key"],
                    "secret": self.config["api_secret"],
                    "enableRateLimit": True,
                    "options": {
                        "defaultType": "future",
                        "testnet": self.config.get("testnet", True),
                    },
                }
            )

            # Проверка подключения
            exchange.load_markets()
            self.markets = exchange.markets
            self.timeframes = exchange.timeframes
            self.symbols = list(exchange.markets.keys())
            return exchange

        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            raise

    async def get_market_data(
        self, symbol: str, interval: str, limit: int = 100
    ) -> List[MarketData]:
        """Получение рыночных данных с кэшированием"""
        try:
            # Проверка кэша
            cache_key = f"{symbol}_{interval}"
            cached_data = self.data_cache.get(cache_key)

            if (
                cached_data
                and (datetime.now() - cached_data.timestamp).total_seconds()
                < self.cache_ttl
            ):
                return [cached_data]

            # Получение новых данных
            if not self.exchange:
                logger.error("Exchange not initialized")
                return []

            klines = self.exchange.fetch_ohlcv(symbol, interval, limit=limit)
            if not klines or not isinstance(klines, list):
                return []

            # Создание объекта данных
            market_data = MarketData(
                pair=symbol,
                timestamp=datetime.now(),
                open=float(klines[-1][1]),
                high=float(klines[-1][2]),
                low=float(klines[-1][3]),
                close=float(klines[-1][4]),
                volume=float(klines[-1][5]),
            )

            # Обновление кэша
            self.data_cache[cache_key] = market_data

            return [market_data]

        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["error_count"] += 1
            return []

    def _process_response(self, data: Optional[Dict]) -> Optional[Dict]:
        """Обработка ответа от API"""
        if not data or not isinstance(data, dict):
            return None
        return data

    def _get_value(self, result: Optional[Dict], key: str) -> Optional[Any]:
        """Безопасное получение значения из словаря"""
        if not result or not isinstance(result, dict):
            return None
        return result.get(key)

    async def get_position_risk(self, symbol: str) -> Optional[Dict]:
        """Получение риска позиции"""
        if not self.client:
            return None
        try:
            return await self.client.get_position_risk(symbol)
        except Exception as e:
            logger.error(f"Error getting position risk: {str(e)}")
            return None

    def create_order(
        self,
        pair: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Optional[Order]:
        """
        Создание ордера.

        Args:
            pair: Торговая пара
            order_type: Тип ордера (market, limit)
            side: Сторона (buy, sell)
            amount: Количество
            price: Цена (для limit ордеров)

        Returns:
            Optional[Order]: Созданный ордер или None в случае ошибки
        """
        try:
            if self.client is None:
                logger.error("Exchange client is not initialized")
                return None

            order_data = {
                "symbol": pair,
                "type": order_type,
                "side": side,
                "amount": amount,
            }

            if price is not None:
                order_data["price"] = price

            result = self.client.create_order(**order_data)
            return Order.from_exchange_data(result)

        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """
        Отмена ордера.

        Args:
            order_id: ID ордера

        Returns:
            bool: True если ордер успешно отменен
        """
        try:
            if self.client is None:
                logger.error("Exchange client is not initialized")
                return False

            self.client.cancel_order(order_id)
            return True

        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Получение информации об ордере.

        Args:
            order_id: ID ордера

        Returns:
            Optional[Order]: Информация об ордере или None в случае ошибки
        """
        try:
            if self.client is None:
                logger.error("Exchange client is not initialized")
                return None

            result = self.client.get_order(order_id)
            return Order.from_exchange_data(result)

        except Exception as e:
            logger.error(f"Error getting order: {e}")
            return None

    def get_open_orders(self, pair: Optional[str] = None) -> List[Order]:
        """
        Получение списка открытых ордеров.

        Args:
            pair: Торговая пара (опционально)

        Returns:
            List[Order]: Список открытых ордеров
        """
        try:
            if self.client is None:
                logger.error("Exchange client is not initialized")
                return []

            orders = (
                self.client.get_open_orders(symbol=pair)
                if pair
                else self.client.get_open_orders()
            )
            return [Order.from_exchange_data(order) for order in orders]

        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    def get_balance(self) -> Dict[str, float]:
        """
        Получение баланса.

        Returns:
            Dict[str, float]: Словарь с балансами
        """
        if not self.client:
            return {}
        try:
            # Получение баланса
            account = self.client.get_account()

            # Преобразование в словарь
            return {
                asset["asset"]: float(asset["free"])
                for asset in account["balances"]
                if float(asset["free"]) > 0
            }

        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {}

    def get_positions(self) -> List[Position]:
        """Получение текущих позиций."""
        try:
            if not self.exchange:
                return []

            positions = self.exchange.fetch_positions()
            if not positions or not isinstance(positions, list):
                return []

            result = []
            for pos in positions:
                if not isinstance(pos, dict):
                    continue
                if float(pos.get("contracts", 0)) != 0:
                    result.append(
                        Position(
                            pair=pos["symbol"],
                            side=pos["side"],
                            size=float(pos["contracts"]),
                            entry_price=float(pos["entryPrice"]),
                            entry_time=datetime.fromtimestamp(pos["timestamp"] / 1000),
                            current_price=float(pos["markPrice"]),
                            pnl=float(pos["unrealizedPnl"]),
                            stop_loss=None,
                            take_profit=None,
                        )
                    )
            return result

        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []

    def get_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Trade]:
        """Получение истории сделок"""
        if not self.client:
            return []
        try:
            if self.client is None:
                logger.error("Exchange client not initialized")
                return []

            trades = self.client.get_my_trades(symbol, limit)
            if trades is None:
                return []

            return [
                Trade(
                    id=str(trade.get("id", "")),
                    pair=str(trade.get("symbol", "")),
                    side=str(trade.get("side", "")),
                    size=float(trade.get("size", 0.0)),
                    price=float(trade.get("price", 0.0)),
                    timestamp=datetime.fromtimestamp(trade.get("timestamp", 0) / 1000),
                    fee=float(trade.get("fee", 0.0)),
                    pnl=float(trade.get("pnl")) if trade.get("pnl") else None,
                )
                for trade in trades
            ]
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []

    def get_account(self) -> Optional[Account]:
        """Получение информации об аккаунте"""
        try:
            if self.client is None:
                logger.error("Exchange client not initialized")
                return None

            account = self.client.get_account()
            if account is None:
                return None

            return Account(
                balance=float(account.get("balance", 0.0)),
                equity=float(account.get("equity", 0.0)),
                margin=float(account.get("margin", 0.0)),
                free_margin=float(account.get("free_margin", 0.0)),
                margin_level=float(account.get("margin_level", 0.0)),
                positions=[],
                orders=[],
                trades=[],
                timestamp=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return None

    def get_ticker(self, symbol: str) -> Optional[Dict[str, float]]:
        """Получение тикера"""
        try:
            if self.client is None:
                logger.error("Exchange client not initialized")
                return None

            ticker = self.client.get_ticker(symbol)
            if ticker is None:
                return None

            return {
                "bid": float(ticker.get("bid", 0.0)),
                "ask": float(ticker.get("ask", 0.0)),
                "last": float(ticker.get("last", 0.0)),
                "volume": float(ticker.get("volume", 0.0)),
                "high": float(ticker.get("high", 0.0)),
                "low": float(ticker.get("low", 0.0)),
            }
        except Exception as e:
            logger.error(f"Error getting ticker: {e}")
            return None

    def get_orderbook(
        self, symbol: str, limit: int = 100
    ) -> Optional[Dict[str, List[Dict[str, float]]]]:
        """Получение стакана"""
        if not self.client:
            return None
        try:
            orderbook = self.client.get_orderbook(symbol, limit)
            if orderbook is None:
                return None

            return {
                "bids": [
                    {"price": float(bid[0]), "amount": float(bid[1])}
                    for bid in orderbook.get("bids", [])
                ],
                "asks": [
                    {"price": float(ask[0]), "amount": float(ask[1])}
                    for ask in orderbook.get("asks", [])
                ],
            }
        except Exception as e:
            logger.error(f"Error getting orderbook: {e}")
            return None

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Получение OHLCV данных."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not ohlcv or not isinstance(ohlcv, list):
                return []
            result = []
            for candle in ohlcv:
                if not isinstance(candle, (list, tuple)) or len(candle) < 6:
                    continue
                result.append(
                    {
                        "timestamp": int(candle[0]),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                    }
                )
            return result

        except Exception as e:
            logger.error(f"Error getting OHLCV: {e}")
            return []

    def get_closed_orders(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Получение закрытых ордеров."""
        try:
            orders = self.exchange.fetch_closed_orders(symbol, since, limit)
            return [
                {
                    "id": order["id"],
                    "symbol": order["symbol"],
                    "type": order["type"],
                    "side": order["side"],
                    "amount": order["amount"],
                    "price": order["price"],
                    "status": order["status"],
                    "filled": order["filled"],
                    "cost": order["cost"],
                    "timestamp": order["timestamp"],
                }
                for order in orders
            ]

        except Exception as e:
            logger.error(f"Error getting closed orders: {e}")
            return []

    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Получение ставки финансирования."""
        try:
            funding = self.exchange.fetch_funding_rate(symbol)
            return {
                "symbol": funding["symbol"],
                "rate": funding["fundingRate"],
                "timestamp": funding["fundingTimestamp"],
                "datetime": funding["fundingDateTime"],
            }

        except Exception as e:
            logger.error(f"Error getting funding rate: {e}")
            return {}

    def get_leverage(self, symbol: str) -> int:
        """Получение текущего плеча."""
        try:
            if not self.exchange:
                return 1
            leverage = self.exchange.fetch_leverage(symbol)
            return int(leverage.get("leverage", 1))
        except Exception as e:
            logger.error(f"Error getting leverage: {e}")
            return 1

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Установка плеча."""
        try:
            self.exchange.set_leverage(leverage, symbol)
            return True

        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return False

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Получение информации о позиции."""
        try:
            position = self.exchange.fetch_position(symbol)
            if not position:
                return {}
            return {
                "symbol": position["symbol"],
                "size": float(position["contracts"]),
                "side": position["side"],
                "entry_price": float(position["entryPrice"]),
                "current_price": float(position["markPrice"]),
                "unrealized_pnl": float(position["unrealizedPnl"]),
                "liquidation_price": float(position["liquidationPrice"]),
                "leverage": float(position["leverage"]),
            }

        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return {}

    def get_markets(self) -> Dict[str, Any]:
        """Получение информации о рынках."""
        try:
            if not self.exchange:
                return {}

            markets = self.exchange.fetch_markets()
            if not markets or not isinstance(markets, list):
                return {}
            result = {}
            for market in markets:
                if not isinstance(market, dict) or "symbol" not in market:
                    continue
                result[market["symbol"]] = market
            return result

        except Exception as e:
            logger.error(f"Error getting markets: {str(e)}")
            return {}

    def get_timeframes(self) -> Dict[str, str]:
        """Получение доступных таймфреймов."""
        try:
            return self.exchange.timeframes

        except Exception as e:
            logger.error(f"Error getting timeframes: {e}")
            return {}

    def get_exchange_info(self) -> Dict[str, Any]:
        """Получение информации о бирже."""
        try:
            if not self.exchange:
                return {}
            return {
                "name": getattr(self.exchange, "name", ""),
                "url": getattr(self.exchange, "urls", {}).get("www", ""),
                "version": getattr(self.exchange, "version", ""),
                "timeframes": getattr(self.exchange, "timeframes", {}),
                "has": getattr(self.exchange, "has", {}),
            }
        except Exception as e:
            logger.error(f"Error getting exchange info: {e}")
            return {}

    def get_risk_metrics(self) -> Dict[str, float]:
        """Получение метрик риска"""
        try:
            if self.client is None:
                logger.error("Exchange client not initialized")
                return {}

            risk = self.client.get_position_risk()
            if risk is None:
                return {}

            return {
                "margin_ratio": float(risk.get("margin_ratio", 0.0)),
                "maintenance_margin": float(risk.get("maintenance_margin", 0.0)),
                "initial_margin": float(risk.get("initial_margin", 0.0)),
                "unrealized_pnl": float(risk.get("unrealized_pnl", 0.0)),
                "leverage": float(risk.get("leverage", 0.0)),
            }
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {}


class BinanceExchange(Exchange):
    """Класс для работы с биржей Binance."""

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация биржи Binance.

        Args:
            config: Конфигурация биржи
        """
        super().__init__(config)
        self.client = self._initialize_client()
        logger.info("Binance exchange initialized")

    def _initialize_client(self) -> Optional[Any]:
        """Инициализация клиента Binance."""
        try:
            # Здесь должна быть инициализация клиента Binance
            return None  # Временное решение
        except Exception as e:
            logger.error(f"Error initializing Binance client: {e}")
            return None

    def create_order(
        self,
        pair: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Optional[Order]:
        """
        Создание ордера.

        Args:
            pair: Торговая пара
            order_type: Тип ордера (market, limit)
            side: Сторона (buy, sell)
            amount: Количество
            price: Цена (для limit ордеров)

        Returns:
            Optional[Order]: Созданный ордер или None в случае ошибки
        """
        try:
            if self.client is None:
                logger.error("Exchange client is not initialized")
                return None

            order_data = {
                "symbol": pair,
                "type": order_type,
                "side": side,
                "amount": amount,
            }

            if price is not None:
                order_data["price"] = price

            result = self.client.create_order(**order_data)
            return Order.from_exchange_data(result)

        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """
        Отмена ордера.

        Args:
            order_id: ID ордера

        Returns:
            bool: True если ордер успешно отменен
        """
        try:
            if self.client is None:
                logger.error("Exchange client is not initialized")
                return False

            self.client.cancel_order(order_id)
            return True

        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Получение информации об ордере.

        Args:
            order_id: ID ордера

        Returns:
            Optional[Order]: Информация об ордере или None в случае ошибки
        """
        try:
            if self.client is None:
                logger.error("Exchange client is not initialized")
                return None

            result = self.client.get_order(order_id)
            return Order.from_exchange_data(result)

        except Exception as e:
            logger.error(f"Error getting order: {e}")
            return None

    def get_open_orders(self, pair: Optional[str] = None) -> List[Order]:
        """
        Получение списка открытых ордеров.

        Args:
            pair: Торговая пара (опционально)

        Returns:
            List[Order]: Список открытых ордеров
        """
        try:
            if self.client is None:
                logger.error("Exchange client is not initialized")
                return []

            orders = (
                self.client.get_open_orders(symbol=pair)
                if pair
                else self.client.get_open_orders()
            )
            return [Order.from_exchange_data(order) for order in orders]

        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    def get_balance(self) -> Dict[str, float]:
        """
        Получение баланса.

        Returns:
            Dict[str, float]: Словарь с балансами
        """
        if not self.client:
            return {}
        try:
            # Получение баланса
            account = self.client.get_account()

            # Преобразование в словарь
            return {
                asset["asset"]: float(asset["free"])
                for asset in account["balances"]
                if float(asset["free"]) > 0
            }

        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {}

    def get_positions(self) -> List[Position]:
        """Получение текущих позиций."""
        try:
            if not self.exchange:
                return []

            positions = self.exchange.fetch_positions()
            if not positions or not isinstance(positions, list):
                return []

            result = []
            for pos in positions:
                if not isinstance(pos, dict):
                    continue
                if float(pos.get("contracts", 0)) != 0:
                    result.append(
                        Position(
                            pair=pos["symbol"],
                            side=pos["side"],
                            size=float(pos["contracts"]),
                            entry_price=float(pos["entryPrice"]),
                            entry_time=datetime.fromtimestamp(pos["timestamp"] / 1000),
                            current_price=float(pos["markPrice"]),
                            pnl=float(pos["unrealizedPnl"]),
                            stop_loss=None,
                            take_profit=None,
                        )
                    )
            return result

        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []

    def get_trades(self, pair: Optional[str] = None, limit: int = 1000) -> List[Trade]:
        """
        Получение истории сделок.

        Args:
            pair: Торговая пара (опционально)
            limit: Количество сделок

        Returns:
            List[Trade]: Список сделок
        """
        if not self.client:
            return []
        try:
            # Получение сделок
            trades = (
                self.client.get_my_trades(symbol=pair)
                if pair
                else self.client.get_my_trades()
            )

            # Преобразование в Trade
            return [
                Trade(
                    id=str(trade["id"]),
                    pair=trade["symbol"],
                    side="buy" if trade["isBuyer"] else "sell",
                    size=float(trade["qty"]),
                    price=float(trade["price"]),
                    timestamp=pd.to_datetime(trade["time"], unit="ms"),
                    fee=float(trade["commission"]),
                    pnl=None,  # P&L не предоставляется API
                )
                for trade in trades[:limit]
            ]

        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, List[float]]:
        if not self.client:
            return {}
        try:
            orderbook = self.client.get_orderbook(symbol, limit)
            if orderbook is None:
                return {}

            return {
                "bids": [float(bid[0]) for bid in orderbook.get("bids", [])],
                "asks": [float(ask[0]) for ask in orderbook.get("asks", [])],
            }
        except Exception as e:
            logger.error(f"Error getting orderbook: {e}")
            return {}

    def get_markets(self) -> Dict[str, Any]:
        """Получение информации о рынках."""
        try:
            if not self.exchange:
                return {}

            markets = self.exchange.fetch_markets()
            if not markets or not isinstance(markets, list):
                return {}
            result = {}
            for market in markets:
                if not isinstance(market, dict) or "symbol" not in market:
                    continue
                result[market["symbol"]] = market
            return result

        except Exception as e:
            logger.error(f"Error getting markets: {str(e)}")
            return {}

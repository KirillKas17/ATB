# -*- coding: utf-8 -*-
"""
Модуль для работы с биржами.
"""

import asyncio
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

import ccxt
import pandas as pd

from domain.entities.account import Account
from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.entities.position import Position, PositionSide
from domain.entities.trade import Trade
from domain.entities.trading_pair import TradingPair as TradingPairEntity
from domain.type_definitions import OrderId, PositionId, TradeId, TradingPair, PortfolioId
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.type_definitions import VolumeValue
from domain.value_objects.symbol import Symbol

logger = logging.getLogger(__name__)


class Exchange:
    """Базовый класс для работы с биржами."""

    def __init__(self, config: Dict[str, Any]):
        """Инициализация биржи.
        Args:
            config: Конфигурация биржи
        """
        self.config = config
        self.exchange = self._initialize_exchange()
        self.client = None
        self.logger = logging.getLogger(__name__)

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Инициализация биржи.
        Returns:
            ccxt.Exchange: Объект биржи
        """
        try:
            exchange_id = self.config.get("exchange_id", "binance")
            api_key = self.config.get("api_key")
            secret = self.config.get("secret")
            sandbox = self.config.get("sandbox", False)

            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                "apiKey": api_key,
                "secret": secret,
                "sandbox": sandbox,
                    "enableRateLimit": True,
            })

            return exchange
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            raise

    async def get_market_data(
        self, symbol: str, interval: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Получение рыночных данных.
        Args:
            symbol: Торговая пара
            interval: Интервал
            limit: Количество свечей
        Returns:
            List[Dict[str, Any]]: Список свечей
        """
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, interval, limit=limit)
            if not ohlcv:
                return []

            result = []
            for candle in ohlcv:
                result.append({
                    "timestamp": candle[0],
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                })
            return result
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return []

    def _process_response(self, data: Optional[Dict]) -> Optional[Dict]:
        """Обработка ответа от биржи.
        Args:
            data: Данные от биржи
        Returns:
            Optional[Dict]: Обработанные данные
        """
        if data is None:
            return None
        return data

    def _get_value(self, result: Optional[Dict], key: str) -> Optional[Any]:
        """Получение значения из результата.
        Args:
            result: Результат
            key: Ключ
        Returns:
            Optional[Any]: Значение
        """
        if result is None:
            return None
        return result.get(key)

    async def get_position_risk(self, symbol: str) -> Optional[Dict]:
        """Получение риска позиции.
        Args:
            symbol: Торговая пара
        Returns:
            Optional[Dict]: Риск позиции
        """
        try:
            position = await self.exchange.fetch_position(symbol)
            if position is None:
                return None
            return {
                "symbol": position.get("symbol"),
                "size": float(position.get("contracts", 0)),
                "notional": float(position.get("notional", 0)),
                "leverage": float(position.get("leverage", 1)),
                "unrealized_pnl": float(position.get("unrealizedPnl", 0)),
            }
        except Exception as e:
            logger.error(f"Error getting position risk: {e}")
            return None

    def create_order(
        self,
        pair: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Optional[Order]:
        """Создание ордера.
        Args:
            pair: Торговая пара
            order_type: Тип ордера
            side: Сторона (buy/sell)
            amount: Количество
            price: Цена (для лимитных ордеров)
        Returns:
            Optional[Order]: Созданный ордер
        """
        try:
            if self.client is None:
                logger.error("Exchange client is not initialized")
                return None

            result = self.client.create_order(
                symbol=pair,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
            )

            # Парсим символ для создания TradingPair
            base_currency, quote_currency = self._parse_symbol(pair)
            
            return Order(
                trading_pair=TradingPairEntity(
                    Symbol(base_currency),
                    Symbol(quote_currency)
                ),
                side=OrderSide.BUY if result.get("side") == "buy" else OrderSide.SELL,
                order_type=(
                    OrderType.MARKET
                    if result.get("type") == "market"
                    else OrderType.LIMIT
                ),
                quantity=VolumeValue(Decimal(str(result.get("amount", amount)))),
                price=(
                    Price(
                        Decimal(str(result.get("price", price or 0))),
                        Currency.USD,
                    )
                    if result.get("price")
                    else None
                ),
                status=(
                    OrderStatus.FILLED
                    if result.get("status") == "filled"
                    else OrderStatus.PENDING
                ),
            )
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return None

    def _parse_symbol(self, symbol: str) -> tuple[str, str]:
        """Парсинг символа торговой пары.
        Args:
            symbol: Символ торговой пары
        Returns:
            tuple[str, str]: Базовая и котируемая валюта
        """
        if "USDT" in symbol:
            return symbol.replace("USDT", ""), "USDT"
        elif "BTC" in symbol:
            return symbol.replace("BTC", ""), "BTC"
        else:
            return symbol[:3], symbol[3:]

    def cancel_order(self, order_id: str) -> bool:
        """Отмена ордера.
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
        """Получение информации об ордере.
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
            # Создание Order с правильными параметрами
            from domain.value_objects.currency import Currency
            from domain.value_objects.money import Money
            from domain.value_objects.price import Price
            from domain.value_objects.volume import Volume

            # Парсим символ для создания TradingPair
            base_currency, quote_currency = self._parse_symbol(result.get("symbol", ""))
            
            return Order(
                trading_pair=TradingPairEntity(
                    Currency(base_currency),
                    Currency(quote_currency)
                ),
                side=OrderSide.BUY if result.get("side") == "buy" else OrderSide.SELL,
                order_type=(
                    OrderType.MARKET
                    if result.get("type") == "market"
                    else OrderType.LIMIT
                ),
                quantity=VolumeValue(Decimal(str(result.get("amount", 0)))),
                price=(
                    Price(
                        Decimal(str(result.get("price", 0))),
                        Currency.USD,
                    )
                    if result.get("price")
                    else None
                ),
                status=(
                    OrderStatus.FILLED
                    if result.get("status") == "filled"
                    else OrderStatus.PENDING
                ),
            )
        except Exception as e:
            logger.error(f"Error getting order: {e}")
            return None

    def get_open_orders(self, pair: Optional[str] = None) -> List[Order]:
        """Получение списка открытых ордеров.
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
            # Создание списка Order с правильными параметрами
            from domain.value_objects.currency import Currency
            from domain.value_objects.money import Money
            from domain.value_objects.price import Price
            from domain.value_objects.volume import Volume

            result = []
            for order_data in orders:
                # Парсим символ для создания TradingPair
                base_currency, quote_currency = self._parse_symbol(order_data.get("symbol", ""))
                
                order = Order(
                    trading_pair=TradingPairEntity(
                        Currency(base_currency),
                        Currency(quote_currency)
                    ),
                    side=(
                        OrderSide.BUY
                        if order_data.get("side") == "buy"
                        else OrderSide.SELL
                    ),
                    order_type=(
                        OrderType.MARKET
                        if order_data.get("type") == "market"
                        else OrderType.LIMIT
                    ),
                    quantity=VolumeValue(Decimal(str(order_data.get("amount", 0)))),
                    price=(
                        Price(
                            Decimal(str(order_data.get("price", 0))),
                            Currency.USDT,
                        )
                        if order_data.get("price")
                        else None
                    ),
                    status=OrderStatus.PENDING,
                )
                result.append(order)
            return result
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    def get_balance(self) -> Dict[str, float]:
        """Получение баланса.
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
        """Получение текущих позиций.
        Returns:
            List[Position]: Список позиций.
        """
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
                    from domain.value_objects.currency import Currency
                    from domain.value_objects.money import Money
                    from domain.value_objects.price import Price
                    from domain.value_objects.volume import Volume

                    # Парсим символ для создания TradingPair
                    base_currency, quote_currency = self._parse_symbol(pos["symbol"])
                    
                    result.append(
                        Position(
                            id=PositionId(uuid4()),
                            portfolio_id=PortfolioId(uuid4()),
                            trading_pair=TradingPairEntity(
                                symbol=Symbol(f"{base_currency}{quote_currency}"),
                                base_currency=Currency(base_currency),
                                quote_currency=Currency(quote_currency)
                            ),
                            side=PositionSide.LONG if pos.get("side") == "long" else PositionSide.SHORT,
                            volume=Volume(Decimal(str(pos.get("contracts", 0)))),
                            entry_price=Price(
                                Decimal(str(pos.get("entryPrice", 0))),
                                Currency.USDT,
                            ),
                            current_price=Price(
                                Decimal(str(pos.get("markPrice", 0))),
                                Currency.USDT,
                            ),
                            unrealized_pnl=Money(
                                Decimal(str(pos.get("unrealizedPnl", 0))), Currency.USDT
                            ),
                        )
                    )
            return result
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []

    def get_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Trade]:
        """Получение истории сделок.
        Args:
            symbol: Торговая пара (опционально)
            limit: Количество сделок
        Returns:
            List[Trade]: Список сделок
        """
        try:
            if self.client is None:
                return []
            trades = self.client.get_my_trades(symbol=symbol, limit=limit)
            if not trades:
                return []
            result = []
            for trade_data in trades:
                # Парсим символ для создания TradingPair
                base_currency, quote_currency = self._parse_symbol(trade_data.get("symbol", ""))
                
                result.append(
                    Trade(
                        id=TradeId(uuid4()),
                        trading_pair=TradingPairEntity(
                            symbol=Symbol(f"{base_currency}{quote_currency}"),
                            base_currency=Currency(base_currency),
                            quote_currency=Currency(quote_currency)
                        ),
                        side=(
                            OrderSide.BUY
                            if trade_data.get("side") == "buy"
                            else OrderSide.SELL
                        ),
                        quantity=Volume(Decimal(str(trade_data.get("amount", 0)))),
                        price=Price(
                            Decimal(str(trade_data.get("price", 0))),
                            Currency.USD,
                        ),
                        commission=Money(
                            Decimal(str(trade_data.get("fee", {}).get("cost", 0))),
                            Currency.USD,
                        ),
                        timestamp=trade_data.get("timestamp", 0) / 1000,
                    )
                )
            return result
        except Exception as e:
            logger.error(f"Error getting trades: {str(e)}")
            return []

    def get_account(self) -> Optional[Account]:
        """Получение информации об аккаунте.
        Returns:
            Optional[Account]: Информация об аккаунте.
        """
        try:
            if self.client is None:
                logger.error("Exchange client not initialized")
                return None
            account = self.client.get_account()
            if account is None:
                return None
            return Account(
                balance=Money(Decimal(str(account.get("balance", 0.0))), Currency.USD),
                equity=Money(Decimal(str(account.get("equity", 0.0))), Currency.USD),
                margin=Money(Decimal(str(account.get("margin", 0.0))), Currency.USD),
                free_margin=Money(
                    Decimal(str(account.get("free_margin", 0.0))), Currency.USD
                ),
                margin_level=Decimal(str(account.get("margin_level", 0.0))),
            )
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return None

    def get_ticker(self, symbol: str) -> Optional[Dict[str, float]]:
        """Получение тикера.
        Args:
            symbol (str): Торговая пара.
        Returns:
            Optional[Dict[str, float]]: Данные тикера.
        """
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
        """Получение стакана.
        Args:
            symbol (str): Торговая пара.
            limit (int): Количество уровней.
        Returns:
            Optional[Dict[str, List[Dict[str, float]]]]: Данные стакана.
        """
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
        """Получение закрытых ордеров.
        Args:
            symbol (Optional[str]): Торговая пара.
            since (Optional[int]): Время начала.
            limit (Optional[int]): Количество ордеров.
        Returns:
            List[Dict[str, Any]]: Список закрытых ордеров.
        """
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
        """Получение ставки финансирования.
        Args:
            symbol (str): Торговая пара.
        Returns:
            Dict[str, Any]: Данные о ставке финансирования.
        """
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
        """Получение текущего плеча.
        Args:
            symbol (str): Торговая пара.
        Returns:
            int: Текущее плечо.
        """
        try:
            if not self.exchange:
                return 1
            position = self.exchange.fetch_position(symbol)
            if position is None:
                return 1
            return int(position.get("leverage", 1))
        except Exception as e:
            logger.error(f"Error getting leverage: {e}")
            return 1

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Установка плеча.
        Args:
            symbol (str): Торговая пара.
            leverage (int): Плечо.
        Returns:
            bool: True если плечо успешно установлено.
        """
        try:
            if not self.exchange:
                return False
            self.exchange.set_leverage(leverage, symbol)
            return True
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return False

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Получение позиции.
        Args:
            symbol (str): Торговая пара.
        Returns:
            Dict[str, Any]: Данные позиции.
        """
        try:
            if not self.exchange:
                return {}
            position = self.exchange.fetch_position(symbol)
            if position is None:
                return {}
            return {
                "symbol": position.get("symbol"),
                "size": float(position.get("contracts", 0)),
                "notional": float(position.get("notional", 0)),
                "leverage": float(position.get("leverage", 1)),
                "unrealized_pnl": float(position.get("unrealizedPnl", 0)),
                "entry_price": float(position.get("entryPrice", 0)),
                "mark_price": float(position.get("markPrice", 0)),
            }
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return {}

    def get_markets(self) -> Dict[str, Any]:
        """Получение рынков.
        Returns:
            Dict[str, Any]: Данные рынков
        """
        try:
            if not self.exchange:
                return {}
            markets = self.exchange.load_markets()
            return markets
        except Exception as e:
            logger.error(f"Error getting markets: {e}")
            return {}

    def get_timeframes(self) -> Dict[str, str]:
        """Получение временных фреймов.
        Returns:
            Dict[str, str]: Временные фреймы.
        """
        try:
            if not self.exchange:
                return {}
            timeframes = self.exchange.timeframes
            return timeframes
        except Exception as e:
            logger.error(f"Error getting timeframes: {e}")
            return {}

    def get_exchange_info(self) -> Dict[str, Any]:
        """Получение информации о бирже.
        Returns:
            Dict[str, Any]: Информация о бирже.
        """
        try:
            if not self.exchange:
                return {}
            info = {
                "id": self.exchange.id,
                "name": self.exchange.name,
                "urls": self.exchange.urls,
                "version": self.exchange.version,
                "timeframes": self.exchange.timeframes,
            }
            return info
        except Exception as e:
            logger.error(f"Error getting exchange info: {e}")
            return {}

    def get_risk_metrics(self) -> Dict[str, float]:
        """Получение метрик риска.
        Returns:
            Dict[str, float]: Метрики риска.
        """
        try:
            if not self.exchange:
                return {}
            # Здесь можно добавить расчет метрик риска
            return {
                "total_exposure": 0.0,
                "max_leverage": 1.0,
                "margin_ratio": 0.0,
            }
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {}


class BinanceExchange(Exchange):
    """Класс для работы с Binance."""

    def __init__(self, config: Dict[str, Any]):
        """Инициализация Binance.
        Args:
            config: Конфигурация
        """
        super().__init__(config)
        self.client = self._initialize_client()

    def _initialize_client(self) -> Optional[Any]:
        """Инициализация клиента Binance.
        Returns:
            Optional[Any]: Клиент Binance
        """
        try:
            # Временная замена для binance.client
            class BinanceClientMock:
                def __init__(self, api_key: str, secret: str, testnet: bool = False) -> None:
                    self.api_key = api_key
                    self.secret = secret
                    self.testnet = testnet
                
                def create_order(self, **kwargs: Any) -> Dict[str, Any]:
                    return {"id": "mock_order_id", "status": "FILLED"}
                
                def cancel_order(self, **kwargs: Any) -> Dict[str, Any]:
                    return {"status": "CANCELLED"}
                
                def get_order(self, **kwargs: Any) -> Dict[str, Any]:
                    return {"id": "mock_order_id", "status": "FILLED"}
                
                def get_open_orders(self, **kwargs: Any) -> List[Dict[str, Any]]:
                    return []
                
                def get_account(self) -> Dict[str, Any]:
                    return {"balances": []}
                
                def get_my_trades(self, **kwargs: Any) -> List[Dict[str, Any]]:
                    return []
                
                def get_ticker(self, **kwargs: Any) -> Dict[str, Any]:
                    return {"bid": 0.0, "ask": 0.0, "last": 0.0}
                
                def get_order_book(self, **kwargs: Any) -> Dict[str, Any]:
                    return {"bids": [], "asks": []}
                
                def get_exchange_info(self) -> Dict[str, Any]:
                    return {}

            api_key = self.config.get("api_key")
            secret = self.config.get("secret")
            testnet = self.config.get("sandbox", False)
            client = BinanceClientMock(api_key, secret, testnet=testnet)
            return client
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
        """Создание ордера на Binance.
        Args:
            pair: Торговая пара
            order_type: Тип ордера
            side: Сторона (buy/sell)
            amount: Количество
            price: Цена (для лимитных ордеров)
        Returns:
            Optional[Order]: Созданный ордер
        """
        try:
            if self.client is None:
                logger.error("Binance client is not initialized")
                return None

            result = self.client.create_order(
                symbol=pair,
                side=side,
                type=order_type,
                quantity=amount,
                price=price,
            )

            # Парсим символ для создания TradingPair
            base_currency, quote_currency = self._parse_symbol(pair)
            
            return Order(
                trading_pair=TradingPairEntity(
                    Symbol(base_currency),
                    Symbol(quote_currency)
                ),
                side=OrderSide.BUY if result.get("side") == "BUY" else OrderSide.SELL,
                order_type=(
                    OrderType.MARKET
                    if result.get("type") == "MARKET"
                    else OrderType.LIMIT
                ),
                quantity=VolumeValue(Decimal(str(result.get("executedQty", amount)))),
                price=(
                    Price(
                        Decimal(str(result.get("price", price or 0))),
                        Currency.USD,
                    )
                    if result.get("price")
                    else None
                ),
                status=(
                    OrderStatus.FILLED
                    if result.get("status") == "FILLED"
                    else OrderStatus.PENDING
                ),
            )
        except Exception as e:
            logger.error(f"Error creating Binance order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Отмена ордера на Binance.
        Args:
            order_id: ID ордера
        Returns:
            bool: True если ордер успешно отменен
        """
        try:
            if self.client is None:
                logger.error("Binance client is not initialized")
                return False
            self.client.cancel_order(orderId=order_id)
            return True
        except Exception as e:
            logger.error(f"Error canceling Binance order: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Получение информации об ордере на Binance.
        Args:
            order_id: ID ордера
        Returns:
            Optional[Order]: Информация об ордере или None в случае ошибки
        """
        try:
            if self.client is None:
                logger.error("Binance client is not initialized")
                return None
            result = self.client.get_order(orderId=order_id)
            # Создание Order с правильными параметрами
            from domain.value_objects.currency import Currency
            from domain.value_objects.money import Money
            from domain.value_objects.price import Price
            from domain.value_objects.volume import Volume

            # Парсим символ для создания TradingPair
            base_currency, quote_currency = self._parse_symbol(result.get("symbol", ""))
            
            return Order(
                trading_pair=TradingPairEntity(
                    Currency(base_currency),
                    Currency(quote_currency)
                ),
                side=OrderSide.BUY if result.get("side") == "BUY" else OrderSide.SELL,
                order_type=(
                    OrderType.MARKET
                    if result.get("type") == "MARKET"
                    else OrderType.LIMIT
                ),
                quantity=VolumeValue(Decimal(str(result.get("origQty", 0)))),
                price=(
                    Price(
                        Decimal(str(result.get("price", 0))),
                        Currency.USD,
                    )
                    if result.get("price")
                    else None
                ),
                status=(
                    OrderStatus.FILLED
                    if result.get("status") == "FILLED"
                    else OrderStatus.PENDING
                ),
            )
        except Exception as e:
            logger.error(f"Error getting Binance order: {e}")
            return None

    def get_open_orders(self, pair: Optional[str] = None) -> List[Order]:
        """Получение списка открытых ордеров на Binance.
        Args:
            pair: Торговая пара (опционально)
        Returns:
            List[Order]: Список открытых ордеров
        """
        try:
            if self.client is None:
                logger.error("Binance client is not initialized")
                return []
            orders = (
                self.client.get_open_orders(symbol=pair)
                if pair
                else self.client.get_open_orders()
            )
            # Создание списка Order с правильными параметрами
            from domain.value_objects.currency import Currency
            from domain.value_objects.money import Money
            from domain.value_objects.price import Price
            from domain.value_objects.volume import Volume

            result = []
            for order_data in orders:
                # Парсим символ для создания TradingPair
                base_currency, quote_currency = self._parse_symbol(order_data.get("symbol", ""))
                
                order = Order(
                    trading_pair=TradingPairEntity(
                        Symbol(base_currency),
                        Symbol(quote_currency)
                    ),
                    side=(
                        OrderSide.BUY
                        if order_data.get("side") == "BUY"
                        else OrderSide.SELL
                    ),
                    order_type=(
                        OrderType.MARKET
                        if order_data.get("type") == "MARKET"
                        else OrderType.LIMIT
                    ),
                    quantity=VolumeValue(Decimal(str(order_data.get("origQty", 0)))),
                    price=(
                        Price(
                            Decimal(str(order_data.get("price", 0))),
                            Currency.USDT,
                        )
                        if order_data.get("price")
                        else None
                    ),
                    status=OrderStatus.PENDING,
                )
                result.append(order)
            return result
        except Exception as e:
            logger.error(f"Error getting Binance open orders: {e}")
            return []

    def get_balance(self) -> Dict[str, float]:
        """Получение баланса на Binance.
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
            logger.error(f"Error getting Binance balance: {e}")
            return {}

    def get_positions(self) -> List[Position]:
        """Получение текущих позиций на Binance.
        Returns:
            List[Position]: Список позиций.
        """
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
                    from domain.value_objects.currency import Currency
                    from domain.value_objects.money import Money
                    from domain.value_objects.price import Price
                    from domain.value_objects.volume import Volume

                    # Парсим символ для создания TradingPair
                    base_currency, quote_currency = self._parse_symbol(pos["symbol"])
                    
                    result.append(
                        Position(
                            id=PositionId(uuid4()),
                            portfolio_id=PortfolioId(uuid4()),
                            trading_pair=TradingPairEntity(
                                symbol=Symbol(f"{base_currency}{quote_currency}"),
                                base_currency=Currency(base_currency),
                                quote_currency=Currency(quote_currency)
                            ),
                            side=PositionSide.LONG if pos.get("side") == "long" else PositionSide.SHORT,
                            volume=Volume(Decimal(str(pos.get("contracts", 0)))),
                            entry_price=Price(
                                Decimal(str(pos.get("entryPrice", 0))),
                                Currency.USDT,
                            ),
                            current_price=Price(
                                Decimal(str(pos.get("markPrice", 0))),
                                Currency.USDT,
                            ),
                            unrealized_pnl=Money(
                                Decimal(str(pos.get("unrealizedPnl", 0))), Currency.USDT
                            ),
                        )
                    )
            return result
        except Exception as e:
            logger.error(f"Error getting Binance positions: {str(e)}")
            return []

    def get_trades(self, pair: Optional[str] = None, limit: int = 1000) -> List[Trade]:
        """Получение истории сделок на Binance.
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
            result = []
            for trade in trades[:limit]:
                # Парсим символ для создания TradingPair
                base_currency, quote_currency = self._parse_symbol(trade["symbol"])
                
                result.append(
                    Trade(
                        id=TradeId(uuid4()),
                        trading_pair=TradingPairEntity(
                            symbol=Symbol(f"{base_currency}{quote_currency}"),
                            base_currency=Currency(base_currency),
                            quote_currency=Currency(quote_currency)
                        ),
                        side=(
                            OrderSide.BUY
                            if trade["isBuyer"] else OrderSide.SELL
                        ),
                        quantity=Volume(Decimal(str(trade["qty"]))),
                        price=Price(
                            Decimal(str(trade["price"])),
                            Currency.USDT,
                        ),
                        commission=Money(
                            Decimal(str(trade["commission"])), Currency.USDT
                        ),
                        timestamp=trade["time"] / 1000,
                    )
                )
            return result
        except Exception as e:
            logger.error(f"Error getting Binance trades: {e}")
            return []

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, List[float]]:
        """Получение стакана на Binance.
        Args:
            symbol: Торговая пара
            limit: Количество уровней
        Returns:
            Dict[str, List[float]]: Данные стакана
        """
        try:
            if self.client is None:
                return {"bids": [], "asks": []}
            order_book = self.client.get_order_book(symbol=symbol, limit=limit)
            return {
                "bids": [[float(bid[0]), float(bid[1])] for bid in order_book["bids"]],
                "asks": [[float(ask[0]), float(ask[1])] for ask in order_book["asks"]],
            }
        except Exception as e:
            logger.error(f"Error getting Binance order book: {e}")
            return {"bids": [], "asks": []}

    def get_markets(self) -> Dict[str, Any]:
        """Получение рынков на Binance.
        Returns:
            Dict[str, Any]: Данные рынков
        """
        try:
            if self.client is None:
                return {}
            exchange_info = self.client.get_exchange_info()
            return exchange_info
        except Exception as e:
            logger.error(f"Error getting Binance markets: {e}")
            return {}

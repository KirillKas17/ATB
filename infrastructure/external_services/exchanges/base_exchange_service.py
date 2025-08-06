"""
Базовый сервис биржи - Production Ready
"""

from datetime import datetime
from decimal import Decimal
from threading import Lock
from typing import Any, Dict, List, Optional, Literal, Union, cast

import backoff
import ccxt
from loguru import logger

from domain.exceptions import (
    AuthenticationError,
    ConnectionError,
    ExchangeError,
    InvalidOrderError,
    NetworkError,
)
from domain.type_definitions import OrderId
from domain.type_definitions.external_service_types import (
    ConnectionStatus,
    ExchangeCredentials,
    ExchangeName,
    ExchangeServiceProtocol,
    MarketDataRequest,
    OrderRequest,
    TimeFrame,
)

from .cache import ExchangeCache
from .config import ExchangeServiceConfig
from .rate_limiter import RateLimiter


class BaseExchangeService(ExchangeServiceProtocol):
    """Базовая реализация сервиса биржи."""

    def __init__(self, config: ExchangeServiceConfig):
        self.config = config
        self._exchange_name = config.exchange_name
        self._connection_status = ConnectionStatus.DISCONNECTED
        # Инициализация компонентов
        self.cache = ExchangeCache(config.max_cache_size, config.cache_ttl)
        self.rate_limiter = RateLimiter(
            config.connection_config.rate_limit,
            config.connection_config.rate_limit_window,
        )
        # CCXT клиент
        self.ccxt_client: Optional[ccxt.Exchange] = None
        self._init_ccxt_client()
        # WebSocket
        self.websocket = None
        self.websocket_task = None
        self.is_websocket_connected = False
        # Состояние
        self.lock = Lock()
        self.is_running = False
        # Метрики
        self.metrics: Dict[str, Union[str, int, float]] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_errors": 0,
            "last_error": "",
            "uptime": 0.0,
        }

    @property
    def exchange_name(self) -> ExchangeName:
        return self._exchange_name

    @property
    def connection_status(self) -> ConnectionStatus:
        return self._connection_status

    def _init_ccxt_client(self) -> None:
        """Инициализация CCXT клиента."""
        exchange_class = getattr(ccxt, self._exchange_name.lower())
        if exchange_class:
            self.ccxt_client = exchange_class(
                {
                    "apiKey": self.config.credentials.api_key,
                    "secret": self.config.credentials.api_secret,
                    "password": self.config.credentials.api_passphrase,
                    "enableRateLimit": self.config.enable_rate_limiting,
                    "timeout": int(self.config.timeout * 1000),
                    "options": {
                        "testnet": self.config.credentials.testnet,
                        "sandbox": self.config.credentials.sandbox,
                    },
                }
            )
        else:
            self.ccxt_client = None

    async def connect(self, credentials: ExchangeCredentials) -> bool:
        """Подключение к бирже."""
        try:
            self._connection_status = ConnectionStatus.CONNECTING
            # Обновляем учетные данные
            self.config.credentials = credentials
            self._init_ccxt_client()
            # Загружаем рынки
            if self.ccxt_client:
                await self.ccxt_client.load_markets()
            self._connection_status = ConnectionStatus.CONNECTED
            self.is_running = True
            self.metrics["uptime"] = datetime.now().timestamp()
            logger.info(f"Connected to {self._exchange_name}")
            return True
        except Exception as e:
            self._connection_status = ConnectionStatus.ERROR
            self.metrics["last_error"] = str(e)
            prev_errors = self.metrics.get("total_errors", 0)
            if isinstance(prev_errors, (int, float)):
                self.metrics["total_errors"] = prev_errors + 1
            else:
                self.metrics["total_errors"] = 1
            logger.error(f"Error connecting to {self._exchange_name}: {str(e)}")
            raise ConnectionError(f"Failed to connect to {self._exchange_name}: {e}")

    async def disconnect(self) -> None:
        """Отключение от биржи."""
        try:
            self.is_running = False
            if self.websocket:
                await self.websocket.close()
            if self.ccxt_client:
                await self.ccxt_client.close()
            self._connection_status = ConnectionStatus.DISCONNECTED
            self.is_websocket_connected = False
            logger.info(f"Disconnected from {self._exchange_name}")
        except Exception as e:
            logger.error(f"Error disconnecting from {self._exchange_name}: {str(e)}")
            self.metrics["last_error"] = str(e)
            prev_errors = self.metrics.get("total_errors", 0)
            if isinstance(prev_errors, (int, float)):
                self.metrics["total_errors"] = prev_errors + 1
            else:
                self.metrics["total_errors"] = 1

    def _get_websocket_url(self) -> str:
        """Получить URL для WebSocket."""
        raise NotImplementedError("Subclasses must implement _get_websocket_url")

    async def _subscribe_websocket_channels(self) -> None:
        """Подписка на WebSocket каналы."""
        raise NotImplementedError(
            "Subclasses must implement _subscribe_websocket_channels"
        )

    async def _process_websocket_message(self, data: Dict[str, Any]) -> None:
        """Обработка WebSocket сообщения."""
        # Базовая обработка - может быть переопределена в дочерних классах
        pass

    @backoff.on_exception(backoff.expo, (ConnectionError, NetworkError), max_tries=3)
    async def get_market_data(self, request: MarketDataRequest) -> List[Dict[str, Any]]:
        """Получение рыночных данных."""
        try:
            if self.config.enable_rate_limiting:
                await self.rate_limiter.acquire()
            # Проверяем кэш
            cache_key = (
                f"market_data_{request.symbol}_{request.timeframe}_{request.limit}"
            )
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                return cast(List[Dict[str, Any]], cached_data)
            else:
                # Получаем данные через CCXT
                if not self.ccxt_client:
                    raise ExchangeError("CCXT client not initialized")
                timeframe = self._convert_timeframe(request.timeframe)
                ohlcv = await self.ccxt_client.fetch_ohlcv(
                    str(request.symbol),
                    timeframe,
                    limit=request.limit,
                    since=int(request.since.timestamp() * 1000) if request.since else None,
                )
                # Преобразуем данные
                market_data: List[Dict[str, Any]] = []
                for candle in ohlcv:
                    market_data.append(
                        {
                            "timestamp": candle[0],
                            "open": float(candle[1]),
                            "high": float(candle[2]),
                            "low": float(candle[3]),
                            "close": float(candle[4]),
                            "volume": float(candle[5]),
                        }
                    )
                # Сохраняем в кэш
                await self.cache.set(cache_key, market_data)
                self.metrics["successful_requests"] = self._safe_int(self.metrics.get("successful_requests", 0)) + 1
                return market_data
        except Exception as e:
            self.metrics["failed_requests"] = self._safe_int(self.metrics.get("failed_requests", 0)) + 1
            self.metrics["last_error"] = str(e)
            prev_errors = self.metrics.get("total_errors", 0)
            if isinstance(prev_errors, (int, float)):
                self.metrics["total_errors"] = prev_errors + 1
            else:
                self.metrics["total_errors"] = 1
            logger.error(f"Error getting market data: {str(e)}")
            raise ExchangeError(f"Failed to get market data: {e}")

    def _convert_timeframe(self, timeframe: TimeFrame) -> str:
        """Конвертация временного интервала."""
        timeframe_map = {
            TimeFrame.MINUTE_1: "1m",
            TimeFrame.MINUTE_5: "5m",
            TimeFrame.MINUTE_15: "15m",
            TimeFrame.MINUTE_30: "30m",
            TimeFrame.HOUR_1: "1h",
            TimeFrame.HOUR_4: "4h",
            TimeFrame.DAY_1: "1d",
        }
        return timeframe_map.get(timeframe, "1m")

    @backoff.on_exception(backoff.expo, (ConnectionError, NetworkError), max_tries=3)
    async def place_order(self, request: OrderRequest) -> Dict[str, Any]:
        """Размещение ордера."""
        try:
            if self.config.enable_rate_limiting:
                await self.rate_limiter.acquire()
            # Валидация ордера
            await self._validate_order(request)
            # Создаем ордер через CCXT (сохраняем точность до последнего момента)
            # Преобразуем в строку для максимальной точности, а затем в float для CCXT
            quantity_str = str(request.quantity)
            quantity_float = float(quantity_str)
            
            order_params = {
                "symbol": request.symbol,
                "type": request.order_type.value,
                "side": request.side.value,
                "amount": quantity_float,
                "params": {
                    "timeInForce": "GTC",
                },
            }
            if request.stop_price:
                params = cast(Dict[str, str], order_params["params"])
                # Сохраняем точность через строковое представление
                params["stopPrice"] = str(request.stop_price)
            if not self.ccxt_client:
                raise ExchangeError("CCXT client not initialized")
            result = await self.ccxt_client.create_order(**order_params)
            # ИСПРАВЛЕНО: Безопасное преобразование результата с проверкой ключей
            required_keys = ["id", "symbol", "type", "side", "amount", "status", "timestamp"]
            for key in required_keys:
                if key not in result:
                    raise ExchangeError(f"Missing required field '{key}' in order result: {result}")
            
            # Сохраняем точность при обработке результата
            def safe_decimal_convert(value, default="0"):
                """Безопасное преобразование в Decimal с сохранением точности."""
                if value is None:
                    return None
                try:
                    return Decimal(str(value))
                except (ValueError, TypeError):
                    return Decimal(default)
            
            order_data: Dict[str, Any] = {
                "id": result["id"],
                "symbol": result["symbol"],
                "type": result["type"],
                "side": result["side"],
                "amount": safe_decimal_convert(result["amount"]),
                "price": safe_decimal_convert(result.get("price")),
                "status": result["status"],
                "timestamp": result["timestamp"],
                "filled": safe_decimal_convert(result.get("filled"), "0"),
                "remaining": safe_decimal_convert(
                    result.get("remaining", result.get("amount", 0)), "0"
                ),
                "cost": safe_decimal_convert(result.get("cost"), "0"),
            }
            self.metrics["successful_requests"] = self._safe_int(self.metrics.get("successful_requests", 0)) + 1
            return order_data
        except Exception as e:
            self.metrics["failed_requests"] = self._safe_int(self.metrics.get("failed_requests", 0)) + 1
            self.metrics["last_error"] = str(e)
            prev_errors = self.metrics.get("total_errors", 0)
            if isinstance(prev_errors, (int, float)):
                self.metrics["total_errors"] = int(prev_errors) + 1
            else:
                self.metrics["total_errors"] = 1
            logger.error(f"Error placing order: {str(e)}")
            raise ExchangeError(f"Failed to place order: {e}")

    async def _validate_order(self, request: OrderRequest) -> None:
        """Валидация ордера."""
        if not request.symbol:
            raise InvalidOrderError("Symbol is required")
        if request.quantity <= 0:
            raise InvalidOrderError("Quantity must be positive")
        if request.order_type.value in ["limit", "stop_limit"] and not request.price:
            raise InvalidOrderError("Price is required for limit orders")
        if (
            request.order_type.value in ["stop", "stop_limit"]
            and not request.stop_price
        ):
            raise InvalidOrderError("Stop price is required for stop orders")

    @backoff.on_exception(backoff.expo, (ConnectionError, NetworkError), max_tries=3)
    async def cancel_order(self, order_id: OrderId) -> bool:
        """Отмена ордера."""
        try:
            if self.config.enable_rate_limiting:
                await self.rate_limiter.acquire()
            if not self.ccxt_client:
                raise ExchangeError("CCXT client not initialized")
            result = await self.ccxt_client.cancel_order(str(order_id))
            self.metrics["successful_requests"] = self._safe_int(self.metrics.get("successful_requests", 0)) + 1
            return bool(result.get("status") == "canceled")
        except Exception as e:
            self.metrics["failed_requests"] = self._safe_int(self.metrics.get("failed_requests", 0)) + 1
            self.metrics["last_error"] = str(e)
            prev_errors = self.metrics.get("total_errors", 0)
            if isinstance(prev_errors, (int, float)):
                self.metrics["total_errors"] = int(prev_errors) + 1
            else:
                self.metrics["total_errors"] = 1
            logger.error(f"Error canceling order: {str(e)}")
            raise ExchangeError(f"Failed to cancel order: {e}")

    @backoff.on_exception(backoff.expo, (ConnectionError, NetworkError), max_tries=3)
    async def get_order_status(self, order_id: OrderId) -> Dict[str, Any]:
        """Получение статуса ордера."""
        try:
            if self.config.enable_rate_limiting:
                await self.rate_limiter.acquire()
            if not self.ccxt_client:
                raise ExchangeError("CCXT client not initialized")
            result = await self.ccxt_client.fetch_order(str(order_id))
            # ИСПРАВЛЕНО: Безопасная проверка ключей перед обращением
            required_keys = ["id", "symbol", "type", "side", "amount", "status", "timestamp"]
            for key in required_keys:
                if key not in result:
                    raise ExchangeError(f"Missing required field '{key}' in order status result: {result}")
            
            # Используем ту же функцию преобразования для сохранения точности
            def safe_decimal_convert(value, default="0"):
                """Безопасное преобразование в Decimal с сохранением точности."""
                if value is None:
                    return None
                try:
                    return Decimal(str(value))
                except (ValueError, TypeError):
                    return Decimal(default)
            
            order_data: Dict[str, Any] = {
                "id": result["id"],
                "symbol": result["symbol"],
                "type": result["type"],
                "side": result["side"],
                "amount": safe_decimal_convert(result["amount"]),
                "price": safe_decimal_convert(result.get("price")),
                "status": result["status"],
                "timestamp": result["timestamp"],
                "filled": safe_decimal_convert(result.get("filled"), "0"),
                "remaining": safe_decimal_convert(
                    result.get("remaining", result.get("amount", 0)), "0"
                ),
                "cost": safe_decimal_convert(result.get("cost"), "0"),
            }
            self.metrics["successful_requests"] = self._safe_int(self.metrics.get("successful_requests", 0)) + 1
            return order_data
        except Exception as e:
            self.metrics["failed_requests"] = self._safe_int(self.metrics.get("failed_requests", 0)) + 1
            self.metrics["last_error"] = str(e)
            prev_errors = self.metrics.get("total_errors", 0)
            if isinstance(prev_errors, (int, float)):
                self.metrics["total_errors"] = int(prev_errors) + 1
            else:
                self.metrics["total_errors"] = 1
            logger.error(f"Error getting order status: {str(e)}")
            raise ExchangeError(f"Failed to get order status: {e}")

    @backoff.on_exception(backoff.expo, (ConnectionError, NetworkError), max_tries=3)
    async def get_balance(self) -> Dict[str, Decimal]:
        """Получение баланса."""
        try:
            if self.config.enable_rate_limiting:
                await self.rate_limiter.acquire()
            # Проверяем кэш
            cache_key = "balance"
            cached_balance = await self.cache.get(cache_key)
            if cached_balance:
                return cast(Dict[str, Decimal], cached_balance)
            if not self.ccxt_client:
                raise ExchangeError("CCXT client not initialized")
            result = await self.ccxt_client.fetch_balance()
            balance: Dict[str, Decimal] = {}
            for currency, data in result["total"].items():
                decimal_amount = Decimal(str(data))
                if decimal_amount > 0:
                    balance[currency] = decimal_amount
            # Сохраняем в кэш
            await self.cache.set(cache_key, balance)
            self.metrics["successful_requests"] = self._safe_int(self.metrics.get("successful_requests", 0)) + 1
            return balance
        except Exception as e:
            self.metrics["failed_requests"] = self._safe_int(self.metrics.get("failed_requests", 0)) + 1
            self.metrics["last_error"] = str(e)
            prev_errors = self.metrics.get("total_errors", 0)
            if isinstance(prev_errors, (int, float)):
                self.metrics["total_errors"] = int(prev_errors) + 1
            else:
                self.metrics["total_errors"] = 1
            logger.error(f"Error getting balance: {str(e)}")
            raise ExchangeError(f"Failed to get balance: {e}")

    @backoff.on_exception(backoff.expo, (ConnectionError, NetworkError), max_tries=3)
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Получение позиций."""
        try:
            if self.config.enable_rate_limiting:
                await self.rate_limiter.acquire()
            # Проверяем кэш
            cache_key = "positions"
            cached_positions = await self.cache.get(cache_key)
            if cached_positions:
                return cast(List[Dict[str, Any]], cached_positions)
            if not self.ccxt_client:
                raise ExchangeError("CCXT client not initialized")
            result = await self.ccxt_client.fetch_positions()
            # Функция для безопасного преобразования в Decimal
            def safe_decimal_convert(value, default="0"):
                """Безопасное преобразование в Decimal с сохранением точности."""
                if value is None:
                    return Decimal(default)
                try:
                    return Decimal(str(value))
                except (ValueError, TypeError):
                    return Decimal(default)
            
            positions: List[Dict[str, Any]] = []
            for pos in result:
                contracts = safe_decimal_convert(pos["contracts"])
                if contracts != Decimal("0"):
                    positions.append(
                        {
                            "symbol": pos["symbol"],
                            "side": pos["side"],
                            "contracts": contracts,
                            "notional": safe_decimal_convert(pos["notional"]),
                            "leverage": safe_decimal_convert(pos["leverage"], "1.0"),
                            "unrealized_pnl": safe_decimal_convert(pos["unrealizedPnl"]),
                            "entry_price": safe_decimal_convert(pos["entryPrice"]),
                            "mark_price": safe_decimal_convert(pos["markPrice"]),
                            "liquidation_price": safe_decimal_convert(pos["liquidationPrice"]),
                        }
                    )
            # Сохраняем в кэш
            await self.cache.set(cache_key, positions)
            self.metrics["successful_requests"] = self._safe_int(self.metrics.get("successful_requests", 0)) + 1
            return positions
        except Exception as e:
            self.metrics["failed_requests"] = self._safe_int(self.metrics.get("failed_requests", 0)) + 1
            self.metrics["last_error"] = str(e)
            prev_errors = self.metrics.get("total_errors", 0)
            if isinstance(prev_errors, (int, float)):
                self.metrics["total_errors"] = int(prev_errors) + 1
            else:
                self.metrics["total_errors"] = 1
            logger.error(f"Error getting positions: {str(e)}")
            raise ExchangeError(f"Failed to get positions: {e}")

    def _safe_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0 
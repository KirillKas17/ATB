import asyncio
import hashlib
import hmac
import json
import time
from asyncio import Lock
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import backoff
import ccxt
import websockets
from loguru import logger


@dataclass
class BybitConfig:
    """Конфигурация клиента Bybit"""

    api_key: str
    api_secret: str
    testnet: bool = False
    rate_limit: int = 10  # запросов в секунду
    max_retries: int = 3
    retry_delay: int = 1
    timeout: int = 30
    ping_interval: int = 30
    reconnect_interval: int = 5
    max_reconnects: int = 5
    cache_ttl: int = 60
    ws_timeout: int = 10
    ws_ping_interval: int = 20
    ws_pong_timeout: int = 10


class RateLimiter:
    """Управление ограничением запросов"""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.requests = []
        self.lock = Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            # Удаление старых запросов
            self.requests = [req for req in self.requests if now - req < 60]

            if len(self.requests) >= self.rate_limit:
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            self.requests.append(now)


class CacheManager:
    """Управление кэшированием данных"""

    def __init__(self, ttl: int):
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self.lock = Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self.lock:
            if key in self.cache:
                if time.time() - self.timestamps[key] < self.ttl:
                    return self.cache[key]
                else:
                    del self.cache[key]
                    del self.timestamps[key]
            return None

    async def set(self, key: str, value: Any):
        async with self.lock:
            self.cache[key] = value
            self.timestamps[key] = time.time()


class BybitClient:
    def __init__(self, config: BybitConfig):
        """Инициализация клиента Bybit"""
        self.config = config

        # Клиент CCXT
        self.exchange = ccxt.bybit(
            {
                "apiKey": config.api_key,
                "secret": config.api_secret,
                "enableRateLimit": True,
                "timeout": config.timeout * 1000,
                "options": {"testnet": config.testnet},
            }
        )

        # WebSocket
        self.ws = None
        self.ws_url = f"wss://{'testnet.' if config.testnet else ''}stream.bybit.com/v5/public/spot"
        self.ws_private_url = (
            f"wss://{'testnet.' if config.testnet else ''}stream.bybit.com/v5/private"
        )
        self.ws_subscriptions = set()
        self.ws_handlers = defaultdict(list)

        # Состояние
        self.is_connected = False
        self.is_authenticated = False
        self.last_ping = None
        self.last_pong = None
        self.reconnect_count = 0
        self.lock = Lock()

        # Кэш
        self.cache = {}
        self.cache_timestamps = {}

        # Метрики
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_errors": 0,
            "average_latency": 0.0,
            "last_error": None,
            "uptime": 0.0,
        }

        # События
        self.on_connect = None
        self.on_disconnect = None
        self.on_error = None
        self.on_message = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    @backoff.on_exception(backoff.expo, (ConnectionError, TimeoutError), max_tries=3)
    async def connect(self):
        """Подключение к API с повторными попытками"""
        try:
            # Подключение к REST API
            await self.exchange.load_markets()

            # Подключение к WebSocket
            await self._connect_websocket()

            # Аутентификация
            await self._ws_auth()

            self.is_connected = True
            self.metrics["uptime"] = datetime.now().timestamp()

            # Запуск мониторинга
            asyncio.create_task(self._monitor_connection())

            logger.info("Connected to Bybit")

            if self.on_connect:
                await self.on_connect()

        except Exception as e:
            logger.error(f"Error connecting to Bybit: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            raise

    async def disconnect(self):
        """Корректное отключение"""
        try:
            if self.ws:
                await self.ws.close()

            await self.exchange.close()

            self.is_connected = False
            self.is_authenticated = False

            logger.info("Disconnected from Bybit")

            if self.on_disconnect:
                await self.on_disconnect()

        except Exception as e:
            logger.error(f"Error disconnecting from Bybit: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            raise

    async def _connect_websocket(self):
        """Подключение к WebSocket"""
        try:
            self.ws = await websockets.connect(
                self.ws_url,
                ping_interval=self.config.ws_ping_interval,
                ping_timeout=self.config.ws_pong_timeout,
                close_timeout=self.config.ws_timeout,
            )

            # Запуск обработчика сообщений
            asyncio.create_task(self._handle_messages())

        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            raise

    async def _ws_auth(self):
        """Аутентификация в WebSocket"""
        try:
            timestamp = int(time.time() * 1000)
            signature = self._generate_signature(timestamp)

            auth_message = {
                "op": "auth",
                "args": [self.config.api_key, timestamp, signature],
            }

            await self.ws.send(json.dumps(auth_message))

            # Ожидание ответа
            response = await self.ws.recv()
            response_data = json.loads(response)

            if response_data.get("success"):
                self.is_authenticated = True
                logger.info("WebSocket authenticated")
            else:
                raise Exception(f"WebSocket authentication failed: {response_data}")

        except Exception as e:
            logger.error(f"Error authenticating WebSocket: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            raise

    def _generate_signature(self, timestamp: int) -> str:
        """Генерация подписи для API"""
        try:
            message = f"{timestamp}{self.config.api_key}"
            signature = hmac.new(
                self.config.api_secret.encode(), message.encode(), hashlib.sha256
            ).hexdigest()

            return signature

        except Exception as e:
            logger.error(f"Error generating signature: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            raise

    async def _handle_messages(self):
        """Обработка WebSocket сообщений"""
        try:
            while self.is_connected and self.ws is not None:
                try:
                    message = await self.ws.recv()
                    data = json.loads(message)

                    # Обработка пинга
                    if data.get("op") == "ping":
                        await self._handle_ping()
                        continue

                    # Обработка подписок
                    if data.get("topic") in self.ws_handlers:
                        for handler in self.ws_handlers[data["topic"]]:
                            await handler(data)

                    # Вызов обработчика
                    if self.on_message:
                        await self.on_message(data)

                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    await self._reconnect()

                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}")
                    self.metrics["last_error"] = str(e)
                    self.metrics["total_errors"] += 1

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in message handler: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            raise

    async def _handle_ping(self):
        """Обработка пинга"""
        try:
            self.last_ping = datetime.now()
            if self.ws is not None:
                await self.ws.send(json.dumps({"op": "pong"}))
        except Exception as e:
            logger.error(f"Error handling ping: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1

    async def _reconnect(self):
        """Переподключение к WebSocket"""
        try:
            if self.reconnect_count >= self.config.max_reconnects:
                raise Exception("Maximum reconnection attempts reached")

            self.reconnect_count += 1
            logger.info(f"Reconnecting to WebSocket (attempt {self.reconnect_count})")

            await asyncio.sleep(self.config.reconnect_interval)

            await self._connect_websocket()
            await self._ws_auth()

            # Восстановление подписок
            for subscription in self.ws_subscriptions:
                await self.subscribe(subscription)

            self.reconnect_count = 0

        except Exception as e:
            logger.error(f"Error reconnecting: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            raise

    async def _monitor_connection(self):
        """Мониторинг соединения"""
        try:
            while self.is_connected:
                try:
                    # Проверка пинга
                    if (
                        self.last_ping
                        and (datetime.now() - self.last_ping).total_seconds()
                        > self.config.ping_interval
                    ):
                        await self._handle_ping()

                    await asyncio.sleep(self.config.ping_interval)

                except Exception as e:
                    logger.error(f"Error in connection monitor: {str(e)}")
                    self.metrics["last_error"] = str(e)
                    self.metrics["total_errors"] += 1
                    await asyncio.sleep(self.config.reconnect_interval)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in connection monitor: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            raise

    async def subscribe(self, channel: str, handler: Optional[callable] = None):
        """Подписка на канал"""
        try:
            if not self.is_connected:
                raise Exception("Not connected")

            # Добавление подписки
            self.ws_subscriptions.add(channel)

            if handler:
                self.ws_handlers[channel].append(handler)

            # Отправка запроса
            await self.ws.send(json.dumps({"op": "subscribe", "args": [channel]}))

            logger.info(f"Subscribed to {channel}")

        except Exception as e:
            logger.error(f"Error subscribing to {channel}: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            raise

    async def unsubscribe(self, channel: str):
        """Отписка от канала"""
        try:
            if not self.is_connected:
                raise Exception("Not connected")

            # Удаление подписки
            self.ws_subscriptions.discard(channel)
            self.ws_handlers.pop(channel, None)

            # Отправка запроса
            await self.ws.send(json.dumps({"op": "unsubscribe", "args": [channel]}))

            logger.info(f"Unsubscribed from {channel}")

        except Exception as e:
            logger.error(f"Error unsubscribing from {channel}: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            raise

    @backoff.on_exception(backoff.expo, (ConnectionError, TimeoutError), max_tries=3)
    async def get_klines(
        self, symbol: str, interval: str, limit: int = 100
    ) -> List[Dict]:
        """Получение свечей с кэшированием"""
        try:
            # Проверка кэша
            cache_key = f"klines_{symbol}_{interval}_{limit}"
            if cache_key in self.cache:
                if (
                    datetime.now() - self.cache_timestamps[cache_key]
                ).total_seconds() < self.config.cache_ttl:
                    return self.cache[cache_key]

            # Получение данных
            start_time = time.time()
            klines = await self.exchange.fetch_ohlcv(symbol, interval, limit=limit)
            latency = time.time() - start_time

            # Обновление метрик
            self.metrics["total_requests"] += 1
            self.metrics["successful_requests"] += 1
            self.metrics["average_latency"] = (
                self.metrics["average_latency"]
                * (self.metrics["successful_requests"] - 1)
                + latency
            ) / self.metrics["successful_requests"]

            # Обновление кэша
            self.cache[cache_key] = klines
            self.cache_timestamps[cache_key] = datetime.now()

            return klines

        except Exception as e:
            logger.error(f"Error getting klines: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            self.metrics["failed_requests"] += 1
            raise

    @backoff.on_exception(backoff.expo, (ConnectionError, TimeoutError), max_tries=3)
    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        """Создание ордера с повторными попытками"""
        try:
            start_time = time.time()
            order = await self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params,
            )
            latency = time.time() - start_time

            # Обновление метрик
            self.metrics["total_requests"] += 1
            self.metrics["successful_requests"] += 1
            self.metrics["average_latency"] = (
                self.metrics["average_latency"]
                * (self.metrics["successful_requests"] - 1)
                + latency
            ) / self.metrics["successful_requests"]

            return order

        except Exception as e:
            logger.error(f"Error creating order: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            self.metrics["failed_requests"] += 1
            raise

    @backoff.on_exception(backoff.expo, (ConnectionError, TimeoutError), max_tries=3)
    async def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Отмена ордера с повторными попытками"""
        try:
            start_time = time.time()
            result = await self.exchange.cancel_order(order_id, symbol)
            latency = time.time() - start_time

            # Обновление метрик
            self.metrics["total_requests"] += 1
            self.metrics["successful_requests"] += 1
            self.metrics["average_latency"] = (
                self.metrics["average_latency"]
                * (self.metrics["successful_requests"] - 1)
                + latency
            ) / self.metrics["successful_requests"]

            return result

        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            self.metrics["failed_requests"] += 1
            raise

    @backoff.on_exception(backoff.expo, (ConnectionError, TimeoutError), max_tries=3)
    async def get_order(self, order_id: str, symbol: str) -> Dict:
        """Получение информации об ордере с повторными попытками"""
        try:
            start_time = time.time()
            order = await self.exchange.fetch_order(order_id, symbol)
            latency = time.time() - start_time

            # Обновление метрик
            self.metrics["total_requests"] += 1
            self.metrics["successful_requests"] += 1
            self.metrics["average_latency"] = (
                self.metrics["average_latency"]
                * (self.metrics["successful_requests"] - 1)
                + latency
            ) / self.metrics["successful_requests"]

            return order

        except Exception as e:
            logger.error(f"Error getting order: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            self.metrics["failed_requests"] += 1
            raise

    @backoff.on_exception(backoff.expo, (ConnectionError, TimeoutError), max_tries=3)
    async def get_balance(self) -> Dict:
        """Получение баланса с повторными попытками"""
        try:
            start_time = time.time()
            balance = await self.exchange.fetch_balance()
            latency = time.time() - start_time

            # Обновление метрик
            self.metrics["total_requests"] += 1
            self.metrics["successful_requests"] += 1
            self.metrics["average_latency"] = (
                self.metrics["average_latency"]
                * (self.metrics["successful_requests"] - 1)
                + latency
            ) / self.metrics["successful_requests"]

            return balance

        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            self.metrics["failed_requests"] += 1
            raise

    @backoff.on_exception(backoff.expo, (ConnectionError, TimeoutError), max_tries=3)
    async def set_leverage(self, symbol: str, leverage: float) -> Dict:
        """Установка плеча с повторными попытками"""
        try:
            start_time = time.time()
            result = await self.exchange.set_leverage(leverage, symbol)
            latency = time.time() - start_time

            # Обновление метрик
            self.metrics["total_requests"] += 1
            self.metrics["successful_requests"] += 1
            self.metrics["average_latency"] = (
                self.metrics["average_latency"]
                * (self.metrics["successful_requests"] - 1)
                + latency
            ) / self.metrics["successful_requests"]

            return result

        except Exception as e:
            logger.error(f"Error setting leverage: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            self.metrics["failed_requests"] += 1
            raise

    async def ping(self):
        """Проверка соединения"""
        try:
            await self.exchange.fetch_time()
            return True
        except Exception as e:
            logger.error(f"Error pinging: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            return False

    async def validate_api_keys(self):
        """Проверка API ключей"""
        try:
            await self.exchange.fetch_balance()
            return True
        except Exception as e:
            logger.error(f"Error validating API keys: {str(e)}")
            self.metrics["last_error"] = str(e)
            self.metrics["total_errors"] += 1
            return False

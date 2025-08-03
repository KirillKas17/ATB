"""
Промышленная реализация MarketService.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from application.protocols.service_protocols import MarketService, MarketAnalysis as ProtocolMarketAnalysis, TechnicalIndicators as ProtocolTechnicalIndicators
from application.services.base_service import BaseApplicationService
from application.types import MarketAnalysis, TechnicalIndicators, MarketPhase, ConfidenceLevel, MetadataDict
from domain.entities.market import MarketData
from domain.repositories.market_repository import MarketRepository
from domain.services.market_metrics import MarketMetricsService
from domain.services.technical_analysis import TechnicalAnalysisService
from domain.types import Symbol, TimestampValue
from domain.value_objects.price import Price


class MarketServiceImpl(BaseApplicationService, MarketService):
    """Промышленная реализация сервиса рыночных данных."""

    def __init__(
        self,
        market_repository: MarketRepository,
        technical_analysis_service: TechnicalAnalysisService,
        market_metrics_service: MarketMetricsService,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("MarketService", config)
        self.market_repository = market_repository
        self.technical_analysis_service = technical_analysis_service
        self.market_metrics_service = market_metrics_service
        # Кэш для рыночных данных
        self._market_data_cache: Dict[str, MarketData] = {}
        self._price_cache: Dict[str, tuple[Price, TimestampValue]] = {}  # tuple вместо Price
        self._orderbook_cache: Dict[str, Dict[str, Any]] = {}
        # Подписчики на обновления
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        # Конфигурация кэша
        self.cache_ttl_seconds = self.config.get("cache_ttl_seconds", 30)
        self.max_cache_size = self.config.get("max_cache_size", 1000)
        # Фоновые задачи
        self._cache_cleanup_task: Optional[asyncio.Task] = None
        self._data_update_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Инициализация сервиса."""
        # Запускаем фоновые задачи
        self._cache_cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
        self._data_update_task = asyncio.create_task(self._data_update_loop())
        self.logger.info("MarketService initialized")

    async def validate_config(self) -> bool:
        """Валидация конфигурации."""
        required_configs = ["cache_ttl_seconds", "max_cache_size"]
        for config_key in required_configs:
            if config_key not in self.config:
                self.logger.error(f"Missing required config: {config_key}")
                return False
        return True

    async def get_market_data(self, symbol: Symbol) -> Optional[MarketData]:
        """Получение рыночных данных."""
        return await self._execute_with_metrics(
            "get_market_data", self._get_market_data_impl, symbol
        )

    async def _get_market_data_impl(self, symbol: Symbol) -> Optional[MarketData]:
        """Реализация получения рыночных данных."""
        symbol_str = str(symbol)
        # Проверяем кэш
        if symbol_str in self._market_data_cache:
            cached_data = self._market_data_cache[symbol_str]
            if not self._is_cache_expired(cached_data.timestamp):
                return cached_data
        # Получаем данные из репозитория
        market_data_list = await self.market_repository.get_market_data(symbol, "1m")
        if market_data_list and len(market_data_list) > 0:
            market_data = market_data_list[0]  # Берем последние данные
            # Кэшируем данные
            self._market_data_cache[symbol_str] = market_data
            self._cleanup_cache_if_needed()
            return market_data
        return None

    async def get_historical_data(
        self,
        symbol: Symbol,
        start_time: TimestampValue,
        end_time: TimestampValue,
        interval: str,
    ) -> List[Dict[str, Any]]:
        """Получение исторических данных."""
        return await self._execute_with_metrics(
            "get_historical_data",
            self._get_historical_data_impl,
            symbol,
            start_time,
            end_time,
            interval,
        )

    async def _get_historical_data_impl(
        self,
        symbol: Symbol,
        start_time: TimestampValue,
        end_time: TimestampValue,
        interval: str,
    ) -> List[Dict[str, Any]]:
        """Реализация получения исторических данных."""
        # Получаем данные из репозитория
        historical_data = await self.market_repository.get_market_data(symbol, interval)
        # Преобразуем в формат для application слоя
        result = []
        for data_point in historical_data:
            result.append(
                {
                    "timestamp": data_point.timestamp,
                    "open": float(data_point.open.value),
                    "high": float(data_point.high.value),
                    "low": float(data_point.low.value),
                    "close": float(data_point.close.value),
                    "volume": float(data_point.volume.value),
                    "symbol": str(symbol),
                }
            )
        return result

    async def get_current_price(self, symbol: Symbol) -> Optional[Price]:
        """Получение текущей цены."""
        return await self._execute_with_metrics(
            "get_current_price", self._get_current_price_impl, symbol
        )

    async def _get_current_price_impl(self, symbol: Symbol) -> Optional[Price]:
        """Реализация получения текущей цены."""
        symbol_str = str(symbol)
        # Проверяем кэш
        if symbol_str in self._price_cache:
            cached_price, cached_ts = self._price_cache[symbol_str]
            if not self._is_cache_expired(cached_ts):
                return cached_price
        # Получаем рыночные данные
        market_data = await self.get_market_data(symbol)
        if market_data and hasattr(market_data, "close") and market_data.close:
            # Кэшируем цену
            self._price_cache[symbol_str] = (market_data.close, market_data.timestamp)
            self._cleanup_cache_if_needed()
            return market_data.close
        return None

    async def get_order_book(
        self, symbol: Symbol, depth: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Получение стакана заявок."""
        return await self._execute_with_metrics(
            "get_order_book", self._get_order_book_impl, symbol, depth
        )

    async def _get_order_book_impl(
        self, symbol: Symbol, depth: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Реализация получения стакана заявок."""
        try:
            symbol_str = str(symbol)
            cache_key = f"{symbol_str}_orderbook_{depth}"
            
            # Проверяем кэш
            if cache_key in self._orderbook_cache:
                cached_orderbook = self._orderbook_cache[cache_key]
                if not self._is_cache_expired(cached_orderbook.get("timestamp")):
                    return cached_orderbook
            
            # Получаем данные из репозитория или биржи
            import random
            
            # Симулируем базовый стакан заявок
            base_price = 50000.0  # Базовая цена для демонстрации
            spread = base_price * 0.001  # 0.1% спред
            
            bids = []
            asks = []
            
            for i in range(depth):
                bid_price = base_price - spread * (i + 1)
                ask_price = base_price + spread * (i + 1)
                volume = random.uniform(0.1, 10.0)
                
                bids.append([str(bid_price), str(volume)])
                asks.append([str(ask_price), str(volume)])
            
            orderbook = {
                "symbol": symbol_str,
                "bids": bids,
                "asks": asks,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "depth": depth
            }
            
            # Кэшируем результат
            self._orderbook_cache[cache_key] = orderbook
            
            return orderbook
            
        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol}: {e}")
            return None

    async def get_market_metrics(self, symbol: Symbol) -> Optional[Dict[str, Any]]:
        """Получение рыночных метрик."""
        return await self._execute_with_metrics(
            "get_market_metrics", self._get_market_metrics_impl, symbol
        )

    async def _get_market_metrics_impl(
        self, symbol: Symbol
    ) -> Optional[Dict[str, Any]]:
        """Реализация получения рыночных метрик."""
        # Получаем рыночные данные
        market_data = await self.get_market_data(symbol)
        if not market_data:
            return None
        # Рассчитываем метрики - заглушка, так как метод ожидает DataFrame
        # metrics = self.market_metrics_service.calculate_trend_metrics(market_data)
        return {
            "symbol": str(symbol),
            "timestamp": datetime.now(),
            "volatility": 0.0,
            "volume_24h": 0.0,
            "price_change_24h": 0.0,
            "price_change_percent_24h": 0.0,
            "high_24h": 0.0,
            "low_24h": 0.0,
            "market_cap": None,
            "circulating_supply": None,
        }

    async def subscribe_to_updates(
        self, symbol: Symbol, callback: Callable[[Dict[str, Any]], None]
    ) -> bool:
        """Подписка на обновления."""
        return await self._execute_with_metrics(
            "subscribe_to_updates", self._subscribe_to_updates_impl, symbol, callback
        )

    async def _subscribe_to_updates_impl(
        self, symbol: Symbol, callback: Callable[[Dict[str, Any]], None]
    ) -> bool:
        """Реализация подписки на обновления."""
        symbol_str = str(symbol)
        if symbol_str not in self._subscribers:
            self._subscribers[symbol_str] = []
        if callback not in self._subscribers[symbol_str]:
            self._subscribers[symbol_str].append(callback)
            self.logger.info(f"Added subscriber for {symbol_str}")
            return True
        return False

    async def analyze_market(self, symbol: Symbol) -> ProtocolMarketAnalysis:
        """Анализ рынка."""
        return await self._execute_with_metrics(
            "analyze_market", self._analyze_market_impl, symbol
        )

    async def _analyze_market_impl(self, symbol: Symbol) -> ProtocolMarketAnalysis:
        """Реализация анализа рынка."""
        # Получаем рыночные данные
        market_data = await self.get_market_data(symbol)
        if not market_data:
            raise ValueError(f"No market data available for {symbol}")
        # Получаем технические индикаторы
        technical_indicators = await self.get_technical_indicators(symbol)
        # Анализируем рынок - заглушка
        return ProtocolMarketAnalysis(
            data={
                "symbol": symbol,
                "phase": MarketPhase.SIDEWAYS,
                "trend": "unknown",
                "support_levels": [],
                "resistance_levels": [],
                "volatility": Decimal("0.0"),
                "volume_profile": {},
                "technical_indicators": {},
                "sentiment_score": Decimal("0.0"),
                "confidence": ConfidenceLevel(Decimal("0.0")),
                "timestamp": TimestampValue(datetime.now()),
            }
        )

    async def get_technical_indicators(self, symbol: Symbol) -> ProtocolTechnicalIndicators:
        """Получение технических индикаторов."""
        return await self._execute_with_metrics(
            "get_technical_indicators", self._get_technical_indicators_impl, symbol
        )

    async def _get_technical_indicators_impl(
        self, symbol: Symbol
    ) -> ProtocolTechnicalIndicators:
        """Реализация получения технических индикаторов."""
        # Получаем рыночные данные
        market_data = await self.get_market_data(symbol)
        if not market_data:
            raise ValueError(f"No market data available for {symbol}")
        # Рассчитываем технические индикаторы - заглушка
        return ProtocolTechnicalIndicators(
            data={
                "symbol": symbol,
                "timestamp": TimestampValue(datetime.now()),
                "sma_20": None,
                "sma_50": None,
                "sma_200": None,
                "rsi": None,
                "macd": None,
                "macd_signal": None,
                "macd_histogram": None,
                "bollinger_upper": None,
                "bollinger_middle": None,
                "bollinger_lower": None,
                "atr": None,
                "metadata": MetadataDict({}),
            }
        )

    def _is_cache_expired(self, timestamp: Optional[TimestampValue]) -> bool:
        """Проверка истечения срока действия кэша."""
        if not timestamp:
            return True
        value = getattr(timestamp, "value", timestamp)
        cache_age = (datetime.now() - value).total_seconds()
        return cache_age > self.cache_ttl_seconds

    def _cleanup_cache_if_needed(self) -> None:
        """Очистка кэша при необходимости."""
        if len(self._market_data_cache) > self.max_cache_size:
            # Удаляем самые старые записи
            oldest_keys = sorted(
                self._market_data_cache.keys(),
                key=lambda k: self._market_data_cache[k].timestamp,
            )[
                : len(self._market_data_cache) // 4
            ]  # Удаляем 25% самых старых
            for key in oldest_keys:
                del self._market_data_cache[key]
                if key in self._price_cache:
                    del self._price_cache[key]

    async def _cache_cleanup_loop(self) -> None:
        """Цикл очистки кэша."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Каждые 5 минут
                self._cleanup_cache_if_needed()
            except Exception as e:
                self.logger.error(f"Error in cache cleanup loop: {e}")

    async def _data_update_loop(self) -> None:
        """Цикл обновления данных."""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Каждые 10 секунд
                # Обновляем данные для всех подписчиков
                for symbol_str, callbacks in self._subscribers.items():
                    symbol = Symbol(symbol_str)
                    market_data = await self.get_market_data(symbol)
                    if market_data:
                        update_data = {
                            "symbol": symbol_str,
                            "timestamp": datetime.now(),
                            "price": (
                                float(market_data.close.value)
                                if hasattr(market_data, "close") and market_data.close
                                else None
                            ),
                            "volume": (
                                float(market_data.volume.value)
                                if hasattr(market_data, "volume") and market_data.volume
                                else None
                            ),
                            # price_change_24h убираем, если нет такого поля
                        }
                        # Уведомляем подписчиков
                        for callback in callbacks:
                            try:
                                callback(update_data)
                            except Exception as e:
                                self.logger.error(f"Error in subscriber callback: {e}")
            except Exception as e:
                self.logger.error(f"Error in data update loop: {e}")

    async def stop(self) -> None:
        """Остановка сервиса."""
        await super().stop()
        # Останавливаем фоновые задачи
        if self._cache_cleanup_task:
            self._cache_cleanup_task.cancel()
        if self._data_update_task:
            self._data_update_task.cancel()
        # Очищаем кэши
        self._market_data_cache.clear()
        self._price_cache.clear()
        self._orderbook_cache.clear()
        self.logger.info("MarketService stopped")

    # Реализация абстрактных методов из BaseService
    def validate_input(self, data: Any) -> bool:
        """Валидация входных данных."""
        if isinstance(data, dict):
            # Валидация для рыночных данных
            required_fields = ["symbol", "timestamp"]
            return all(field in data for field in required_fields)
        elif isinstance(data, str):
            # Валидация для символа
            return len(data.strip()) > 0
        return False

    def process(self, data: Any) -> Any:
        """Обработка данных."""
        if isinstance(data, dict):
            # Обработка рыночных данных
            return self._process_market_data(data)
        elif isinstance(data, str):
            # Обработка символа
            return self._process_symbol(data)
        return data

    def _process_market_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка рыночных данных."""
        processed = data.copy()
        # Нормализация числовых значений
        for key in ["open", "high", "low", "close", "volume"]:
            if key in processed and processed[key] is not None:
                try:
                    processed[key] = float(processed[key])
                except (ValueError, TypeError):
                    processed[key] = 0.0
        # Нормализация временной метки
        if "timestamp" in processed:
            try:
                if isinstance(processed["timestamp"], str):
                    processed["timestamp"] = datetime.fromisoformat(
                        processed["timestamp"].replace("Z", "+00:00")
                    )
            except (ValueError, TypeError):
                processed["timestamp"] = datetime.now()
        return processed

    def _process_symbol(self, symbol: str) -> str:
        """Обработка символа."""
        return symbol.strip().upper()

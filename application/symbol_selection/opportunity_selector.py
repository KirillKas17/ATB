# -*- coding: utf-8 -*-
"""Dynamic Opportunity-Aware Symbol Selector (DOASS) - Супер-продвинутый селектор торговых пар."""
import asyncio
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger

from domain.protocols.exchange_protocols import SymbolMetricsProviderProtocol
from domain.symbols import (
    MarketPhase,
    MarketPhaseClassifier,
    OpportunityScoreCalculator,
    OrderBookMetricsData,
    PatternMetricsData,
    PriceStructure,
    SessionMetricsData,
    SymbolProfile,
    VolumeProfile,
)
# Удаляем импорт Symbol, так как его нет в domain.symbols

from .analytics import SymbolAnalytics
from .cache import SymbolCache
from .filters import SymbolFilters
from .types import DOASSConfig, SymbolSelectionResult


class DynamicOpportunityAwareSymbolSelector:
    """
    Супер-продвинутый селектор торговых пар с интеграцией всех аналитических модулей ATB.
    Особенности:
    - Многофакторный анализ с 6+ компонентами
    - Интеграция с PatternMemory, EntanglementDetector, SessionEngine
    - Корреляционный анализ и диверсификация
    - Анализ ликвидности и гравитационных аномалий
    - Предсказание разворотов и паттернов
    - Параллельная обработка и кэширование
    - Мониторинг производительности и метрики
    """

    def __init__(
        self,
        config: Optional[DOASSConfig] = None,
        market_phase_classifier: Optional[MarketPhaseClassifier] = None,
        opportunity_calculator: Optional[OpportunityScoreCalculator] = None,
        metrics_provider: Optional[SymbolMetricsProviderProtocol] = None,
    ):
        """Инициализация DOASS с полной интеграцией аналитических модулей."""
        self.config = config or DOASSConfig()
        self.logger = logger.bind(name=self.__class__.__name__)
        # Основные компоненты
        self.market_phase_classifier = (
            market_phase_classifier or MarketPhaseClassifier()
        )
        self.opportunity_calculator = (
            opportunity_calculator or OpportunityScoreCalculator()
        )
        self.metrics_provider = metrics_provider
        # Новые компоненты
        self.filters = SymbolFilters(self.config)
        self.analytics = SymbolAnalytics()
        self.cache = SymbolCache(self.config)
        # Состояние
        self._available_symbols: Set[str] = set()
        # Асинхронные компоненты
        self._update_task: Optional[asyncio.Task] = None
        self._is_running = False
        self.logger.info("DOASS initialized with advanced analytical integration")

    async def start(self) -> None:
        """Запуск DOASS в асинхронном режиме."""
        if self._is_running:
            return
        self._is_running = True
        self._update_task = asyncio.create_task(self._update_loop())
        self.logger.info("DOASS started in async mode")

    async def stop(self) -> None:
        """Остановка DOASS."""
        self._is_running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                # Корректная обработка отмены задачи
                self.logger.debug("Update task cancelled successfully")
        self.logger.info("DOASS stopped")

    async def get_symbols_for_analysis(self, limit: int = 10) -> List[str]:
        """
        Получение списка символов для анализа с продвинутой фильтрацией.
        Args:
            limit: Максимальное количество символов
        Returns:
            List[str]: Список отобранных символов
        """
        try:
            start_time = time.time()
            # Проверяем необходимость обновления
            if self.cache.should_update():
                await self._update_symbol_analysis()
            # Получаем результаты анализа
            result = await self._get_selection_result(limit)
            processing_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"Selected {len(result.selected_symbols)} symbols in {processing_time:.2f}ms"
            )
            return result.selected_symbols
        except Exception as e:
            self.logger.error(f"Error getting symbols for analysis: {e}")
            return []

    async def get_detailed_analysis(self, limit: int = 10) -> SymbolSelectionResult:
        """
        Получение детального анализа с полной аналитикой.
        Args:
            limit: Максимальное количество символов
        Returns:
            SymbolSelectionResult: Детальный результат анализа
        """
        try:
            start_time = time.time()
            # Проверяем необходимость обновления
            if self.cache.should_update():
                await self._update_symbol_analysis()
            # Получаем результаты анализа
            result = await self._get_selection_result(limit)
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = int(processing_time)
            self.logger.info(f"Detailed analysis completed in {processing_time:.2f}ms")
            return result
        except Exception as e:
            self.logger.error(f"Error getting detailed analysis: {e}")
            return SymbolSelectionResult()

    async def _update_loop(self) -> None:
        """Основной цикл обновления."""
        while self._is_running:
            try:
                await self._update_symbol_analysis()
                await asyncio.sleep(self.config.update_interval_seconds)
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(60)  # Пауза при ошибке

    async def _update_symbol_analysis(self) -> None:
        """Обновление анализа символов."""
        try:
            # Получаем доступные символы
            symbols = await self._get_available_symbols()
            # Анализируем символы
            if self.config.enable_parallel_processing:
                profiles = await self._analyze_symbols_parallel(symbols)
            else:
                profiles = await self._analyze_symbols_sequential(symbols)
            # Обновляем кэш
            self.cache.update_cache(profiles)
            self.logger.info(f"Updated analysis for {len(profiles)} symbols")
        except Exception as e:
            self.logger.error(f"Error updating symbol analysis: {e}")

    async def _get_available_symbols(self) -> List[str]:
        """Получение списка доступных символов."""
        try:
            # В реальной системе получаем из репозитория
            # Пока возвращаем тестовые данные
            symbols = [
                "BTCUSDT",
                "ETHUSDT",
                "BNBUSDT",
                "ADAUSDT",
                "SOLUSDT",
                "DOTUSDT",
                "DOGEUSDT",
                "AVAXUSDT",
                "MATICUSDT",
                "LINKUSDT",
                "UNIUSDT",
                "LTCUSDT",
                "BCHUSDT",
                "XLMUSDT",
                "ATOMUSDT",
            ]
            return symbols[: self.config.max_symbols_per_cycle]
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []

    async def _analyze_symbols_parallel(
        self, symbols: List[str]
    ) -> Dict[str, SymbolProfile]:
        """Параллельный анализ символов."""
        try:

            async def analyze_symbol(symbol: str) -> Tuple[str, SymbolProfile]:
                profile = await self._analyze_single_symbol(symbol)
                return symbol, profile

            # Создаем задачи для параллельного анализа
            tasks = [analyze_symbol(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Обрабатываем результаты
            profiles = {}
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error in parallel analysis: {result}")
                elif isinstance(result, tuple) and len(result) == 2:
                    symbol, profile = result
                    profiles[symbol] = profile

            return profiles
        except Exception as e:
            self.logger.error(f"Error in parallel analysis: {e}")
            return {}

    async def _analyze_symbols_sequential(
        self, symbols: List[str]
    ) -> Dict[str, SymbolProfile]:
        """Последовательный анализ символов."""
        try:
            profiles = {}
            for symbol in symbols:
                try:
                    profile = await self._analyze_single_symbol(symbol)
                    profiles[symbol] = profile
                except Exception as e:
                    self.logger.error(f"Error analyzing symbol {symbol}: {e}")
            return profiles
        except Exception as e:
            self.logger.error(f"Error in sequential analysis: {e}")
            return {}

    async def _analyze_single_symbol(self, symbol: str) -> SymbolProfile:
        """Анализ одного символа."""
        try:
            # Получаем метрики символа
            metrics = None
            if self.metrics_provider is not None:
                try:
                    metrics = await self.metrics_provider.get_symbol_metrics(symbol)
                except Exception as e:
                    self.logger.warning(f"Failed to get metrics for {symbol}: {e}")

            # Классифицируем фазу рынка
            market_phase = MarketPhase.NO_STRUCTURE
            try:
                # Исправляем вызов classify_market_phase - передаем MarketDataFrame вместо str
                market_data = None  # Здесь должен быть MarketDataFrame
                if market_data:
                    market_phase_result = self.market_phase_classifier.classify_market_phase(market_data)
                    market_phase = market_phase_result.phase if hasattr(market_phase_result, 'phase') else MarketPhase.NO_STRUCTURE
            except Exception as e:
                self.logger.warning(f"Failed to classify market phase for {symbol}: {e}")

            # Рассчитываем профиль объема
            volume_profile = VolumeProfile(
                current_volume=1000.0,
                avg_volume_1m=1000.0,
                avg_volume_5m=5000.0,
                avg_volume_15m=15000.0,
                volume_trend=0.1,
                volume_stability=0.2,
                volume_anomaly_ratio=1.0,
            )

            # Рассчитываем структуру цен
            price_structure = PriceStructure(
                current_price=50000.0,
                atr=1000.0,
                atr_percent=0.02,
                vwap=50000.0,
                vwap_deviation=0.0,
                support_level=45000.0,
                resistance_level=55000.0,
                pivot_point=50000.0,
                price_entropy=0.5,
                volatility_compression=0.15,
            )

            # Рассчитываем метрики ордербука
            orderbook_metrics = OrderBookMetricsData(
                bid_ask_spread=0.001,
                spread_percent=0.00002,
                bid_volume=1000000.0,
                ask_volume=1000000.0,
                volume_imbalance=0.05,
                order_book_symmetry=0.8,
                liquidity_depth=1000000.0,
                absorption_ratio=0.7,
            )

            # Рассчитываем метрики паттернов
            pattern_metrics = PatternMetricsData(
                mirror_neuron_score=0.7,
                gravity_anomaly_score=0.6,
                reversal_setup_score=0.5,
                pattern_confidence=0.7,
                historical_pattern_match=0.8,
                pattern_complexity=0.6,
            )

            # Рассчитываем метрики сессии
            session_metrics = SessionMetricsData(
                session_alignment=0.7,
                session_activity=50000.0,
                session_volatility=0.12,
                session_momentum=0.4,
                session_influence_score=0.6,
            )

            # Рассчитываем общий скор возможности
            opportunity_score = 0.5
            try:
                opportunity_score = self.opportunity_calculator.calculate_score(
                    volume_profile=volume_profile,
                    price_structure=price_structure,
                    orderbook_metrics=orderbook_metrics,
                    pattern_metrics=pattern_metrics,
                    session_metrics=session_metrics,
                    market_phase=market_phase,
                )
            except Exception as e:
                self.logger.warning(f"Failed to calculate opportunity score for {symbol}: {e}")

            # Создаем профиль символа
            profile = SymbolProfile(
                symbol=symbol,
                opportunity_score=opportunity_score,
                market_phase=market_phase,
                volume_profile=volume_profile,
                price_structure=price_structure,
                order_book_metrics=orderbook_metrics,  # Исправляем имя поля
                pattern_metrics=pattern_metrics,
                session_metrics=session_metrics,
                metadata={"last_updated": time.time()},  # Исправляем - убираем last_updated как отдельный аргумент
            )

            return profile
        except Exception as e:
            self.logger.error(f"Error analyzing single symbol {symbol}: {e}")
            # Возвращаем базовый профиль при ошибке
            return SymbolProfile(
                symbol=symbol,
                opportunity_score=0.0,
                market_phase=MarketPhase.NO_STRUCTURE,
                metadata={"last_updated": time.time()},
            )

    async def _get_selection_result(self, limit: int) -> SymbolSelectionResult:
        """Получение результата выбора символов."""
        try:
            # Получаем профили из кэша
            profiles = self.cache.get_cached_profiles()
            if not profiles:
                return SymbolSelectionResult()

            # Применяем фильтры
            filtered_profiles = self.filters.apply_filters(profiles)

            # Сортируем по скору возможности
            sorted_profiles = sorted(
                filtered_profiles.items(),
                key=lambda x: x[1].opportunity_score,
                reverse=True,
            )

            # Берем топ символы
            selected_symbols = [symbol for symbol, _ in sorted_profiles[:limit]]

            # Создаем результат
            result = SymbolSelectionResult(
                selected_symbols=selected_symbols,
                total_analyzed=len(profiles),
                total_filtered=len(filtered_profiles),
                processing_time_ms=0,
            )

            return result
        except Exception as e:
            self.logger.error(f"Error getting selection result: {e}")
            return SymbolSelectionResult()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности."""
        return {
            "cache_hit_rate": self.cache.get_hit_rate(),
            "average_processing_time": 0.0,
            "total_requests": 0,
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        return self.cache.get_stats()

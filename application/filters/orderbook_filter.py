# -*- coding: utf-8 -*-
"""Order Book Pre-Filter with Neural Noise Analysis."""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, AsyncGenerator, Dict, List, Optional

from loguru import logger

from domain.entities.orderbook import OrderBookSnapshot
from domain.intelligence.noise_analyzer import NoiseAnalysisResult, NoiseAnalyzer
from domain.types import MetadataDict
from domain.types.intelligence_types import (
    AnalysisMetadata,
    NoiseAnalysisConfig,
    NoiseMetrics,
    NoiseType,
)
from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume


@dataclass
class FilterConfig:
    """Конфигурация фильтра ордербука."""

    enabled: bool = True
    fractal_dimension_lower: float = 1.2
    fractal_dimension_upper: float = 1.4
    entropy_threshold: float = 0.7
    min_data_points: int = 50
    window_size: int = 100
    confidence_threshold: float = 0.8
    log_filtered: bool = True
    log_analysis: bool = False


class OrderBookPreFilter:
    """Пре-фильтр ордербуков с анализом нейронного шума."""

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        noise_config = NoiseAnalysisConfig(
            fractal_dimension_lower=self.config.fractal_dimension_lower,
            fractal_dimension_upper=self.config.fractal_dimension_upper,
            entropy_threshold=self.config.entropy_threshold,
            min_data_points=self.config.min_data_points,
            window_size=self.config.window_size,
            confidence_threshold=self.config.confidence_threshold,
        )
        self.noise_analyzer = NoiseAnalyzer(config=noise_config)
        # Статистика фильтрации
        self.stats: Dict[str, Any] = {
            "total_processed": 0,
            "filtered_out": 0,
            "synthetic_noise_detected": 0,
            "last_filter_time": None,
        }
        logger.info(f"OrderBookPreFilter initialized with config: {self.config}")

    def _create_order_book_snapshot(
        self,
        exchange: str,
        symbol: str,
        bids: List[tuple],
        asks: List[tuple],
        timestamp: float,
        sequence_id: Optional[int] = None,
    ) -> OrderBookSnapshot:
        """Создание снимка ордербука из сырых данных."""
        try:
            # Преобразуем bids
            bid_data = [
                (
                    Price(value=Decimal(str(price)), currency=Currency.USD),
                    Volume(value=Decimal(str(volume)), currency=Currency.USD),
                )
                for price, volume in bids
            ]
            # Преобразуем asks
            ask_data = [
                (
                    Price(value=Decimal(str(price)), currency=Currency.USD),
                    Volume(value=Decimal(str(volume)), currency=Currency.USD),
                )
                for price, volume in asks
            ]
            return OrderBookSnapshot(
                exchange=exchange,
                symbol=symbol,
                bids=bid_data,
                asks=ask_data,
                timestamp=Timestamp(timestamp),
                sequence_id=sequence_id,
                meta=MetadataDict({}),
            )
        except Exception as e:
            logger.error(f"Error creating order book snapshot: {e}")
            # Возвращаем пустой снимок
            return OrderBookSnapshot(
                exchange=exchange,
                symbol=symbol,
                bids=[],
                asks=[],
                timestamp=Timestamp(timestamp),
                sequence_id=sequence_id,
                meta=MetadataDict({"error": str(e)}),
            )

    def _apply_noise_analysis(
        self, order_book: OrderBookSnapshot
    ) -> NoiseAnalysisResult:
        """Применение анализа нейронного шума."""
        try:
            if not self.config.enabled:
                # Возвращаем нейтральный результат если фильтр отключен
                metadata: AnalysisMetadata = {
                    "data_points": 0,
                    "confidence": 1.0,
                    "processing_time_ms": 0.0,
                    "algorithm_version": "1.0",
                    "parameters": {"filter_disabled": True},
                    "quality_metrics": {"filter_disabled": True},
                }
                metrics: NoiseMetrics = {
                    "fractal_dimension": 1.0,
                    "entropy": 0.5,
                    "noise_type": NoiseType.NATURAL,
                    "synthetic_probability": 0.0,
                    "natural_probability": 1.0,
                }
                return NoiseAnalysisResult(
                    fractal_dimension=1.0,
                    entropy=0.5,
                    is_synthetic_noise=False,
                    confidence=1.0,
                    metadata=metadata,
                    timestamp=order_book.timestamp,
                    noise_type=NoiseType.NATURAL,
                    metrics=metrics,
                )
            # Выполняем анализ
            # Приводим к правильному типу для analyze_noise
            from domain.types.intelligence_types import OrderBookSnapshot as IntelligenceOrderBookSnapshot
            
            # Создаем объект правильного типа
            intelligence_order_book = IntelligenceOrderBookSnapshot(
                exchange=order_book.exchange,
                symbol=order_book.symbol,
                bids=order_book.bids,
                asks=order_book.asks,
                timestamp=order_book.timestamp,
                sequence_id=order_book.sequence_id,
                meta=order_book.meta
            )
            
            result = self.noise_analyzer.analyze_noise(intelligence_order_book)
            # Логируем анализ если включено
            if self.config.log_analysis:
                logger.debug(
                    f"Noise analysis for {order_book.exchange}:{order_book.symbol}: "
                    f"FD={result.fractal_dimension:.3f}, "
                    f"Entropy={result.entropy:.3f}, "
                    f"Synthetic={result.is_synthetic_noise}, "
                    f"Confidence={result.confidence:.3f}"
                )
            return result
        except Exception as e:
            logger.error(f"Error applying noise analysis: {e}")
            # Возвращаем нейтральный результат при ошибке
            error_metadata: AnalysisMetadata = {
                "data_points": 0,
                "confidence": 0.0,
                "processing_time_ms": 0.0,
                "algorithm_version": "1.0",
                "parameters": {"error": str(e)},
                "quality_metrics": {"error": 0.0},
            }
            error_metrics: NoiseMetrics = {
                "fractal_dimension": 1.0,
                "entropy": 0.5,
                "noise_type": NoiseType.UNKNOWN,
                "synthetic_probability": 0.0,
                "natural_probability": 0.0,
            }
            return NoiseAnalysisResult(
                fractal_dimension=1.0,
                entropy=0.5,
                is_synthetic_noise=False,
                confidence=0.0,
                metadata=error_metadata,
                timestamp=order_book.timestamp,
                noise_type=NoiseType.UNKNOWN,
                metrics=error_metrics,
            )

    def _update_order_book_meta(
        self, order_book: OrderBookSnapshot, analysis_result: NoiseAnalysisResult
    ) -> None:
        """Обновление метаданных ордербука результатами анализа."""
        try:
            # Добавляем результаты анализа в метаданные
            order_book.meta.update(
                {
                    "synthetic_noise": analysis_result.is_synthetic_noise,
                    "noise_analysis": analysis_result.to_dict(),
                    "filtered": analysis_result.is_synthetic_noise,
                    "filter_confidence": analysis_result.confidence,
                }
            )
            # Добавляем статистику фильтра
            order_book.meta["filter_stats"] = {
                "total_processed": self.stats["total_processed"],
                "filtered_out": self.stats["filtered_out"],
                "synthetic_noise_detected": self.stats["synthetic_noise_detected"],
            }
        except Exception as e:
            logger.error(f"Error updating order book meta: {e}")

    def _update_statistics(self, analysis_result: NoiseAnalysisResult) -> None:
        """Обновление статистики фильтрации."""
        self.stats["total_processed"] = int(self.stats.get("total_processed", 0)) + 1
        
        if analysis_result.is_synthetic_noise:
            self.stats["filtered_out"] = int(self.stats.get("filtered_out", 0)) + 1
            self.stats["synthetic_noise_detected"] = int(self.stats.get("synthetic_noise_detected", 0)) + 1
        
        self.stats["last_filter_time"] = datetime.now()
        
        if self.config.log_filtered and analysis_result.is_synthetic_noise:
            logger.info(
                f"Filtered synthetic noise: FD={analysis_result.fractal_dimension:.3f}, "
                f"Entropy={analysis_result.entropy:.3f}, "
                f"Confidence={analysis_result.confidence:.3f}"
            )

    def filter_order_book(
        self,
        exchange: str,
        symbol: str,
        bids: List[tuple],
        asks: List[tuple],
        timestamp: float,
        sequence_id: Optional[int] = None,
    ) -> OrderBookSnapshot:
        """
        Фильтрация ордербука с анализом нейронного шума.
        Args:
            exchange: Название биржи
            symbol: Торговая пара
            bids: Список bid ордеров [(price, volume), ...]
            asks: Список ask ордеров [(price, volume), ...]
            timestamp: Временная метка
            sequence_id: ID последовательности
        Returns:
            OrderBookSnapshot: Отфильтрованный снимок ордербука
        """
        try:
            # Создаем снимок ордербука
            order_book = self._create_order_book_snapshot(
                exchange, symbol, bids, asks, timestamp, sequence_id
            )
            # Применяем анализ нейронного шума
            analysis_result = self._apply_noise_analysis(order_book)
            # Обновляем метаданные
            self._update_order_book_meta(order_book, analysis_result)
            # Обновляем статистику
            self._update_statistics(analysis_result)
            return order_book
        except Exception as e:
            logger.error(f"Error in filter_order_book: {e}")
            # Возвращаем ордербук с ошибкой
            return OrderBookSnapshot(
                exchange=exchange,
                symbol=symbol,
                bids=[],
                asks=[],
                timestamp=Timestamp(datetime.fromtimestamp(timestamp)),
                sequence_id=sequence_id,
                meta=MetadataDict({"error": str(e), "filtered": False}),
            )

    async def filter_order_book_async(
        self,
        exchange: str,
        symbol: str,
        bids: List[tuple],
        asks: List[tuple],
        timestamp: float,
        sequence_id: Optional[int] = None,
    ) -> OrderBookSnapshot:
        """
        Асинхронная фильтрация ордербука.
        Args:
            exchange: Название биржи
            symbol: Торговая пара
            bids: Список bid ордеров [(price, volume), ...]
            asks: Список ask ордеров [(price, volume), ...]
            timestamp: Временная метка
            sequence_id: ID последовательности
        Returns:
            OrderBookSnapshot: Отфильтрованный снимок ордербука
        """
        try:
            # Выполняем фильтрацию в отдельном потоке для избежания блокировки
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.filter_order_book,
                exchange,
                symbol,
                bids,
                asks,
                timestamp,
                sequence_id,
            )
            return result
        except Exception as e:
            logger.error(f"Error in filter_order_book_async: {e}")
            # Возвращаем ордербук с ошибкой
            return OrderBookSnapshot(
                exchange=exchange,
                symbol=symbol,
                bids=[],
                asks=[],
                timestamp=Timestamp(datetime.fromtimestamp(timestamp)),
                sequence_id=sequence_id,
                meta=MetadataDict({"error": str(e), "filtered": False}),
            )

    async def filter_order_book_stream(
        self, order_book_stream: AsyncGenerator[Dict[str, Any], None]
    ) -> AsyncGenerator[OrderBookSnapshot, None]:
        """
        Фильтрация потока ордербуков.
        Args:
            order_book_stream: Поток сырых данных ордербуков
        Yields:
            OrderBookSnapshot: Отфильтрованные снимки ордербуков
        """
        try:
            async for order_book_data in order_book_stream:
                try:
                    # Извлекаем данные
                    exchange = order_book_data.get("exchange", "unknown")
                    symbol = order_book_data.get("symbol", "unknown")
                    bids = order_book_data.get("bids", [])
                    asks = order_book_data.get("asks", [])
                    timestamp = order_book_data.get("timestamp", 0.0)
                    sequence_id = order_book_data.get("sequence_id")
                    # Фильтруем ордербук
                    filtered_order_book = await self.filter_order_book_async(
                        exchange, symbol, bids, asks, timestamp, sequence_id
                    )
                    # Возвращаем результат
                    yield filtered_order_book
                except Exception as e:
                    logger.error(f"Error processing order book in stream: {e}")
                    # Продолжаем обработку потока
                    continue
        except Exception as e:
            logger.error(f"Error in filter_order_book_stream: {e}")

    def get_filter_statistics(self) -> Dict[str, Any]:
        """Получение статистики фильтрации."""
        try:
            stats = self.stats.copy()
            # Добавляем процентные соотношения
            total_processed = stats.get("total_processed", 0)
            filtered_out = stats.get("filtered_out", 0)
            synthetic_noise_detected = stats.get("synthetic_noise_detected", 0)
            if total_processed and total_processed > 0:
                stats["filter_rate"] = filtered_out / total_processed
                stats["synthetic_noise_rate"] = (
                    synthetic_noise_detected / total_processed
                )
            else:
                stats["filter_rate"] = 0.0
                stats["synthetic_noise_rate"] = 0.0
            # Добавляем конфигурацию
            stats["config"] = {
                "enabled": self.config.enabled,
                "fractal_dimension_range": [
                    self.config.fractal_dimension_lower,
                    self.config.fractal_dimension_upper,
                ],
                "entropy_threshold": self.config.entropy_threshold,
                "confidence_threshold": self.config.confidence_threshold,
            }
            # Добавляем статистику анализатора
            try:
                stats["analyzer_stats"] = self.noise_analyzer.get_analysis_statistics()
            except AttributeError:
                stats["analyzer_stats"] = {"error": "analyzer_stats_not_available"}
            return stats
        except Exception as e:
            logger.error(f"Error getting filter statistics: {e}")
            return {"error": str(e)}

    def reset_statistics(self) -> None:
        """Сброс статистики фильтрации."""
        try:
            self.stats.clear()
            self.stats.update({
                "total_processed": 0,
                "filtered_out": 0,
                "synthetic_noise_detected": 0,
                "last_filter_time": None,
            })
            self.noise_analyzer.reset_history()
            logger.info("OrderBookPreFilter statistics reset")
        except Exception as e:
            logger.error(f"Error resetting statistics: {e}")

    def update_config(self, new_config: FilterConfig) -> None:
        """Обновление конфигурации фильтра."""
        try:
            self.config = new_config
            # Пересоздаем анализатор с новой конфигурацией
            noise_config = NoiseAnalysisConfig(
                fractal_dimension_lower=self.config.fractal_dimension_lower,
                fractal_dimension_upper=self.config.fractal_dimension_upper,
                entropy_threshold=self.config.entropy_threshold,
                min_data_points=self.config.min_data_points,
                window_size=self.config.window_size,
                confidence_threshold=self.config.confidence_threshold,
            )
            self.noise_analyzer = NoiseAnalyzer(config=noise_config)
            logger.info(f"OrderBookPreFilter config updated: {self.config}")
        except Exception as e:
            logger.error(f"Error updating config: {e}")

    def is_order_book_filtered(self, order_book: OrderBookSnapshot) -> bool:
        """Проверка, был ли ордербук отфильтрован."""
        try:
            return order_book.meta.get("filtered", False)
        except Exception as e:
            logger.error(f"Error checking if order book is filtered: {e}")
            return False

    def get_noise_analysis_result(
        self, order_book: OrderBookSnapshot
    ) -> Optional[Dict[str, Any]]:
        """Получение результатов анализа нейронного шума для ордербука."""
        try:
            return order_book.meta.get("noise_analysis")
        except Exception as e:
            logger.error(f"Error getting noise analysis result: {e}")
            return None

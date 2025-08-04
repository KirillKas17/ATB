# -*- coding: utf-8 -*-
"""Quantum Entanglement Detection for Cross-Exchange Order Books."""
import logging
import time
from collections import deque
from typing import Any, Dict, Final, List, Optional, Tuple, cast

from shared.numpy_utils import np

# Настройка логгера
logger = logging.getLogger(__name__)
from domain.type_definitions.intelligence_types import (
    AnalysisMetadata,
    CorrelationMatrix,
    CorrelationMethod,
    EntanglementConfig,
    EntanglementResult,
    EntanglementStrength,
    EntanglementType,
    OrderBookSnapshot,
    OrderBookUpdate,
)
from domain.value_objects.timestamp import Timestamp

# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_CONFIG: Final[EntanglementConfig] = EntanglementConfig()
ALGORITHM_VERSION: Final[str] = "2.0.0"
MIN_CORRELATION_THRESHOLD: Final[float] = 0.7
MAX_LAG_MS: Final[float] = 10.0
MIN_DATA_POINTS: Final[int] = 50


# =============================================================================
# ENHANCED ENTANGLEMENT DETECTOR
# =============================================================================
class EntanglementDetector:
    """Продвинутый детектор квантовой запутанности ордербуков."""

    def __init__(
        self,
        config: Optional[EntanglementConfig] = None,
        enable_advanced_metrics: bool = True,
        enable_cross_correlation: bool = True,
    ):
        self.config = config or DEFAULT_CONFIG
        self.enable_advanced_metrics = enable_advanced_metrics
        self.enable_cross_correlation = enable_cross_correlation
        # Буферы для исторических данных
        self.price_buffers: Dict[str, deque] = {}
        self.volume_buffers: Dict[str, deque] = {}
        self.spread_buffers: Dict[str, deque] = {}
        self.timestamp_buffers: Dict[str, deque] = {}
        # Статистика детектора
        self.statistics: Dict[str, Any] = {
            "total_analyses": 0,
            "entanglement_detections": 0,
            "average_processing_time_ms": 0.0,
            "last_analysis_timestamp": None,
        }
        logger.info(
            f"EntanglementDetector initialized with config: {self.config}, "
            f"advanced_metrics: {enable_advanced_metrics}, "
            f"cross_correlation: {enable_cross_correlation}"
        )

    def detect_entanglement(
        self,
        symbol: str,
        exchange_data: Dict[str, Any],
        max_lag_ms: Optional[float] = None,
        correlation_threshold: Optional[float] = None,
    ) -> EntanglementResult:
        """
        Обнаружение запутанности между биржами для символа.
        Args:
            symbol: Торговый символ
            exchange_data: Данные с разных бирж
            max_lag_ms: Максимальный лаг в миллисекундах
            correlation_threshold: Порог корреляции
        Returns:
            EntanglementResult с результатами анализа
        """
        start_time = time.time()
        try:
            # Используем конфигурационные значения по умолчанию
            max_lag_ms = max_lag_ms or self.config.max_lag_ms
            correlation_threshold = (
                correlation_threshold or self.config.correlation_threshold
            )
            # Проверяем достаточность данных
            if len(exchange_data) < 2:
                return self._create_default_result(
                    symbol, "insufficient_exchanges", start_time
                )
            # Извлекаем данные ордербуков
            order_books = self._extract_order_books(exchange_data)
            if len(order_books) < 2:
                return self._create_default_result(
                    symbol, "insufficient_orderbooks", start_time
                )
            # Анализируем запутанность
            correlation_matrix = self._compute_correlation_matrix(order_books)
            lag_matrix = self._compute_lag_matrix(order_books)
            # Определяем наиболее запутанную пару
            best_pair, best_correlation, best_lag = self._find_best_entanglement_pair(
                correlation_matrix, lag_matrix, max_lag_ms
            )
            # Проверяем порог корреляции
            is_entangled = best_correlation >= correlation_threshold
            # Вычисляем уверенность
            confidence = self._calculate_confidence(
                best_correlation, best_lag, len(order_books)
            )
            # Создаем метаданные
            metadata = cast(
                AnalysisMetadata,
                {
                    "data_points": len(order_books),
                    "confidence": confidence,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "algorithm_version": ALGORITHM_VERSION,
                    "parameters": {
                        "max_lag_ms": max_lag_ms,
                        "correlation_threshold": correlation_threshold,
                        "num_exchanges": len(order_books),
                    },
                    "quality_metrics": {
                        "best_correlation": best_correlation,
                        "best_lag": best_lag,
                    },
                },
            )
            # Создаем результат
            result = EntanglementResult(
                symbol=symbol,
                is_entangled=is_entangled,
                correlation_score=best_correlation,
                lag_ms=best_lag,
                confidence=confidence,
                exchange_pair=best_pair,
                timestamp=Timestamp.now(),
                metadata=metadata,
            )
            # Обновляем статистику
            self._update_statistics(result, start_time)
            # Логируем результат
            if is_entangled:
                logger.warning(
                    f"ENTANGLEMENT detected for {symbol}: "
                    f"exchanges={best_pair}, correlation={best_correlation:.3f}, "
                    f"lag={best_lag:.2f}ms, confidence={confidence:.3f}"
                )
            else:
                logger.debug(
                    f"No entanglement for {symbol}: "
                    f"best_correlation={best_correlation:.3f}, "
                    f"threshold={correlation_threshold:.3f}"
                )
            return result
        except Exception as e:
            logger.error(f"Error detecting entanglement for {symbol}: {e}")
            return self._create_default_result(symbol, "error", start_time)

    def _extract_order_books(
        self, exchange_data: Dict[str, Any]
    ) -> Dict[str, OrderBookSnapshot]:
        """Извлечение ордербуков из данных бирж."""
        order_books = {}
        for exchange, data in exchange_data.items():
            try:
                if not isinstance(data, dict):
                    continue
                # Извлекаем данные ордербука
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                timestamp = data.get("timestamp")
                if not bids or not asks or not timestamp:
                    continue
                # Создаем снимок ордербука
                order_book = OrderBookSnapshot(
                    symbol=data.get("symbol", "unknown"),
                    exchange=exchange,
                    timestamp=(
                        Timestamp.from_iso(timestamp)
                        if isinstance(timestamp, str)
                        else Timestamp.now()
                    ),
                    bids=bids,
                    asks=asks,
                    meta=data.get("metadata", {}),
                )
                order_books[exchange] = order_book
            except Exception as e:
                logger.warning(f"Error extracting orderbook from {exchange}: {e}")
                continue
        return order_books

    def _compute_correlation_matrix(
        self, order_books: Dict[str, OrderBookSnapshot]
    ) -> Dict[Tuple[str, str], float]:
        """Вычисление матрицы корреляций между биржами."""
        correlation_matrix = {}
        exchanges = list(order_books.keys())
        for i, exchange1 in enumerate(exchanges):
            for j, exchange2 in enumerate(exchanges[i + 1 :], i + 1):
                try:
                    # Извлекаем цены
                    prices1 = self._extract_prices(order_books[exchange1])
                    prices2 = self._extract_prices(order_books[exchange2])
                    if len(prices1) < MIN_DATA_POINTS or len(prices2) < MIN_DATA_POINTS:
                        correlation_matrix[(exchange1, exchange2)] = 0.0
                        continue
                    # Выравниваем длины
                    min_length = min(len(prices1), len(prices2))
                    prices1 = prices1[-min_length:]
                    prices2 = prices2[-min_length:]
                    # Вычисляем корреляцию
                    correlation = self._calculate_correlation(prices1, prices2)
                    correlation_matrix[(exchange1, exchange2)] = correlation
                except Exception as e:
                    logger.warning(
                        f"Error computing correlation {exchange1}-{exchange2}: {e}"
                    )
                    correlation_matrix[(exchange1, exchange2)] = 0.0
        return correlation_matrix

    def _extract_prices(self, order_book: OrderBookSnapshot) -> List[float]:
        """Извлечение цен из ордербука."""
        prices = []
        # Добавляем bid цены
        for bid_price, _ in order_book.bids:
            prices.append(float(bid_price.amount))
        # Добавляем ask цены
        for ask_price, _ in order_book.asks:
            prices.append(float(ask_price.amount))
        return prices

    def _calculate_correlation(
        self, prices1: List[float], prices2: List[float]
    ) -> float:
        """Расчет корреляции между двумя списками цен."""
        if len(prices1) < 2 or len(prices2) < 2:
            return 0.0
        # Приводим к одинаковой длине
        min_length = min(len(prices1), len(prices2))
        prices1 = prices1[:min_length]
        prices2 = prices2[:min_length]
        try:
            correlation = np.corrcoef(prices1, prices2)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0

    def _compute_lag_matrix(
        self, order_books: Dict[str, OrderBookSnapshot]
    ) -> Dict[Tuple[str, str], float]:
        """Вычисление матрицы лагов между биржами."""
        lag_matrix = {}
        exchanges = list(order_books.keys())
        for i, exchange1 in enumerate(exchanges):
            for j, exchange2 in enumerate(exchanges):
                if i >= j:
                    continue
                try:
                    # Извлекаем временные метки
                    timestamp1 = order_books[exchange1].timestamp
                    timestamp2 = order_books[exchange2].timestamp
                    # Вычисляем лаг в миллисекундах
                    lag_seconds = timestamp1.time_difference(timestamp2)
                    lag_ms = abs(lag_seconds * 1000)
                    lag_matrix[(exchange1, exchange2)] = lag_ms
                    lag_matrix[(exchange2, exchange1)] = lag_ms
                except Exception as e:
                    logger.warning(
                        f"Error computing lag between {exchange1} and {exchange2}: {e}"
                    )
                    lag_matrix[(exchange1, exchange2)] = float("inf")
                    lag_matrix[(exchange2, exchange1)] = float("inf")
        return lag_matrix

    def _find_best_entanglement_pair(
        self,
        correlation_matrix: Dict[Tuple[str, str], float],
        lag_matrix: Dict[Tuple[str, str], float],
        max_lag_ms: float,
    ) -> Tuple[Tuple[str, str], float, float]:
        """Поиск лучшей пары запутанности."""
        best_pair = ("", "")
        best_correlation = 0.0
        best_lag = float("inf")
        for (exchange1, exchange2), correlation in correlation_matrix.items():
            lag = lag_matrix.get((exchange1, exchange2), float("inf"))
            # Проверяем ограничения
            if lag > max_lag_ms:
                continue
            # Ищем максимальную корреляцию
            if correlation > best_correlation:
                best_pair = (exchange1, exchange2)
                best_correlation = correlation
                best_lag = lag
        return best_pair, best_correlation, best_lag

    def _calculate_confidence(
        self, correlation: float, lag_ms: float, num_exchanges: int
    ) -> float:
        """Вычисление уверенности в результате."""
        # Базовая уверенность на основе корреляции
        correlation_confidence = correlation
        # Штраф за лаг (чем больше лаг, тем меньше уверенность)
        lag_penalty = max(0, 1 - (lag_ms / 100))  # Нормализуем к 100мс
        # Бонус за количество бирж
        exchange_bonus = min(0.1, (num_exchanges - 2) * 0.02)
        # Итоговая уверенность
        confidence = correlation_confidence * lag_penalty + exchange_bonus
        return min(1.0, max(0.0, confidence))

    def _create_analysis_metadata(
        self,
        order_books: Dict[str, OrderBookSnapshot],
        correlation_matrix: Dict[Tuple[str, str], float],
        lag_matrix: Dict[Tuple[str, str], float],
        start_time: float,
    ) -> Dict[str, Any]:
        """Создание метаданных анализа."""
        processing_time = (time.time() - start_time) * 1000
        return {
            "data_points": len(order_books),
            "confidence": 0.8,
            "processing_time_ms": processing_time,
            "algorithm_version": ALGORITHM_VERSION,
            "parameters": {
                "max_lag_ms": self.config.max_lag_ms,
                "correlation_threshold": self.config.correlation_threshold,
                "window_size": self.config.window_size,
            },
            "quality_metrics": {
                "avg_correlation": (
                    sum(correlation_matrix.values()) / len(correlation_matrix)
                    if correlation_matrix
                    else 0
                ),
                "avg_lag_ms": (
                    sum(lag_matrix.values()) / len(lag_matrix) if lag_matrix else 0
                ),
                "num_exchanges": len(order_books),
            },
        }

    def _create_default_result(
        self, symbol: str, reason: str, start_time: float
    ) -> EntanglementResult:
        """Создание результата по умолчанию."""
        processing_time = (time.time() - start_time) * 1000
        metadata = cast(
            AnalysisMetadata,
            {
                "data_points": 0,
                "confidence": 0.0,
                "processing_time_ms": processing_time,
                "algorithm_version": ALGORITHM_VERSION,
                "parameters": {
                    "max_lag_ms": self.config.max_lag_ms,
                    "correlation_threshold": self.config.correlation_threshold,
                },
                "quality_metrics": {
                    "error_reason": reason,
                },
            },
        )
        return EntanglementResult(
            symbol=symbol,
            is_entangled=False,
            correlation_score=0.0,
            lag_ms=0.0,
            confidence=0.0,
            exchange_pair=("", ""),
            timestamp=Timestamp.now(),
            metadata=metadata,
        )

    def _update_statistics(self, result: EntanglementResult, start_time: float) -> None:
        """Обновление статистики детектора."""
        self.statistics["total_analyses"] += 1
        if result.is_entangled:
            self.statistics["entanglement_detections"] += 1
        # Обновляем среднее время обработки
        processing_time = (time.time() - start_time) * 1000
        current_avg = self.statistics["average_processing_time_ms"]
        total_analyses = self.statistics["total_analyses"]
        self.statistics["average_processing_time_ms"] = (
            current_avg * (total_analyses - 1) + processing_time
        ) / total_analyses
        self.statistics["last_analysis_timestamp"] = time.time()

    def get_detector_statistics(self) -> Dict[str, Any]:
        """Получение статистики детектора."""
        return {
            **self.statistics,
            "config": {
                "correlation_threshold": self.config.correlation_threshold,
                "max_lag_ms": self.config.max_lag_ms,
                "min_data_points": self.config.min_data_points,
            },
            "advanced_metrics_enabled": self.enable_advanced_metrics,
            "cross_correlation_enabled": self.enable_cross_correlation,
        }

    def reset_statistics(self) -> None:
        """Сброс статистики."""
        self.statistics = {
            "total_analyses": 0,
            "entanglement_detections": 0,
            "average_processing_time_ms": 0.0,
            "last_analysis_timestamp": None,
        }
        logger.info("Entanglement detector statistics reset")

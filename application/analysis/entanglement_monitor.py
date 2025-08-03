# -*- coding: utf-8 -*-
"""Entanglement monitoring service for cross-exchange order book analysis."""

import asyncio
import json
import time
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from application.entanglement.stream_manager import StreamManager
from domain.intelligence.entanglement_detector import (
    EntanglementConfig,
    EntanglementDetector,
    EntanglementResult,
    OrderBookUpdate,
)
from domain.protocols.exchange_protocols import MarketDataConnectorProtocol


@dataclass
class ExchangePair:
    """Пара бирж для мониторинга."""

    exchange1: str
    exchange2: str
    symbol: str
    is_active: bool = True


class EntanglementMonitor:
    """Координационный сервис для мониторинга запутанности ордеров."""

    def __init__(
        self,
        connectors: Dict[str, MarketDataConnectorProtocol],
        stream_manager: Optional[StreamManager] = None,
        log_file_path: str = "logs/entanglement_events.json",
        detection_interval: float = 1.0,
        max_lag_ms: float = 3.0,
        correlation_threshold: float = 0.95,
        enable_new_exchanges: bool = True,
    ):
        self.detector = EntanglementDetector(
            config=EntanglementConfig(
                max_lag_ms=max_lag_ms, correlation_threshold=correlation_threshold
            )
        )

        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        self.detection_interval = detection_interval
        self.is_running = False
        self.enable_new_exchanges = enable_new_exchanges

        # Коннекторы к биржам (legacy)
        self.connectors = connectors
        self.exchange_pairs: List[ExchangePair] = []

        # Новый StreamManager для расширенных бирж
        self.stream_manager = stream_manager

        # Буферы для обновлений ордербуков (legacy)
        self.order_book_buffers: Dict[str, List[OrderBookUpdate]] = {}
        self.buffer_max_size = 100

        # Статистика
        self.stats = {
            "total_detections": 0,
            "entangled_detections": 0,
            "last_detection_time": 0.0,
            "start_time": time.time(),
            "exchanges_monitored": 0,
        }

        # Настройка пар бирж для мониторинга
        self._setup_exchange_pairs()

    def _setup_exchange_pairs(self):
        """Настройка пар бирж для мониторинга."""
        # Основные пары для мониторинга (legacy)
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]

        for symbol in symbols:
            # Binance ↔ Coinbase
            self.exchange_pairs.append(ExchangePair("binance", "coinbase", symbol))

            # Binance ↔ Kraken
            self.exchange_pairs.append(ExchangePair("binance", "kraken", symbol))

            # Coinbase ↔ Kraken
            self.exchange_pairs.append(ExchangePair("coinbase", "kraken", symbol))

        # Добавляем пары с новыми биржами
        if self.enable_new_exchanges:
            new_exchanges = ["bingx", "bitget", "bybit"]
            for symbol in symbols:
                for i, exchange1 in enumerate(new_exchanges):
                    for exchange2 in new_exchanges[i + 1 :]:
                        self.exchange_pairs.append(
                            ExchangePair(exchange1, exchange2, symbol)
                        )

                # Смешанные пары (legacy ↔ new)
                for legacy_exchange in ["binance", "coinbase", "kraken"]:
                    for new_exchange in new_exchanges:
                        self.exchange_pairs.append(
                            ExchangePair(legacy_exchange, new_exchange, symbol)
                        )

        self.stats["exchanges_monitored"] = len(
            set(
                [pair.exchange1 for pair in self.exchange_pairs]
                + [pair.exchange2 for pair in self.exchange_pairs]
            )
        )

        logger.info(f"Setup {len(self.exchange_pairs)} exchange pairs for monitoring")

    def _log_entanglement_event(self, result: EntanglementResult):
        """Логирование события запутанности."""
        try:
            event = {
                "event_type": (
                    "entanglement_detected"
                    if result.is_entangled
                    else "entanglement_analysis"
                ),
                "timestamp": result.timestamp.value.isoformat(),
                "exchange_pair": result.exchange_pair,
                "symbol": result.symbol,
                "is_entangled": result.is_entangled,
                "correlation_score": result.correlation_score,
                "lag_ms": result.lag_ms,
                "confidence": result.confidence,
                "metadata": result.metadata,
            }

            # Записываем в файл
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

            # Обновляем статистику
            self.stats["total_detections"] += 1
            if result.is_entangled:
                self.stats["entangled_detections"] += 1
            self.stats["last_detection_time"] = time.time()

        except Exception as e:
            logger.error(f"Failed to log entanglement event: {e}")

    async def _handle_entanglement_result(self, result: EntanglementResult):
        """Обработка результата запутанности от StreamManager."""
        try:
            # Логируем событие
            self._log_entanglement_event(result)

            # Обновляем статистику
            self.stats["total_detections"] += 1
            if result.is_entangled:
                self.stats["entangled_detections"] += 1
                logger.warning(
                    f"ENTANGLED DETECTED (NEW): {result.exchange_pair[0]} ↔ {result.exchange_pair[1]} "
                    f"({result.symbol}) - Lag: {result.lag_ms:.2f}ms, "
                    f"Correlation: {result.correlation_score:.3f}, "
                    f"Confidence: {result.confidence:.3f}"
                )

        except Exception as e:
            logger.error(f"Error handling entanglement result: {e}")

    def _add_order_book_update(self, update: OrderBookUpdate):
        """Добавление обновления ордербука в буфер (legacy)."""
        exchange = update.exchange

        if exchange not in self.order_book_buffers:
            self.order_book_buffers[exchange] = []

        buffer = self.order_book_buffers[exchange]
        buffer.append(update)

        # Ограничиваем размер буфера
        if len(buffer) > self.buffer_max_size:
            buffer.pop(0)

    def _get_recent_updates(
        self, exchange: str, count: int = 10
    ) -> List[OrderBookUpdate]:
        """Получение последних обновлений для биржи (legacy)."""
        buffer = self.order_book_buffers.get(exchange, [])
        return buffer[-count:] if buffer else []

    async def _monitor_exchange_pair(self, pair: ExchangePair):
        """Мониторинг конкретной пары бирж (legacy)."""
        exchange1, exchange2 = pair.exchange1, pair.exchange2

        while self.is_running and pair.is_active:
            try:
                # Получаем последние обновления
                updates1 = self._get_recent_updates(exchange1)
                updates2 = self._get_recent_updates(exchange2)

                if not updates1 or not updates2:
                    await asyncio.sleep(self.detection_interval)
                    continue

                # Объединяем обновления для анализа
                all_updates = updates1 + updates2

                # Обнаруживаем запутанность
                # Преобразуем обновления в формат для детектора
                exchange_data = {
                    exchange1: {
                        "bids": updates1[-1].bids if updates1 else [],
                        "asks": updates1[-1].asks if updates1 else [],
                        "timestamp": (
                            updates1[-1].timestamp.to_iso() if updates1 else ""
                        ),
                        "symbol": pair.symbol,
                    },
                    exchange2: {
                        "bids": updates2[-1].bids if updates2 else [],
                        "asks": updates2[-1].asks if updates2 else [],
                        "timestamp": (
                            updates2[-1].timestamp.to_iso() if updates2 else ""
                        ),
                        "symbol": pair.symbol,
                    },
                }

                result = self.detector.detect_entanglement(pair.symbol, exchange_data)

                # Логируем результат
                if result:
                    self._log_entanglement_event(result)

                    if result.is_entangled:
                        logger.warning(
                            f"ENTANGLED DETECTED (LEGACY): {exchange1} ↔ {exchange2} "
                            f"({pair.symbol}) - Lag: {result.lag_ms:.2f}ms, "
                            f"Correlation: {result.correlation_score:.3f}"
                        )

                await asyncio.sleep(self.detection_interval)

            except Exception as e:
                logger.error(f"Error monitoring pair {exchange1} ↔ {exchange2}: {e}")
                await asyncio.sleep(self.detection_interval)

    async def _simulate_order_book_updates(self):
        """Симуляция обновлений ордербуков для тестирования."""

        while self.is_running:
            try:
                # Симулируем обновления для каждой биржи
                for exchange_name in ["binance", "coinbase", "kraken"]:
                    # Создаем фейковое обновление
                    base_price = 50000 + random.uniform(-1000, 1000)

                    bids = [
                        (Price(base_price - i * 10), Volume(random.uniform(0.1, 2.0)))
                        for i in range(1, 6)
                    ]

                    asks = [
                        (Price(base_price + i * 10), Volume(random.uniform(0.1, 2.0)))
                        for i in range(1, 6)
                    ]

                    update = OrderBookUpdate(
                        exchange=exchange_name,
                        symbol="BTCUSDT",
                        bids=bids,
                        asks=asks,
                        timestamp=Timestamp(time.time()),
                    )

                    self._add_order_book_update(update)

                await asyncio.sleep(0.1)  # 10 обновлений в секунду

            except Exception as e:
                logger.error(f"Error in order book simulation: {e}")
                await asyncio.sleep(1.0)

    async def start_monitoring(self):
        """Запуск мониторинга запутанности."""
        if self.is_running:
            logger.warning("Monitoring is already running")
            return

        self.is_running = True
        logger.info("Starting entanglement monitoring...")

        try:
            # Запускаем задачи мониторинга
            tasks = []

            # Задача симуляции обновлений ордербуков
            tasks.append(asyncio.create_task(self._simulate_order_book_updates()))

            # Задачи мониторинга пар бирж
            for pair in self.exchange_pairs:
                task = asyncio.create_task(self._monitor_exchange_pair(pair))
                tasks.append(task)

            # Ждем завершения всех задач
            await asyncio.gather(*tasks)

        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        finally:
            self.is_running = False
            logger.info("Entanglement monitoring stopped")

    def stop_monitoring(self):
        """Остановка мониторинга."""
        self.is_running = False
        logger.info("Stopping entanglement monitoring...")

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса мониторинга."""
        return {
            "is_running": self.is_running,
            "stats": self.stats.copy(),
            "active_pairs": len([p for p in self.exchange_pairs if p.is_active]),
            "total_pairs": len(self.exchange_pairs),
            "buffer_sizes": {
                exchange: len(buffer)
                for exchange, buffer in self.order_book_buffers.items()
            },
            "detector_status": self.detector.get_detector_statistics(),
        }

    def add_exchange_pair(self, exchange1: str, exchange2: str, symbol: str):
        """Добавление новой пары бирж для мониторинга."""
        pair = ExchangePair(exchange1, exchange2, symbol)
        self.exchange_pairs.append(pair)
        logger.info(f"Added exchange pair: {exchange1} ↔ {exchange2} ({symbol})")

    def remove_exchange_pair(self, exchange1: str, exchange2: str, symbol: str):
        """Удаление пары бирж из мониторинга."""
        self.exchange_pairs = [
            p
            for p in self.exchange_pairs
            if not (
                p.exchange1 == exchange1
                and p.exchange2 == exchange2
                and p.symbol == symbol
            )
        ]
        logger.info(f"Removed exchange pair: {exchange1} ↔ {exchange2} ({symbol})")

    def get_entanglement_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Получение истории запутанности."""
        try:
            history: List[Dict[str, Any]] = []
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(history) >= limit:
                        break
                    try:
                        event = json.loads(line.strip())
                        if event.get("event_type") in ["entanglement_detected", "entanglement_analysis"]:
                            history.append({
                                "timestamp": event["timestamp"],
                                "entanglement_score": event.get("correlation_score", 0.0)
                            })
                    except json.JSONDecodeError:
                        continue
            return history[-limit:]  # Возвращаем последние записи
        except FileNotFoundError:
            return []

    async def analyze_entanglement(self, symbol1: str, symbol2: str, timeframe: str) -> Dict[str, Any]:
        """Анализ запутанности между двумя символами."""
        try:
            # Получаем исторические данные для обоих символов
            prices1 = await self._get_historical_prices(symbol1, timeframe)
            prices2 = await self._get_historical_prices(symbol2, timeframe)
            
            if not self.validate_data(prices1) or not self.validate_data(prices2):
                return {
                    "entanglement_score": 0.0,
                    "correlation": 0.0,
                    "phase_shift": 0.0,
                    "confidence": 0.0,
                    "error": "Insufficient data"
                }
            
            # Выравниваем длины массивов
            min_length = min(len(prices1), len(prices2))
            prices1 = prices1[-min_length:]
            prices2 = prices2[-min_length:]
            
            # Рассчитываем компоненты запутанности
            correlation = self.calculate_correlation(prices1, prices2)
            phase_shift = self.calculate_phase_shift(prices1, prices2)
            volatility_ratio = self.calculate_volatility_ratio(prices1, prices2)
            
            # Рассчитываем общий score запутанности
            entanglement_score = self.calculate_entanglement_score(
                correlation, phase_shift, volatility_ratio
            )
            
            # Рассчитываем доверительный интервал
            confidence_interval = self.calculate_confidence_interval(
                prices1, prices2, 0.95
            )
            
            return {
                "entanglement_score": entanglement_score,
                "correlation": correlation,
                "phase_shift": phase_shift,
                "volatility_ratio": volatility_ratio,
                "confidence": confidence_interval,
                "symbol1": symbol1,
                "symbol2": symbol2,
                "timeframe": timeframe,
                "data_points": min_length
            }
        except Exception as e:
            logger.error(f"Error analyzing entanglement: {e}")
            return {
                "entanglement_score": 0.0,
                "correlation": 0.0,
                "phase_shift": 0.0,
                "confidence": 0.0,
                "error": str(e)
            }

    async def analyze_correlations(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Анализ корреляций между символами."""
        try:
            if len(symbols) < 2:
                return {
                    "correlation_matrix": {},
                    "strong_correlations": [],
                    "weak_correlations": [],
                    "correlation_clusters": [],
                    "error": "Need at least 2 symbols"
                }
            
            # Получаем исторические данные для всех символов
            price_data = {}
            for symbol in symbols:
                prices = await self._get_historical_prices(symbol, timeframe)
                if self.validate_data(prices):
                    price_data[symbol] = prices
            
            if len(price_data) < 2:
                return {
                    "correlation_matrix": {},
                    "strong_correlations": [],
                    "weak_correlations": [],
                    "correlation_clusters": [],
                    "error": "Insufficient valid data"
                }
            
            # Выравниваем длины всех массивов
            min_length = min(len(prices) for prices in price_data.values())
            aligned_data = {symbol: prices[-min_length:] for symbol, prices in price_data.items()}
            
            # Строим матрицу корреляций
            correlation_matrix: Dict[str, Dict[str, float]] = {}
            strong_correlations = []
            weak_correlations = []
            
            symbols_list = list(aligned_data.keys())
            for i, symbol1 in enumerate(symbols_list):
                correlation_matrix[symbol1] = {}
                for j, symbol2 in enumerate(symbols_list):
                    if i == j:
                        correlation_matrix[symbol1][symbol2] = 1.0
                    else:
                        correlation = self.calculate_correlation(
                            aligned_data[symbol1], aligned_data[symbol2]
                        )
                        correlation_matrix[symbol1][symbol2] = correlation
                        
                        # Классифицируем корреляции
                        if abs(correlation) >= 0.7:
                            strong_correlations.append({
                                "symbol1": symbol1,
                                "symbol2": symbol2,
                                "correlation": correlation
                            })
                        elif abs(correlation) <= 0.3:
                            weak_correlations.append({
                                "symbol1": symbol1,
                                "symbol2": symbol2,
                                "correlation": correlation
                            })
            
            # Обнаруживаем кластеры корреляции
            correlation_clusters = self.detect_correlation_clusters(correlation_matrix, threshold=0.6)
            
            return {
                "correlation_matrix": correlation_matrix,
                "strong_correlations": strong_correlations,
                "weak_correlations": weak_correlations,
                "correlation_clusters": correlation_clusters,
                "symbols": symbols,
                "timeframe": timeframe,
                "data_points": min_length
            }
        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return {
                "correlation_matrix": {},
                "strong_correlations": [],
                "weak_correlations": [],
                "correlation_clusters": [],
                "error": str(e)
            }

    def calculate_correlation(self, prices1: List[Any], prices2: List[Any]) -> float:
        """Расчет корреляции между двумя рядами цен."""
        try:
            if not self.validate_data(prices1) or not self.validate_data(prices2):
                return 0.0
            
            # Выравниваем длины
            min_length = min(len(prices1), len(prices2))
            if min_length < 2:
                return 0.0
            
            prices1 = prices1[-min_length:]
            prices2 = prices2[-min_length:]
            
            # Конвертируем в числовые значения
            try:
                p1 = [float(p) for p in prices1]
                p2 = [float(p) for p in prices2]
            except (ValueError, TypeError):
                return 0.0
            
            # Рассчитываем средние значения
            mean1 = sum(p1) / len(p1)
            mean2 = sum(p2) / len(p2)
            
            # Рассчитываем ковариацию и стандартные отклонения
            covariance = sum((p1[i] - mean1) * (p2[i] - mean2) for i in range(len(p1)))
            variance1 = sum((p1[i] - mean1) ** 2 for i in range(len(p1)))
            variance2 = sum((p2[i] - mean2) ** 2 for i in range(len(p2)))
            
            # Проверяем деление на ноль
            if variance1 == 0 or variance2 == 0:
                return 0.0
            
            # Рассчитываем корреляцию Пирсона
            correlation = covariance / (variance1 ** 0.5 * variance2 ** 0.5)
            
            # Ограничиваем результат в диапазоне [-1, 1]
            return max(-1.0, min(1.0, correlation))
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.0

    def calculate_phase_shift(self, prices1: List[Any], prices2: List[Any]) -> float:
        """Расчет фазового сдвига между двумя рядами цен."""
        try:
            if not self.validate_data(prices1) or not self.validate_data(prices2):
                return 0.0
            
            # Выравниваем длины
            min_length = min(len(prices1), len(prices2))
            if min_length < 3:
                return 0.0
            
            prices1 = prices1[-min_length:]
            prices2 = prices2[-min_length:]
            
            # Конвертируем в числовые значения
            try:
                p1 = [float(p) for p in prices1]
                p2 = [float(p) for p in prices2]
            except (ValueError, TypeError):
                return 0.0
            
            # Рассчитываем изменения цен (returns)
            returns1 = [(p1[i] - p1[i-1]) / p1[i-1] if p1[i-1] != 0 else 0 for i in range(1, len(p1))]
            returns2 = [(p2[i] - p2[i-1]) / p2[i-1] if p2[i-1] != 0 else 0 for i in range(1, len(p2))]
            
            # Находим максимальную корреляцию при различных сдвигах
            max_correlation = 0.0
            best_shift = 0
            
            max_shift = min(10, len(returns1) // 2)  # Ограничиваем сдвиг
            
            for shift in range(-max_shift, max_shift + 1):
                if shift <= 0:
                    # Сдвигаем первый ряд влево
                    shifted1 = returns1[-shift:] if shift < 0 else returns1
                    shifted2 = returns2[:len(shifted1)]
                else:
                    # Сдвигаем второй ряд влево
                    shifted2 = returns2[shift:]
                    shifted1 = returns1[:len(shifted2)]
                
                if len(shifted1) < 2 or len(shifted2) < 2:
                    continue
                
                # Рассчитываем корреляцию для данного сдвига
                correlation = self.calculate_correlation(shifted1, shifted2)
                if abs(correlation) > abs(max_correlation):
                    max_correlation = correlation
                    best_shift = shift
            
            # Нормализуем сдвиг относительно длины данных
            normalized_shift = best_shift / len(returns1) if len(returns1) > 0 else 0.0
            
            return normalized_shift
        except Exception as e:
            logger.error(f"Error calculating phase shift: {e}")
            return 0.0

    def calculate_entanglement_score(self, correlation: float, phase_shift: float, volatility_ratio: float) -> float:
        """Расчет оценки запутанности."""
        try:
            # Нормализуем входные параметры
            # Корреляция уже в диапазоне [-1, 1]
            # Фазовый сдвиг нормализуем к [0, 1]
            normalized_phase_shift = abs(phase_shift)
            # Отношение волатильности нормализуем к [0, 1]
            normalized_volatility_ratio = min(volatility_ratio, 1.0) if volatility_ratio > 0 else 0.0
            
            # Веса для компонентов (можно настраивать)
            correlation_weight = 0.5
            phase_shift_weight = 0.3
            volatility_weight = 0.2
            
            # Рассчитываем взвешенную сумму
            score = (
                correlation_weight * abs(correlation) +
                phase_shift_weight * (1.0 - normalized_phase_shift) +  # Меньший сдвиг = больше запутанность
                volatility_weight * normalized_volatility_ratio
            )
            
            # Ограничиваем результат в диапазоне [0, 1]
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.error(f"Error calculating entanglement score: {e}")
            return 0.0

    def detect_correlation_clusters(self, correlation_matrix: Dict[str, Dict[str, float]], threshold: float = 0.6) -> List[List[str]]:
        """Обнаружение кластеров корреляции."""
        try:
            if not correlation_matrix:
                return []
            
            symbols = list(correlation_matrix.keys())
            if len(symbols) < 2:
                return []
            
            # Строим граф корреляций
            clusters = []
            visited = set()
            
            for symbol in symbols:
                if symbol in visited:
                    continue
                
                # Начинаем новый кластер
                cluster = [symbol]
                visited.add(symbol)
                
                # Ищем все связанные символы
                to_visit = [symbol]
                while to_visit:
                    current = to_visit.pop(0)
                    
                    for other_symbol in symbols:
                        if other_symbol in visited:
                            continue
                        
                        # Проверяем корреляцию в обе стороны
                        correlation1 = correlation_matrix.get(current, {}).get(other_symbol, 0.0)
                        correlation2 = correlation_matrix.get(other_symbol, {}).get(current, 0.0)
                        max_correlation = max(abs(correlation1), abs(correlation2))
                        
                        if max_correlation >= threshold:
                            cluster.append(other_symbol)
                            visited.add(other_symbol)
                            to_visit.append(other_symbol)
                
                # Добавляем кластер только если он содержит более одного символа
                if len(cluster) > 1:
                    clusters.append(cluster)
            
            return clusters
        except Exception as e:
            logger.error(f"Error detecting correlation clusters: {e}")
            return []

    def calculate_volatility_ratio(self, prices1: List[Any], prices2: List[Any]) -> float:
        """Расчет отношения волатильности."""
        try:
            if not self.validate_data(prices1) or not self.validate_data(prices2):
                return 1.0
            
            # Выравниваем длины
            min_length = min(len(prices1), len(prices2))
            if min_length < 2:
                return 1.0
            
            prices1 = prices1[-min_length:]
            prices2 = prices2[-min_length:]
            
            # Конвертируем в числовые значения
            try:
                p1 = [float(p) for p in prices1]
                p2 = [float(p) for p in prices2]
            except (ValueError, TypeError):
                return 1.0
            
            # Рассчитываем returns
            returns1 = [(p1[i] - p1[i-1]) / p1[i-1] if p1[i-1] != 0 else 0 for i in range(1, len(p1))]
            returns2 = [(p2[i] - p2[i-1]) / p2[i-1] if p2[i-1] != 0 else 0 for i in range(1, len(p2))]
            
            # Рассчитываем стандартные отклонения (волатильность)
            if len(returns1) < 2 or len(returns2) < 2:
                return 1.0
            
            mean1 = sum(returns1) / len(returns1)
            mean2 = sum(returns2) / len(returns2)
            
            variance1 = sum((r - mean1) ** 2 for r in returns1) / (len(returns1) - 1)
            variance2 = sum((r - mean2) ** 2 for r in returns2) / (len(returns2) - 1)
            
            if variance1 == 0 or variance2 == 0:
                return 1.0
            
            volatility1 = variance1 ** 0.5
            volatility2 = variance2 ** 0.5
            
            # Рассчитываем отношение волатильности
            ratio = volatility1 / volatility2 if volatility2 != 0 else 1.0
            
            # Ограничиваем результат разумными пределами
            return max(0.1, min(10.0, ratio))
        except Exception as e:
            logger.error(f"Error calculating volatility ratio: {e}")
            return 1.0

    async def monitor_changes(self, symbol1: str, symbol2: str, timeframe: str, window_size: int) -> Dict[str, Any]:
        """Мониторинг изменений запутанности."""
        try:
            # Получаем текущую оценку запутанности
            current_analysis = await self.analyze_entanglement(symbol1, symbol2, timeframe)
            current_entanglement = current_analysis.get("entanglement_score", 0.0)
            
            # Получаем исторические данные
            history = self.get_entanglement_history(limit=window_size)
            
            # Извлекаем исторические оценки запутанности для данной пары
            historical_scores = []
            for event in history:
                if (event.get("symbol1") == symbol1 and event.get("symbol2") == symbol2) or \
                   (event.get("symbol1") == symbol2 and event.get("symbol2") == symbol1):
                    historical_scores.append(event.get("entanglement_score", 0.0))
            
            # Рассчитываем тренд
            trend = self.calculate_trend(historical_scores)
            
            # Определяем, есть ли значительные изменения
            change_detected = False
            change_magnitude = 0.0
            
            if len(historical_scores) >= 2:
                recent_score = historical_scores[-1] if historical_scores else current_entanglement
                previous_score = historical_scores[-2] if len(historical_scores) >= 2 else recent_score
                change_magnitude = abs(current_entanglement - previous_score)
                
                # Считаем изменение значительным, если оно больше 0.1
                change_detected = change_magnitude > 0.1
            
            # Проверяем на разрыв запутанности
            breakdown_detected = self.detect_breakdown(historical_scores + [current_entanglement])
            
            return {
                "current_entanglement": current_entanglement,
                "entanglement_trend": trend,
                "change_detected": change_detected,
                "change_magnitude": change_magnitude,
                "breakdown_detected": breakdown_detected,
                "historical_scores_count": len(historical_scores),
                "symbol1": symbol1,
                "symbol2": symbol2,
                "timeframe": timeframe,
                "window_size": window_size
            }
        except Exception as e:
            logger.error(f"Error monitoring changes: {e}")
            return {
                "current_entanglement": 0.0,
                "entanglement_trend": "unknown",
                "change_detected": False,
                "change_magnitude": 0.0,
                "error": str(e)
            }

    def detect_breakdown(self, historical_scores: List[float], threshold: float = 0.5) -> bool:
        """Продвинутое обнаружение разрыва запутанности."""
        
        if len(historical_scores) < 3:
            return False
        
        scores_array = np.array(historical_scores)
        
        # 1. Проверка текущего значения ниже порога
        current_below_threshold = scores_array[-1] < threshold
        
        # 2. Статистический анализ изменений
        if len(scores_array) >= 10:
            # Анализ тренда последних периодов
            recent_scores = scores_array[-10:]
            trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            
            # Резкое снижение тренда
            sharp_decline = trend_slope < -0.05
            
            # Анализ волатильности
            volatility = np.std(recent_scores)
            mean_score = np.mean(recent_scores)
            
            # Высокая волатильность с низким средним
            unstable_pattern = volatility > 0.15 and mean_score < threshold
            
            # Комбинированная оценка разрыва
            breakdown_detected = (
                current_below_threshold and 
                (sharp_decline or unstable_pattern)
            )
        else:
            # Для коротких серий используем простой анализ
            recent_decline = (
                len(scores_array) >= 3 and
                scores_array[-1] < scores_array[-2] < scores_array[-3] and
                scores_array[-1] < threshold * 0.8  # Более строгий порог
            )
            breakdown_detected = current_below_threshold or recent_decline
        
        return breakdown_detected

    def calculate_trend(self, historical_scores: List[float]) -> str:
        """Продвинутый расчет тренда запутанности."""
        from scipy import stats
        
        if len(historical_scores) < 3:
            return "insufficient_data"
        
        scores_array = np.array(historical_scores)
        
        try:
            # Статистический анализ тренда с помощью регрессии
            x = np.arange(len(scores_array))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores_array)
            
            # Определение статистической значимости тренда
            significant = p_value < 0.05
            
            # Классификация тренда
            if significant:
                if slope > 0.01:  # Значимый положительный тренд
                    trend_strength = min(abs(slope) * 100, 1.0)  # Нормализация силы тренда
                    if trend_strength > 0.5:
                        return "strongly_increasing"
                    else:
                        return "increasing"
                elif slope < -0.01:  # Значимый отрицательный тренд
                    trend_strength = min(abs(slope) * 100, 1.0)
                    if trend_strength > 0.5:
                        return "strongly_decreasing"
                    else:
                        return "decreasing"
                else:
                    return "stable"
            else:
                # Если тренд статистически не значим, анализируем волатильность
                volatility = np.std(scores_array)
                if volatility > 0.2:
                    return "volatile"
                else:
                    return "stable"
                    
        except Exception:
            # Fallback на простой анализ
            if len(historical_scores) >= 2:
                if historical_scores[-1] > historical_scores[0]:
                    return "increasing"
                elif historical_scores[-1] < historical_scores[0]:
                    return "decreasing"
                else:
                    return "stable"
            return "unknown"

    def validate_data(self, data: Any) -> bool:
        """Продвинутая валидация входных данных."""
        
        try:
            # Проверка на None
            if data is None:
                return False
            
            # Проверка типов данных
            if isinstance(data, (list, tuple, np.ndarray)):
                if len(data) == 0:
                    return False
                
                # Проверка на числовые значения
                try:
                    numeric_data = np.array(data, dtype=float)
                    
                    # Проверка на NaN и бесконечность
                    if np.any(np.isnan(numeric_data)) or np.any(np.isinf(numeric_data)):
                        return False
                    
                    # Проверка на минимальную длину для анализа
                    if len(numeric_data) < 2:
                        return False
                    
                    # Проверка на разумный диапазон значений (для цен)
                    if np.any(numeric_data < 0):  # Отрицательные цены недопустимы
                        return False
                    
                    # Проверка на экстремальные значения
                    if np.any(numeric_data > 1e10):  # Слишком большие значения
                        return False
                    
                    # Проверка на константные данные
                    if np.std(numeric_data) == 0:
                        return False  # Все значения одинаковые
                    
                    return True
                    
                except (ValueError, TypeError):
                    return False
            
            # Для других типов данных
            elif isinstance(data, (int, float)):
                return not (np.isnan(data) or np.isinf(data) or data < 0)
            
            else:
                return False
                
        except Exception:
            return False
        return True

    def calculate_confidence_interval(self, prices1: List[Any], prices2: List[Any], confidence_level: float) -> Dict[str, float]:
        """Расчет доверительного интервала."""
        try:
            if not self.validate_data(prices1) or not self.validate_data(prices2):
                return {"lower_bound": 0.0, "upper_bound": 1.0}
            
            # Выравниваем длины
            min_length = min(len(prices1), len(prices2))
            if min_length < 3:
                return {"lower_bound": 0.0, "upper_bound": 1.0}
            
            prices1 = prices1[-min_length:]
            prices2 = prices2[-min_length:]
            
            # Конвертируем в числовые значения
            try:
                p1 = [float(p) for p in prices1]
                p2 = [float(p) for p in prices2]
            except (ValueError, TypeError):
                return {"lower_bound": 0.0, "upper_bound": 1.0}
            
            # Рассчитываем корреляцию
            correlation = self.calculate_correlation(p1, p2)
            
            # Рассчитываем доверительный интервал для корреляции Пирсона
            # Используем преобразование Фишера
            if abs(correlation) >= 1.0:
                return {"lower_bound": correlation, "upper_bound": correlation}
            
            # Преобразование Фишера
            z = 0.5 * math.log((1 + correlation) / (1 - correlation))
            
            # Стандартная ошибка
            se = 1.0 / math.sqrt(min_length - 3)
            
            # Z-score для доверительного уровня
            if confidence_level == 0.95:
                z_score = 1.96
            elif confidence_level == 0.99:
                z_score = 2.58
            elif confidence_level == 0.90:
                z_score = 1.645
            else:
                # Приближенное значение для других уровней
                z_score = 1.96
            
            # Доверительный интервал для z
            z_lower = z - z_score * se
            z_upper = z + z_score * se
            
            # Обратное преобразование Фишера
            def fisher_inverse(z_val):
                return (math.exp(2 * z_val) - 1) / (math.exp(2 * z_val) + 1)
            
            lower_bound = fisher_inverse(z_lower)
            upper_bound = fisher_inverse(z_upper)
            
            # Ограничиваем результат в диапазоне [-1, 1]
            lower_bound = max(-1.0, min(1.0, lower_bound))
            upper_bound = max(-1.0, min(1.0, upper_bound))
            
            return {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "correlation": correlation,
                "confidence_level": confidence_level,
                "sample_size": min_length
            }
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            return {"lower_bound": 0.0, "upper_bound": 1.0}

    async def _get_historical_prices(self, symbol: str, timeframe: str) -> List[float]:
        """Получение исторических цен для символа."""
        try:
            # Временная реализация - возвращаем тестовые данные
            # В реальной реализации здесь должен быть вызов к market data service
            base_price = 100.0
            prices = []
            for i in range(100):
                # Генерируем реалистичные цены с небольшими колебаниями
                change = random.uniform(-0.02, 0.02)  # ±2% изменение
                base_price *= (1 + change)
                prices.append(base_price)
            return prices
        except Exception as e:
            logger.error(f"Error getting historical prices for {symbol}: {e}")
            return []

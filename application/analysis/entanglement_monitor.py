# -*- coding: utf-8 -*-
"""Entanglement monitoring service for cross-exchange order book analysis."""

import asyncio
import json
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

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
        import random

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
                    shifted1 = returns1[:len(returns2) - shift]
                    shifted2 = returns2[shift:]
                
                if len(shifted1) < 2 or len(shifted2) < 2:
                    continue
                
                # Рассчитываем корреляцию для данного сдвига
                correlation = self.calculate_correlation(shifted1, shifted2)
                
                if abs(correlation) > abs(max_correlation):
                    max_correlation = correlation
                    best_shift = shift
            
            # Нормализуем фазовый сдвиг к [-1, 1]
            return best_shift / max(max_shift, 1) if max_shift > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating phase shift: {e}")
            return 0.0

    def calculate_entanglement_score(self, correlation: float, phase_shift: float, volatility_ratio: float) -> float:
        """Расчет общего score запутанности на основе компонентов."""
        try:
            # Веса для различных компонентов
            correlation_weight = 0.5  # Основной фактор
            phase_weight = 0.3        # Синхронность движений
            volatility_weight = 0.2   # Схожесть волатильности
            
            # Нормализуем компоненты
            correlation_component = abs(correlation) * correlation_weight
            
            # Фазовый сдвиг: меньший сдвиг = выше запутанность
            phase_component = (1.0 - abs(phase_shift)) * phase_weight
            
            # Волатильность: чем ближе к 1, тем выше запутанность
            volatility_component = (1.0 - abs(volatility_ratio - 1.0)) * volatility_weight
            
            # Общий score
            entanglement_score = correlation_component + phase_component + volatility_component
            
            # Бонус за высокую корреляцию и низкий фазовый сдвиг
            if abs(correlation) > 0.8 and abs(phase_shift) < 0.2:
                entanglement_score *= 1.2  # 20% бонус
            
            # Ограничиваем результат в диапазоне [0, 1]
            return max(0.0, min(1.0, entanglement_score))
            
        except Exception as e:
            logger.error(f"Error calculating entanglement score: {e}")
            return 0.0

    def detect_correlation_clusters(self, correlation_matrix: Dict[str, Dict[str, float]], threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Обнаружение кластеров сильно коррелированных символов."""
        try:
            if not correlation_matrix:
                return []
            
            symbols = list(correlation_matrix.keys())
            visited = set()
            clusters = []
            
            def dfs(symbol: str, current_cluster: List[str], threshold: float) -> None:
                """Поиск в глубину для формирования кластера."""
                if symbol in visited:
                    return
                
                visited.add(symbol)
                current_cluster.append(symbol)
                
                # Ищем связанные символы
                for other_symbol in symbols:
                    if (other_symbol not in visited and 
                        other_symbol in correlation_matrix.get(symbol, {}) and
                        abs(correlation_matrix[symbol][other_symbol]) >= threshold):
                        dfs(other_symbol, current_cluster, threshold)
            
            # Формируем кластеры
            for symbol in symbols:
                if symbol not in visited:
                    cluster = []
                    dfs(symbol, cluster, threshold)
                    
                    if len(cluster) > 1:  # Кластер должен содержать минимум 2 символа
                        # Рассчитываем статистики кластера
                        correlations_in_cluster = []
                        for i, sym1 in enumerate(cluster):
                            for j, sym2 in enumerate(cluster):
                                if i < j:  # Избегаем дублирования
                                    corr = correlation_matrix.get(sym1, {}).get(sym2, 0.0)
                                    correlations_in_cluster.append(abs(corr))
                        
                        avg_correlation = sum(correlations_in_cluster) / len(correlations_in_cluster) if correlations_in_cluster else 0.0
                        min_correlation = min(correlations_in_cluster) if correlations_in_cluster else 0.0
                        max_correlation = max(correlations_in_cluster) if correlations_in_cluster else 0.0
                        
                        clusters.append({
                            "symbols": cluster,
                            "size": len(cluster),
                            "avg_correlation": avg_correlation,
                            "min_correlation": min_correlation,
                            "max_correlation": max_correlation,
                            "strength": "strong" if avg_correlation >= 0.8 else "moderate"
                        })
            
            # Сортируем кластеры по силе корреляции
            clusters.sort(key=lambda x: x["avg_correlation"], reverse=True)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error detecting correlation clusters: {e}")
            return []

    def calculate_volatility_ratio(self, prices1: List[Any], prices2: List[Any]) -> float:
        """Расчет отношения волатильности между двумя рядами цен."""
        try:
            if not self.validate_data(prices1) or not self.validate_data(prices2):
                return 1.0  # Нейтральное отношение
            
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
            
            # Рассчитываем returns (изменения цен)
            returns1 = [(p1[i] - p1[i-1]) / p1[i-1] if p1[i-1] != 0 else 0 for i in range(1, len(p1))]
            returns2 = [(p2[i] - p2[i-1]) / p2[i-1] if p2[i-1] != 0 else 0 for i in range(1, len(p2))]
            
            if not returns1 or not returns2:
                return 1.0
            
            # Рассчитываем стандартные отклонения (волатильность)
            mean1 = sum(returns1) / len(returns1)
            mean2 = sum(returns2) / len(returns2)
            
            variance1 = sum((r - mean1) ** 2 for r in returns1) / len(returns1)
            variance2 = sum((r - mean2) ** 2 for r in returns2) / len(returns2)
            
            volatility1 = variance1 ** 0.5
            volatility2 = variance2 ** 0.5
            
            # Избегаем деления на ноль
            if volatility2 == 0:
                return 1.0 if volatility1 == 0 else float('inf')
            
            ratio = volatility1 / volatility2
            
            # Ограничиваем экстремальные значения
            return max(0.1, min(10.0, ratio))
            
        except Exception as e:
            logger.error(f"Error calculating volatility ratio: {e}")
            return 1.0

    async def monitor_changes(self, symbols: List[str], timeframe: str, callback: Optional[callable] = None) -> Dict[str, Any]:
        """Мониторинг изменений в корреляциях в реальном времени."""
        try:
            if not symbols or len(symbols) < 2:
                return {"error": "Need at least 2 symbols to monitor"}
            
            monitoring_state = {
                "symbols": symbols,
                "timeframe": timeframe,
                "start_time": datetime.now(),
                "observations": [],
                "changes_detected": [],
                "is_active": True
            }
            
            previous_correlations = {}
            change_threshold = 0.1  # Порог для обнаружения значимых изменений
            
            # Начальное измерение
            initial_analysis = await self.analyze_correlations(symbols, timeframe)
            if "correlation_matrix" in initial_analysis:
                previous_correlations = initial_analysis["correlation_matrix"]
            
            observation_count = 0
            max_observations = 10  # Ограничиваем количество наблюдений
            
            while monitoring_state["is_active"] and observation_count < max_observations:
                await asyncio.sleep(30)  # Интервал мониторинга 30 секунд
                
                # Новое измерение
                current_analysis = await self.analyze_correlations(symbols, timeframe)
                
                if "correlation_matrix" not in current_analysis:
                    continue
                
                current_correlations = current_analysis["correlation_matrix"]
                changes = []
                
                # Обнаруживаем изменения
                for symbol1 in symbols:
                    for symbol2 in symbols:
                        if symbol1 >= symbol2:  # Избегаем дублирования
                            continue
                        
                        prev_corr = previous_correlations.get(symbol1, {}).get(symbol2, 0.0)
                        curr_corr = current_correlations.get(symbol1, {}).get(symbol2, 0.0)
                        
                        change = abs(curr_corr - prev_corr)
                        
                        if change >= change_threshold:
                            change_info = {
                                "symbol1": symbol1,
                                "symbol2": symbol2,
                                "previous_correlation": prev_corr,
                                "current_correlation": curr_corr,
                                "change": curr_corr - prev_corr,
                                "timestamp": datetime.now().isoformat()
                            }
                            changes.append(change_info)
                            monitoring_state["changes_detected"].append(change_info)
                
                # Сохраняем наблюдение
                observation = {
                    "timestamp": datetime.now().isoformat(),
                    "correlation_matrix": current_correlations,
                    "changes": changes,
                    "observation_id": observation_count
                }
                monitoring_state["observations"].append(observation)
                
                # Вызываем callback, если предоставлен
                if callback and changes:
                    try:
                        await callback(changes, current_analysis)
                    except Exception as e:
                        logger.warning(f"Callback error in monitoring: {e}")
                
                previous_correlations = current_correlations
                observation_count += 1
            
            monitoring_state["is_active"] = False
            monitoring_state["end_time"] = datetime.now()
            
            return {
                "monitoring_state": monitoring_state,
                "total_observations": observation_count,
                "total_changes": len(monitoring_state["changes_detected"]),
                "significant_changes": [
                    change for change in monitoring_state["changes_detected"] 
                    if abs(change["change"]) >= 0.2
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in monitor_changes: {e}")
            return {"error": str(e)}

    async def detect_breakdown(self, symbol1: str, symbol2: str, timeframe: str, threshold: float = 0.3) -> Dict[str, Any]:
        """Обнаружение распада корреляции между двумя символами."""
        try:
            # Получаем исторические данные для анализа тренда
            current_analysis = await self.analyze_entanglement(symbol1, symbol2, timeframe)
            
            if "error" in current_analysis:
                return {
                    "breakdown_detected": False,
                    "error": current_analysis["error"]
                }
            
            current_correlation = current_analysis.get("correlation", 0.0)
            current_entanglement = current_analysis.get("entanglement_score", 0.0)
            
            # Получаем исторические данные корреляции из кэша/логов
            history = await self.get_entanglement_history(symbol1, symbol2, limit=20)
            
            if len(history) < 3:
                return {
                    "breakdown_detected": False,
                    "current_correlation": current_correlation,
                    "current_entanglement": current_entanglement,
                    "message": "Insufficient historical data"
                }
            
            # Анализируем тренд
            recent_scores = [entry.get("entanglement_score", 0.0) for entry in history[-5:]]
            earlier_scores = [entry.get("entanglement_score", 0.0) for entry in history[:5]]
            
            recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
            earlier_avg = sum(earlier_scores) / len(earlier_scores) if earlier_scores else 0.0
            
            # Определяем распад
            decline = earlier_avg - recent_avg
            breakdown_detected = (
                decline >= threshold and 
                current_entanglement < 0.4 and
                abs(current_correlation) < 0.3
            )
            
            # Рассчитываем дополнительные метрики
            volatility_in_correlation = 0.0
            if len(recent_scores) > 1:
                mean_recent = sum(recent_scores) / len(recent_scores)
                volatility_in_correlation = sum((score - mean_recent) ** 2 for score in recent_scores) / len(recent_scores)
                volatility_in_correlation = volatility_in_correlation ** 0.5
            
            return {
                "breakdown_detected": breakdown_detected,
                "current_correlation": current_correlation,
                "current_entanglement": current_entanglement,
                "decline": decline,
                "recent_average": recent_avg,
                "earlier_average": earlier_avg,
                "volatility": volatility_in_correlation,
                "confidence": min(1.0, decline / threshold) if breakdown_detected else 0.0,
                "threshold": threshold,
                "history_points": len(history)
            }
            
        except Exception as e:
            logger.error(f"Error detecting breakdown: {e}")
            return {
                "breakdown_detected": False,
                "error": str(e)
            }

    def calculate_trend(self, data: List[Any], window: int = 5) -> Dict[str, Any]:
        """Расчет тренда в данных с использованием скользящего окна."""
        try:
            if not data or len(data) < 2:
                return {
                    "trend": "insufficient_data",
                    "slope": 0.0,
                    "strength": 0.0,
                    "direction": "none"
                }
            
            # Конвертируем в числовые значения
            try:
                numeric_data = [float(d) for d in data]
            except (ValueError, TypeError):
                return {
                    "trend": "invalid_data",
                    "slope": 0.0,
                    "strength": 0.0,
                    "direction": "none"
                }
            
            if len(numeric_data) < window:
                window = len(numeric_data)
            
            # Рассчитываем скользящие средние
            moving_averages = []
            for i in range(window - 1, len(numeric_data)):
                window_data = numeric_data[i - window + 1:i + 1]
                avg = sum(window_data) / len(window_data)
                moving_averages.append(avg)
            
            if len(moving_averages) < 2:
                return {
                    "trend": "insufficient_smoothed_data",
                    "slope": 0.0,
                    "strength": 0.0,
                    "direction": "none"
                }
            
            # Линейная регрессия на скользящих средних
            n = len(moving_averages)
            x_values = list(range(n))
            y_values = moving_averages
            
            # Рассчитываем коэффициенты линейной регрессии
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x_values[i] * y_values[i] for i in range(n))
            sum_x_squared = sum(x * x for x in x_values)
            
            # Избегаем деления на ноль
            denominator = n * sum_x_squared - sum_x ** 2
            if denominator == 0:
                slope = 0.0
            else:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
            
            # Рассчитываем силу тренда (R²)
            if n > 1:
                y_mean = sum_y / n
                ss_tot = sum((y - y_mean) ** 2 for y in y_values)
                ss_res = sum((y_values[i] - (slope * x_values[i] + (sum_y - slope * sum_x) / n)) ** 2 for i in range(n))
                
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                strength = max(0.0, min(1.0, r_squared))
            else:
                strength = 0.0
            
            # Определяем направление тренда
            if abs(slope) < 1e-6:
                direction = "sideways"
                trend_type = "stable"
            elif slope > 0:
                direction = "upward"
                trend_type = "bullish" if strength > 0.5 else "weak_bullish"
            else:
                direction = "downward"
                trend_type = "bearish" if strength > 0.5 else "weak_bearish"
            
            return {
                "trend": trend_type,
                "slope": slope,
                "strength": strength,
                "direction": direction,
                "moving_averages": moving_averages,
                "data_points": len(numeric_data),
                "window_size": window
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return {
                "trend": "error",
                "slope": 0.0,
                "strength": 0.0,
                "direction": "none",
                "error": str(e)
            }

    def validate_data(self, data: List[Any]) -> bool:
        """Валидация данных для анализа."""
        try:
            if not data or len(data) < 2:
                return False
            
            # Проверяем, что данные можно конвертировать в числа
            numeric_count = 0
            for item in data:
                try:
                    float(item)
                    numeric_count += 1
                except (ValueError, TypeError):
                    continue
            
            # Требуем минимум 80% валидных данных
            valid_ratio = numeric_count / len(data)
            return valid_ratio >= 0.8 and numeric_count >= 2
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False

    def calculate_confidence_interval(self, prices1: List[Any], prices2: List[Any], confidence_level: float = 0.95) -> Dict[str, float]:
        """Расчет доверительного интервала для корреляции."""
        try:
            if not self.validate_data(prices1) or not self.validate_data(prices2):
                return {
                    "lower_bound": 0.0,
                    "upper_bound": 0.0,
                    "confidence_level": confidence_level,
                    "valid": False
                }
            
            # Выравниваем длины
            min_length = min(len(prices1), len(prices2))
            if min_length < 3:
                return {
                    "lower_bound": 0.0,
                    "upper_bound": 0.0,
                    "confidence_level": confidence_level,
                    "valid": False
                }
            
            # Рассчитываем корреляцию
            correlation = self.calculate_correlation(prices1, prices2)
            
            # Фишеровское z-преобразование
            if abs(correlation) >= 0.9999:
                # Обрабатываем крайние случаи
                fisher_z = 5.0 if correlation > 0 else -5.0
            else:
                fisher_z = 0.5 * math.log((1 + correlation) / (1 - correlation))
            
            # Стандартная ошибка для z-преобразованной корреляции
            standard_error = 1.0 / math.sqrt(min_length - 3) if min_length > 3 else 1.0
            
            # Z-значение для заданного уровня доверия
            # Приблизительные значения для стандартного нормального распределения
            z_values = {
                0.90: 1.645,
                0.95: 1.96,
                0.99: 2.576
            }
            z_critical = z_values.get(confidence_level, 1.96)
            
            # Доверительный интервал для z-преобразованной корреляции
            z_lower = fisher_z - z_critical * standard_error
            z_upper = fisher_z + z_critical * standard_error
            
            # Обратное преобразование в корреляцию
            def inverse_fisher_z(z):
                try:
                    exp_2z = math.exp(2 * z)
                    return (exp_2z - 1) / (exp_2z + 1)
                except OverflowError:
                    return 1.0 if z > 0 else -1.0
            
            lower_bound = inverse_fisher_z(z_lower)
            upper_bound = inverse_fisher_z(z_upper)
            
            # Ограничиваем значения в допустимом диапазоне
            lower_bound = max(-1.0, min(1.0, lower_bound))
            upper_bound = max(-1.0, min(1.0, upper_bound))
            
            return {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "correlation": correlation,
                "confidence_level": confidence_level,
                "standard_error": standard_error,
                "sample_size": min_length,
                "valid": True
            }
            
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            return {
                "lower_bound": 0.0,
                "upper_bound": 0.0,
                "confidence_level": confidence_level,
                "valid": False,
                "error": str(e)
            }

    async def _get_historical_prices(self, symbol: str, timeframe: str) -> List[float]:
        """Получение исторических цен для символа."""
        try:
            # Временная реализация - возвращаем тестовые данные
            # В реальной реализации здесь должен быть вызов к market data service
            import random
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

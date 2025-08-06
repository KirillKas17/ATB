"""
Основной модуль интеграции запутанности.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from .types import (
    EntanglementAnalysis,
    EntanglementConfig,
    EntanglementEvent,
    EntanglementLevel,
    EntanglementType,
)

logger = logger


class EntanglementIntegration:
    """
    Интеграция запутанности для обнаружения корреляций между активами.
    """

    def __init__(self, config: Optional[EntanglementConfig] = None) -> None:
        """
        Инициализация интеграции запутанности.
        :param config: конфигурация запутанности
        """
        self.config = config or EntanglementConfig()

        # Хранилище событий и анализов
        self.entanglement_events: Dict[str, EntanglementEvent] = {}
        self.entanglement_analyses: Dict[str, EntanglementAnalysis] = {}
        self.active_events: Dict[str, EntanglementEvent] = {}

        # Статистика
        self.stats: Dict[str, int] = {
            "total_events": 0,
            "total_analyses": 0,
            "high_correlation_events": 0,
            "alerts": 0,
        }

        # Асинхронные задачи
        self.detection_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False

        logger.info("EntanglementIntegration initialized")

    async def start(self) -> None:
        """Запуск интеграции запутанности."""
        try:
            if self.is_running:
                return

            self.is_running = True

            # Запуск обнаружения запутанности
            self.detection_task = asyncio.create_task(
                self._entanglement_detection_loop()
            )

            # Запуск очистки
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

            logger.info("EntanglementIntegration started")

        except Exception as e:
            logger.error(f"Error starting EntanglementIntegration: {e}")
            self.is_running = False

    async def stop(self) -> None:
        """Остановка интеграции запутанности."""
        try:
            if not self.is_running:
                return

            self.is_running = False

            # Отмена задач
            if self.detection_task:
                self.detection_task.cancel()
                try:
                    await self.detection_task
                except asyncio.CancelledError:
                    pass

            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass

            logger.info("EntanglementIntegration stopped")

        except Exception as e:
            logger.error(f"Error stopping EntanglementIntegration: {e}")

    async def detect_entanglement(
        self, symbols: List[str], data: Dict[str, Any]
    ) -> Optional[EntanglementEvent]:
        """Обнаружение запутанности между символами."""
        try:
            if not self.config.enable_entanglement_detection:
                return None

            if len(symbols) < 2:
                return None

            # Анализируем различные типы запутанности
            price_entanglement = await self._analyze_price_entanglement(symbols, data)
            volume_entanglement = await self._analyze_volume_entanglement(symbols, data)
            volatility_entanglement = await self._analyze_volatility_entanglement(
                symbols, data
            )

            # Выбираем максимальную запутанность
            entanglements = [
                price_entanglement,
                volume_entanglement,
                volatility_entanglement,
            ]

            max_entanglement = max(
                entanglements, key=lambda x: x.correlation_score if x else 0
            )

            if (
                max_entanglement
                and max_entanglement.correlation_score
                >= self.config.correlation_threshold
            ):
                # Сохраняем событие
                self.entanglement_events[max_entanglement.event_id] = max_entanglement
                self.active_events[max_entanglement.event_id] = max_entanglement

                self.stats["total_events"] += 1

                if max_entanglement.correlation_score >= self.config.alert_threshold:
                    self.stats["alerts"] += 1
                    logger.warning(
                        f"High entanglement detected: {max_entanglement.symbols} "
                        f"({max_entanglement.correlation_score:.3f})"
                    )

                return max_entanglement

            return None

        except Exception as e:
            logger.error(f"Error detecting entanglement: {e}")
            return None

    async def get_entanglement_events(
        self, symbols: Optional[List[str]] = None, hours: int = 24
    ) -> List[EntanglementEvent]:
        """Получение событий запутанности."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            events = []
            for event in self.entanglement_events.values():
                if event.timestamp < cutoff_time:
                    continue

                if symbols and not any(symbol in event.symbols for symbol in symbols):
                    continue

                events.append(event)

            # Сортируем по времени
            events.sort(key=lambda x: x.timestamp, reverse=True)

            return events

        except Exception as e:
            logger.error(f"Error getting entanglement events: {e}")
            return []

    async def get_active_entanglements(
        self, symbols: Optional[List[str]] = None
    ) -> List[EntanglementEvent]:
        """Получение активных запутанностей."""
        try:
            active_events = []

            for event in self.active_events.values():
                # Проверяем, не истекло ли событие
                if (datetime.now() - event.timestamp).total_seconds() > 300:  # 5 минут
                    del self.active_events[event.event_id]
                    continue

                if symbols and not any(symbol in event.symbols for symbol in symbols):
                    continue

                active_events.append(event)

            return active_events

        except Exception as e:
            logger.error(f"Error getting active entanglements: {e}")
            return []

    async def analyze_entanglement_patterns(
        self, symbols: List[str]
    ) -> Optional[EntanglementAnalysis]:
        """Анализ паттернов запутанности."""
        try:
            if len(symbols) < 2:
                return None

            # Получаем исторические данные запутанности
            events = await self.get_entanglement_events(symbols, hours=24)

            if not events:
                return None

            # Вычисляем корреляционную матрицу
            correlation_matrix = self._calculate_correlation_matrix(symbols, events)

            # Вычисляем общую корреляцию
            overall_correlation = self._calculate_overall_correlation(
                correlation_matrix
            )

            # Определяем уровень запутанности
            level = self._determine_entanglement_level(overall_correlation)

            # Генерируем рекомендации
            recommendations = self._generate_recommendations(
                symbols, overall_correlation, level
            )

            analysis = EntanglementAnalysis(
                analysis_id=f"analysis_{int(time.time())}",
                symbols=symbols,
                entanglement_type=EntanglementType.CORRELATION,
                correlation_matrix=correlation_matrix,
                overall_correlation=overall_correlation,
                level=level,
                confidence=0.8,
                timestamp=datetime.now(),
                recommendations=recommendations,
            )

            self.entanglement_analyses[analysis.analysis_id] = analysis
            self.stats["total_analyses"] += 1

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing entanglement patterns: {e}")
            return None

    async def _analyze_price_entanglement(
        self, symbols: List[str], data: Dict[str, Any]
    ) -> Optional[EntanglementEvent]:
        """Анализ ценовой запутанности."""
        try:
            # Извлекаем ценовые данные
            prices = {}
            for symbol in symbols:
                if symbol in data and "price" in data[symbol]:
                    prices[symbol] = data[symbol]["price"]

            if len(prices) < 2:
                return None

            # Вычисляем корреляцию цен
            correlation = self._calculate_price_correlation(prices)

            if correlation >= self.config.correlation_threshold:
                level = self._determine_entanglement_level(correlation)

                return EntanglementEvent(
                    event_id=f"price_{int(time.time())}",
                    entanglement_type=EntanglementType.PRICE,
                    symbols=symbols,
                    correlation_score=correlation,
                    level=level,
                    timestamp=datetime.now(),
                    confidence=0.8,
                )

            return None

        except Exception as e:
            logger.error(f"Error analyzing price entanglement: {e}")
            return None

    async def _analyze_volume_entanglement(
        self, symbols: List[str], data: Dict[str, Any]
    ) -> Optional[EntanglementEvent]:
        """Анализ объемной запутанности."""
        try:
            # Извлекаем данные объемов
            volumes = {}
            for symbol in symbols:
                if symbol in data and "volume" in data[symbol]:
                    volumes[symbol] = data[symbol]["volume"]

            if len(volumes) < 2:
                return None

            # Вычисляем корреляцию объемов
            correlation = self._calculate_volume_correlation(volumes)

            if correlation >= self.config.correlation_threshold:
                level = self._determine_entanglement_level(correlation)

                return EntanglementEvent(
                    event_id=f"volume_{int(time.time())}",
                    entanglement_type=EntanglementType.VOLUME,
                    symbols=symbols,
                    correlation_score=correlation,
                    level=level,
                    timestamp=datetime.now(),
                    confidence=0.7,
                )

            return None

        except Exception as e:
            logger.error(f"Error analyzing volume entanglement: {e}")
            return None

    async def _analyze_volatility_entanglement(
        self, symbols: List[str], data: Dict[str, Any]
    ) -> Optional[EntanglementEvent]:
        """Анализ волатильной запутанности."""
        try:
            # Извлекаем данные волатильности
            volatilities = {}
            for symbol in symbols:
                if symbol in data and "volatility" in data[symbol]:
                    volatilities[symbol] = data[symbol]["volatility"]

            if len(volatilities) < 2:
                return None

            # Вычисляем корреляцию волатильности
            correlation = self._calculate_volatility_correlation(volatilities)

            if correlation >= self.config.correlation_threshold:
                level = self._determine_entanglement_level(correlation)

                return EntanglementEvent(
                    event_id=f"volatility_{int(time.time())}",
                    entanglement_type=EntanglementType.VOLATILITY,
                    symbols=symbols,
                    correlation_score=correlation,
                    level=level,
                    timestamp=datetime.now(),
                    confidence=0.7,
                )

            return None

        except Exception as e:
            logger.error(f"Error analyzing volatility entanglement: {e}")
            return None

    def _calculate_price_correlation(self, prices: Dict[str, float]) -> float:
        """Вычисление корреляции цен."""
        try:
            # Упрощенная реализация корреляции
            # В реальном проекте здесь был бы более сложный алгоритм
            price_values = list(prices.values())

            if len(price_values) < 2:
                return 0.0

            # Простая корреляция на основе относительных изменений
            changes = []
            for i in range(1, len(price_values)):
                change = (price_values[i] - price_values[i - 1]) / price_values[i - 1]
                changes.append(change)

            if len(changes) < 2:
                return 0.0

            # Вычисляем корреляцию изменений
            mean_change = sum(changes) / len(changes)
            variance = sum((c - mean_change) ** 2 for c in changes) / len(changes)

            if variance == 0:
                return 0.0

            # Упрощенная корреляция
            correlation = min(1.0, abs(mean_change) * 10)  # Масштабируем

            return correlation

        except Exception as e:
            logger.error(f"Error calculating price correlation: {e}")
            return 0.0

    def _calculate_volume_correlation(self, volumes: Dict[str, float]) -> float:
        """Вычисление корреляции объемов."""
        try:
            # Аналогично ценовой корреляции
            volume_values = list(volumes.values())

            if len(volume_values) < 2:
                return 0.0

            changes = []
            for i in range(1, len(volume_values)):
                change = (volume_values[i] - volume_values[i - 1]) / max(
                    volume_values[i - 1], 1
                )
                changes.append(change)

            if len(changes) < 2:
                return 0.0

            mean_change = sum(changes) / len(changes)
            correlation = min(1.0, abs(mean_change) * 5)

            return correlation

        except Exception as e:
            logger.error(f"Error calculating volume correlation: {e}")
            return 0.0

    def _calculate_volatility_correlation(
        self, volatilities: Dict[str, float]
    ) -> float:
        """Вычисление корреляции волатильности."""
        try:
            # Аналогично другим корреляциям
            vol_values = list(volatilities.values())

            if len(vol_values) < 2:
                return 0.0

            changes = []
            for i in range(1, len(vol_values)):
                change = (vol_values[i] - vol_values[i - 1]) / max(
                    vol_values[i - 1], 0.001
                )
                changes.append(change)

            if len(changes) < 2:
                return 0.0

            mean_change = sum(changes) / len(changes)
            correlation = min(1.0, abs(mean_change) * 3)

            return correlation

        except Exception as e:
            logger.error(f"Error calculating volatility correlation: {e}")
            return 0.0

    def _determine_entanglement_level(self, correlation: float) -> EntanglementLevel:
        """Определение уровня запутанности."""
        try:
            if correlation >= 0.95:
                return EntanglementLevel.CRITICAL
            elif correlation >= 0.85:
                return EntanglementLevel.HIGH
            elif correlation >= 0.75:
                return EntanglementLevel.MEDIUM
            else:
                return EntanglementLevel.LOW

        except Exception as e:
            logger.error(f"Error determining entanglement level: {e}")
            return EntanglementLevel.LOW

    def _calculate_correlation_matrix(
        self, symbols: List[str], events: List[EntanglementEvent]
    ) -> Dict[str, Dict[str, float]]:
        """Вычисление корреляционной матрицы."""
        try:
            matrix: Dict[str, Dict[str, float]] = {}

            for symbol1 in symbols:
                matrix[symbol1] = {}
                for symbol2 in symbols:
                    if symbol1 == symbol2:
                        matrix[symbol1][symbol2] = 1.0
                    else:
                        # Находим события с этими символами
                        symbol_events = [
                            event
                            for event in events
                            if symbol1 in event.symbols and symbol2 in event.symbols
                        ]

                        if symbol_events:
                            avg_correlation = sum(
                                event.correlation_score for event in symbol_events
                            ) / len(symbol_events)
                            matrix[symbol1][symbol2] = avg_correlation
                        else:
                            matrix[symbol1][symbol2] = 0.0

            return matrix

        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return {}

    def _calculate_overall_correlation(
        self, correlation_matrix: Dict[str, Dict[str, float]]
    ) -> float:
        """Вычисление общей корреляции."""
        try:
            if not correlation_matrix:
                return 0.0

            total_correlation = 0.0
            count = 0

            for symbol1, correlations in correlation_matrix.items():
                for symbol2, correlation in correlations.items():
                    if symbol1 != symbol2:
                        total_correlation += correlation
                        count += 1

            if count == 0:
                return 0.0

            return total_correlation / count

        except Exception as e:
            logger.error(f"Error calculating overall correlation: {e}")
            return 0.0

    def _generate_recommendations(
        self, symbols: List[str], correlation: float, level: EntanglementLevel
    ) -> List[str]:
        """Генерация рекомендаций."""
        try:
            recommendations = []

            if level == EntanglementLevel.CRITICAL:
                recommendations.append(
                    "Critical entanglement detected - consider reducing position sizes"
                )
                recommendations.append("Monitor for potential market manipulation")
                recommendations.append("Implement additional risk controls")

            elif level == EntanglementLevel.HIGH:
                recommendations.append("High entanglement detected - exercise caution")
                recommendations.append(
                    "Consider diversifying across uncorrelated assets"
                )
                recommendations.append("Monitor correlation changes closely")

            elif level == EntanglementLevel.MEDIUM:
                recommendations.append(
                    "Medium entanglement detected - normal trading conditions"
                )
                recommendations.append("Continue with standard risk management")

            else:
                recommendations.append("Low entanglement - normal market conditions")

            recommendations.append(f"Correlation level: {correlation:.3f}")
            recommendations.append(f"Affected symbols: {', '.join(symbols)}")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]

    async def _entanglement_detection_loop(self) -> None:
        """Цикл обнаружения запутанности."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.detection_interval)

                # Здесь должна быть логика получения данных и обнаружения запутанности
                # Пока просто очищаем старые события
                await self._cleanup_old_events()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in entanglement detection loop: {e}")

    async def _cleanup_loop(self) -> None:
        """Цикл очистки."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Каждые 5 минут
                await self._cleanup_old_events()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_old_events(self) -> None:
        """Очистка старых событий."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)

            # Очищаем старые события
            old_event_ids = [
                event_id
                for event_id, event in self.entanglement_events.items()
                if event.timestamp < cutoff_time
            ]

            for event_id in old_event_ids:
                del self.entanglement_events[event_id]

            # Очищаем старые анализы
            old_analysis_ids = [
                analysis_id
                for analysis_id, analysis in self.entanglement_analyses.items()
                if analysis.timestamp < cutoff_time
            ]

            for analysis_id in old_analysis_ids:
                del self.entanglement_analyses[analysis_id]

            if old_event_ids or old_analysis_ids:
                logger.info(
                    f"Cleaned up {len(old_event_ids)} old events and {len(old_analysis_ids)} old analyses"
                )

        except Exception as e:
            logger.error(f"Error cleaning up old events: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики запутанности."""
        try:
            return {
                "total_events": len(self.entanglement_events),
                "active_events": len(self.active_events),
                "total_analyses": len(self.entanglement_analyses),
                "entanglement_types": self._get_entanglement_type_stats(),
                "stats": self.stats.copy(),
                "config": {
                    "enable_entanglement_detection": self.config.enable_entanglement_detection,
                    "correlation_threshold": self.config.correlation_threshold,
                    "detection_interval": self.config.detection_interval,
                    "alert_threshold": self.config.alert_threshold,
                },
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    def _get_entanglement_type_stats(self) -> Dict[str, int]:
        """Получение статистики по типам запутанности."""
        try:
            stats: Dict[str, int] = {}
            for event in self.entanglement_events.values():
                event_type = event.entanglement_type.value
                stats[event_type] = stats.get(event_type, 0) + 1
            return stats

        except Exception as e:
            logger.error(f"Error getting entanglement type stats: {e}")
            return {}

    def clear_data(self) -> None:
        """Очистка всех данных."""
        try:
            self.entanglement_events.clear()
            self.entanglement_analyses.clear()
            self.active_events.clear()
            self.stats["total_events"] = 0
            self.stats["total_analyses"] = 0
            self.stats["high_correlation_events"] = 0
            self.stats["alerts"] = 0

            logger.info("Entanglement data cleared")

        except Exception as e:
            logger.error(f"Error clearing entanglement data: {e}")

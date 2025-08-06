"""
Основной модуль интеграции памяти китов.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from shared.logging import setup_logger

from .types import (
    WhaleActivity,
    WhaleActivityType,
    WhaleMemory,
    WhaleMemoryConfig,
    WhalePattern,
    WhaleQuery,
    WhaleSize,
)


class WhaleMemoryIntegration:
    """
    Интеграция памяти китов для отслеживания крупных игроков.
    """

    def __init__(self, config: Optional[WhaleMemoryConfig] = None) -> None:
        """
        Инициализация интеграции памяти китов.
        :param config: конфигурация памяти китов
        """
        self.config = config or WhaleMemoryConfig()
        # Хранилище памяти китов
        self.whale_memories: Dict[str, WhaleMemory] = {}
        self.activity_index: Dict[str, List[str]] = {}
        self.pattern_index: Dict[str, List[str]] = {}
        # Статистика
        self.stats = {
            "total_whales": 0,
            "total_activities": 0,
            "total_patterns": 0,
            "queries": 0,
            "predictions": 0,
        }
        # Асинхронные задачи
        self.cleanup_task: Optional[asyncio.Task] = None
        self.prediction_task: Optional[asyncio.Task] = None
        self.is_running = False
        logger.info("WhaleMemoryIntegration initialized")

    async def start(self) -> None:
        """Запуск интеграции памяти китов."""
        try:
            if self.is_running:
                return
            self.is_running = True
            # Запуск очистки памяти
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            # Запуск предсказаний
            if self.config.enable_prediction:
                self.prediction_task = asyncio.create_task(self._prediction_loop())
            logger.info("WhaleMemoryIntegration started")
        except Exception as e:
            logger.error(f"Error starting WhaleMemoryIntegration: {e}")
            self.is_running = False

    async def stop(self) -> None:
        """Остановка интеграции памяти китов."""
        try:
            if not self.is_running:
                return
            self.is_running = False
            # Отмена задач
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            if self.prediction_task:
                self.prediction_task.cancel()
                try:
                    await self.prediction_task
                except asyncio.CancelledError:
                    pass
            logger.info("WhaleMemoryIntegration stopped")
        except Exception as e:
            logger.error(f"Error stopping WhaleMemoryIntegration: {e}")

    async def record_whale_activity(self, activity: WhaleActivity) -> bool:
        """Запись активности кита."""
        try:
            if not self.config.enable_tracking:
                return True
            # Получаем или создаем память кита
            whale_memory = self._get_or_create_whale_memory(
                activity.whale_address, activity.symbol
            )
            # Добавляем активность
            whale_memory.activities.append(activity)
            whale_memory.last_activity = activity.timestamp
            whale_memory.total_volume += activity.amount
            # Обновляем поведенческий скор
            whale_memory.behavior_score = self._calculate_behavior_score(whale_memory)
            # Обновляем уровень риска
            whale_memory.risk_level = self._calculate_risk_level(whale_memory)
            # Индексируем активность
            self._index_activity(activity)
            self.stats["total_activities"] += 1
            if self.config.log_whale_activities:
                logger.info(
                    f"Recorded whale activity: {activity.whale_address[:8]}... "
                    f"{activity.activity_type.value} {activity.amount} {activity.symbol}"
                )
            return True
        except Exception as e:
            logger.error(f"Error recording whale activity: {e}")
            return False

    async def detect_whale_patterns(self, symbol: str) -> List[WhalePattern]:
        """Обнаружение паттернов китов."""
        try:
            patterns: List[WhalePattern] = []
            # Получаем все активности для символа
            activities = await self.get_whale_activities(symbol, hours=24)
            if not activities:
                return patterns
            # Группируем по адресам китов
            whale_activities: Dict[str, List[WhaleActivity]] = {}
            for activity in activities:
                if activity.whale_address not in whale_activities:
                    whale_activities[activity.whale_address] = []
                whale_activities[activity.whale_address].append(activity)
            # Анализируем паттерны для каждого кита
            for whale_address, whale_acts in whale_activities.items():
                whale_patterns = self._analyze_whale_patterns(whale_address, whale_acts)
                patterns.extend(whale_patterns)
            # Анализируем коллективные паттерны
            collective_patterns = self._analyze_collective_patterns(activities)
            patterns.extend(collective_patterns)
            # Сохраняем паттерны
            for pattern in patterns:
                self._save_pattern(pattern)
            return patterns
        except Exception as e:
            logger.error(f"Error detecting whale patterns for {symbol}: {e}")
            return []

    async def get_whale_activities(
        self, symbol: str, hours: int = 24
    ) -> List[WhaleActivity]:
        """Получение активностей китов."""
        try:
            query = WhaleQuery(
                symbol=symbol,
                start_time=datetime.now() - timedelta(hours=hours),
                limit=1000,
            )
            activities = []
            for whale_memory in self.whale_memories.values():
                if whale_memory.symbol == symbol:
                    for activity in whale_memory.activities:
                        if query.start_time and activity.timestamp < query.start_time:
                            continue
                        if query.end_time and activity.timestamp > query.end_time:
                            continue
                        if activity.confidence < query.min_confidence:
                            continue
                        activities.append(activity)
            # Сортируем по времени
            activities.sort(key=lambda x: x.timestamp, reverse=True)
            return activities[: query.limit]
        except Exception as e:
            logger.error(f"Error getting whale activities for {symbol}: {e}")
            return []

    async def get_whale_memory(
        self, whale_address: str, symbol: str
    ) -> Optional[WhaleMemory]:
        """Получение памяти кита."""
        try:
            memory_key = f"{whale_address}_{symbol}"
            return self.whale_memories.get(memory_key)
        except Exception as e:
            logger.error(f"Error getting whale memory for {whale_address}: {e}")
            return None

    async def predict_whale_behavior(self, symbol: str) -> Dict[str, Any]:
        """Предсказание поведения китов."""
        try:
            self.stats["predictions"] += 1
            # Получаем последние активности
            activities = await self.get_whale_activities(symbol, hours=6)
            if not activities:
                return {"prediction": "no_data", "confidence": 0.0}
            # Анализируем тренды
            accumulation_volume = sum(
                a.amount
                for a in activities
                if a.activity_type == WhaleActivityType.ACCUMULATION
            )
            distribution_volume = sum(
                a.amount
                for a in activities
                if a.activity_type == WhaleActivityType.DISTRIBUTION
            )
            # Вычисляем баланс
            total_volume = accumulation_volume + distribution_volume
            if total_volume == 0:
                return {"prediction": "neutral", "confidence": 0.0}
            accumulation_ratio = accumulation_volume / total_volume
            # Определяем предсказание
            if accumulation_ratio > 0.7:
                prediction = "bullish"
                confidence = min(0.9, accumulation_ratio)
            elif accumulation_ratio < 0.3:
                prediction = "bearish"
                confidence = min(0.9, 1.0 - accumulation_ratio)
            else:
                prediction = "neutral"
                confidence = 0.5
            return {
                "prediction": prediction,
                "confidence": confidence,
                "accumulation_volume": accumulation_volume,
                "distribution_volume": distribution_volume,
                "total_volume": total_volume,
                "accumulation_ratio": accumulation_ratio,
            }
        except Exception as e:
            logger.error(f"Error predicting whale behavior for {symbol}: {e}")
            return {"prediction": "error", "confidence": 0.0}

    def _get_or_create_whale_memory(
        self, whale_address: str, symbol: str
    ) -> WhaleMemory:
        """Получение или создание памяти кита."""
        try:
            memory_key = f"{whale_address}_{symbol}"
            if memory_key not in self.whale_memories:
                whale_memory = WhaleMemory(
                    memory_id=memory_key,
                    whale_address=whale_address,
                    symbol=symbol,
                    activities=[],
                    patterns=[],
                    last_activity=datetime.now(),
                )
                self.whale_memories[memory_key] = whale_memory
                self.stats["total_whales"] += 1
            return self.whale_memories[memory_key]
        except Exception as e:
            logger.error(f"Error getting or creating whale memory: {e}")
            raise

    def _calculate_behavior_score(self, whale_memory: WhaleMemory) -> float:
        """Вычисление поведенческого скора кита."""
        try:
            if not whale_memory.activities:
                return 0.0
            # Анализируем последние активности
            recent_activities = [
                a
                for a in whale_memory.activities
                if (datetime.now() - a.timestamp).total_seconds() < 86400  # 24 часа
            ]
            if not recent_activities:
                return 0.0
            # Вычисляем скор на основе типов активности
            activity_scores = {
                WhaleActivityType.ACCUMULATION: 1.0,
                WhaleActivityType.DISTRIBUTION: -1.0,
                WhaleActivityType.MANIPULATION: -2.0,
                WhaleActivityType.PUMP: 0.5,
                WhaleActivityType.DUMP: -0.5,
                WhaleActivityType.WASH_TRADING: -3.0,
                WhaleActivityType.SPOOFING: -3.0,
            }
            total_score = 0.0
            total_weight = 0.0
            for activity in recent_activities:
                score = activity_scores.get(activity.activity_type, 0.0)
                weight = activity.amount * activity.confidence
                total_score += score * weight
                total_weight += weight
            if total_weight == 0:
                return 0.0
            return total_score / total_weight
        except Exception as e:
            logger.error(f"Error calculating behavior score: {e}")
            return 0.0

    def _calculate_risk_level(self, whale_memory: WhaleMemory) -> float:
        """Вычисление уровня риска кита."""
        try:
            if not whale_memory.activities:
                return 0.0
            # Анализируем рискованные активности
            risky_activities = [
                a
                for a in whale_memory.activities
                if a.activity_type
                in [
                    WhaleActivityType.MANIPULATION,
                    WhaleActivityType.WASH_TRADING,
                    WhaleActivityType.SPOOFING,
                ]
            ]
            if not risky_activities:
                return 0.0
            # Вычисляем риск на основе объема и частоты
            total_risky_volume = sum(a.amount for a in risky_activities)
            risk_ratio = total_risky_volume / whale_memory.total_volume
            return min(1.0, risk_ratio * 2.0)  # Увеличиваем риск
        except Exception as e:
            logger.error(f"Error calculating risk level: {e}")
            return 0.0

    def _index_activity(self, activity: WhaleActivity) -> None:
        """Индексирование активности."""
        try:
            # Индекс по символу
            symbol_key = f"symbol_{activity.symbol}"
            if symbol_key not in self.activity_index:
                self.activity_index[symbol_key] = []
            self.activity_index[symbol_key].append(activity.activity_id)
            # Индекс по типу активности
            type_key = f"type_{activity.activity_type.value}"
            if type_key not in self.activity_index:
                self.activity_index[type_key] = []
            self.activity_index[type_key].append(activity.activity_id)
            # Индекс по размеру кита
            size_key = f"size_{activity.size.value}"
            if size_key not in self.activity_index:
                self.activity_index[size_key] = []
            self.activity_index[size_key].append(activity.activity_id)
        except Exception as e:
            logger.error(f"Error indexing activity {activity.activity_id}: {e}")

    def _analyze_whale_patterns(
        self, whale_address: str, activities: List[WhaleActivity]
    ) -> List[WhalePattern]:
        """Анализ паттернов отдельного кита."""
        try:
            patterns: List[WhalePattern] = []
            if len(activities) < 3:
                return patterns
            # Группируем активности по времени
            time_groups: Dict[datetime, List[WhaleActivity]] = {}
            for activity in activities:
                hour_key = activity.timestamp.replace(minute=0, second=0, microsecond=0)
                if hour_key not in time_groups:
                    time_groups[hour_key] = []
                time_groups[hour_key].append(activity)
            # Анализируем паттерны в группах
            for hour, hour_activities in time_groups.items():
                if len(hour_activities) >= 2:
                    pattern = self._create_pattern_from_activities(
                        whale_address, hour_activities
                    )
                    if pattern:
                        patterns.append(pattern)
            return patterns
        except Exception as e:
            logger.error(f"Error analyzing whale patterns for {whale_address}: {e}")
            return []

    def _analyze_collective_patterns(
        self, activities: List[WhaleActivity]
    ) -> List[WhalePattern]:
        """Анализ коллективных паттернов китов."""
        try:
            patterns: List[WhalePattern] = []
            if len(activities) < 5:
                return patterns
            # Группируем по времени
            time_groups: Dict[datetime, List[WhaleActivity]] = {}
            for activity in activities:
                hour_key = activity.timestamp.replace(minute=0, second=0, microsecond=0)
                if hour_key not in time_groups:
                    time_groups[hour_key] = []
                time_groups[hour_key].append(activity)
            # Ищем коллективные паттерны
            for hour, hour_activities in time_groups.items():
                if len(hour_activities) >= 3:
                    # Проверяем на коллективную активность
                    unique_whales = len(set(a.whale_address for a in hour_activities))
                    total_volume = sum(a.amount for a in hour_activities)
                    if (
                        unique_whales >= 2 and total_volume > 1000
                    ):  # Порог для коллективной активности
                        pattern = WhalePattern(
                            pattern_id=f"collective_{int(hour.timestamp())}",
                            symbol=hour_activities[0].symbol,
                            pattern_type="collective_activity",
                            whale_addresses=list(
                                set(a.whale_address for a in hour_activities)
                            ),
                            total_volume=total_volume,
                            start_time=hour,
                            end_time=hour + timedelta(hours=1),
                            confidence=0.7,
                        )
                        patterns.append(pattern)
            return patterns
        except Exception as e:
            logger.error(f"Error analyzing collective patterns: {e}")
            return []

    def _create_pattern_from_activities(
        self, whale_address: str, activities: List[WhaleActivity]
    ) -> Optional[WhalePattern]:
        """Создание паттерна из активностей."""
        try:
            if len(activities) < 2:
                return None
            # Определяем тип паттерна
            activity_types = [a.activity_type for a in activities]
            if all(t == WhaleActivityType.ACCUMULATION for t in activity_types):
                pattern_type = "accumulation_spree"
            elif all(t == WhaleActivityType.DISTRIBUTION for t in activity_types):
                pattern_type = "distribution_spree"
            elif WhaleActivityType.MANIPULATION in activity_types:
                pattern_type = "manipulation_pattern"
            else:
                pattern_type = "mixed_activity"
            total_volume = sum(a.amount for a in activities)
            start_time = min(a.timestamp for a in activities)
            end_time = max(a.timestamp for a in activities)
            pattern = WhalePattern(
                pattern_id=f"{whale_address}_{int(start_time.timestamp())}",
                symbol=activities[0].symbol,
                pattern_type=pattern_type,
                whale_addresses=[whale_address],
                total_volume=total_volume,
                start_time=start_time,
                end_time=end_time,
                confidence=0.8,
            )
            return pattern
        except Exception as e:
            logger.error(f"Error creating pattern from activities: {e}")
            return None

    def _save_pattern(self, pattern: WhalePattern) -> None:
        """Сохранение паттерна."""
        try:
            # Добавляем паттерн в память каждого кита
            for whale_address in pattern.whale_addresses:
                memory_key = f"{whale_address}_{pattern.symbol}"
                if memory_key in self.whale_memories:
                    self.whale_memories[memory_key].patterns.append(pattern)
            # Индексируем паттерн
            pattern_key = f"pattern_{pattern.pattern_type}"
            if pattern_key not in self.pattern_index:
                self.pattern_index[pattern_key] = []
            self.pattern_index[pattern_key].append(pattern.pattern_id)
            self.stats["total_patterns"] += 1
        except Exception as e:
            logger.error(f"Error saving pattern {pattern.pattern_id}: {e}")

    async def _cleanup_loop(self) -> None:
        """Цикл очистки памяти."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_old_memories()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _prediction_loop(self) -> None:
        """Цикл предсказаний."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Каждые 5 минут
                # Получаем все символы
                symbols = list(set(m.symbol for m in self.whale_memories.values()))
                # Делаем предсказания для каждого символа
                for symbol in symbols:
                    try:
                        prediction = await self.predict_whale_behavior(symbol)
                        if prediction["confidence"] > 0.7:
                            logger.info(
                                f"High confidence whale prediction for {symbol}: {prediction}"
                            )
                    except Exception as e:
                        logger.error(f"Error in prediction loop for {symbol}: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")

    async def _cleanup_old_memories(self) -> None:
        """Очистка старых записей памяти."""
        try:
            current_time = datetime.now()
            memories_to_remove = []
            for memory_key, memory in self.whale_memories.items():
                # Проверяем время последней активности
                if (
                    current_time - memory.last_activity
                ).total_seconds() > self.config.memory_ttl:
                    memories_to_remove.append(memory_key)
            # Удаляем старые записи
            for memory_key in memories_to_remove:
                del self.whale_memories[memory_key]
            if memories_to_remove:
                logger.info(f"Cleaned up {len(memories_to_remove)} old whale memories")
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики памяти китов."""
        try:
            return {
                "total_whales": len(self.whale_memories),
                "total_activities": self.stats["total_activities"],
                "total_patterns": self.stats["total_patterns"],
                "symbols": list(set(m.symbol for m in self.whale_memories.values())),
                "activity_types": self._get_activity_type_stats(),
                "stats": self.stats.copy(),
                "config": {
                    "max_whales": self.config.max_whales,
                    "cleanup_interval": self.config.cleanup_interval,
                    "memory_ttl": self.config.memory_ttl,
                },
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    def _get_activity_type_stats(self) -> Dict[str, int]:
        """Получение статистики по типам активности."""
        try:
            stats: Dict[str, int] = {}
            for memory in self.whale_memories.values():
                for activity in memory.activities:
                    activity_type = activity.activity_type.value
                    stats[activity_type] = stats.get(activity_type, 0) + 1
            return stats
        except Exception as e:
            logger.error(f"Error getting activity type stats: {e}")
            return {}

    def clear_memory(self) -> None:
        """Очистка всей памяти китов."""
        try:
            self.whale_memories.clear()
            self.activity_index.clear()
            self.pattern_index.clear()
            self.stats["total_whales"] = 0
            self.stats["total_activities"] = 0
            self.stats["total_patterns"] = 0
            logger.info("Whale memory cleared")
        except Exception as e:
            logger.error(f"Error clearing whale memory: {e}")

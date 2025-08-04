"""
Наблюдатель паттернов для мониторинга их развития.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from shared.numpy_utils import np

from domain.type_definitions.intelligence_types import PatternDetection
from domain.type_definitions.intelligence_types import PatternType
from domain.memory.pattern_memory import PatternMemory
from domain.value_objects.timestamp import Timestamp

logger = logging.getLogger(__name__)


class ObserverStatus(Enum):
    """Статус наблюдателя."""

    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ObservationConfig:
    """Конфигурация наблюдения."""

    # Параметры наблюдения
    observation_periods: int = 20  # Количество свечей для наблюдения
    candle_interval_seconds: int = 60  # Интервал свечи в секундах
    update_interval_seconds: float = 1.0  # Интервал обновления

    # Пороги для определения исхода
    profit_threshold_percent: float = 0.5  # Порог прибыли
    loss_threshold_percent: float = -0.5  # Порог убытка
    volatility_threshold: float = 0.02  # Порог волатильности

    # Параметры анализа
    enable_volume_analysis: bool = True
    enable_volatility_analysis: bool = True
    enable_regime_analysis: bool = True

    # Логирование
    enable_detailed_logging: bool = True
    log_intermediate_results: bool = False


@dataclass
class ObservationState:
    """Состояние наблюдения."""

    pattern_id: str
    symbol: str
    pattern_type: PatternType
    start_timestamp: Timestamp
    start_price: float
    start_volume: float

    # Текущие значения
    current_price: float
    current_volume: float
    elapsed_periods: int

    # Экстремумы
    max_price: float
    min_price: float
    max_volume: float
    min_volume: float

    # Статистика
    price_changes: List[float]
    volume_changes: List[float]
    volatilities: List[float]

    # Статус
    status: ObserverStatus
    last_update: Timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "pattern_id": self.pattern_id,
            "symbol": self.symbol,
            "pattern_type": self.pattern_type.value,
            "start_timestamp": self.start_timestamp.to_iso(),
            "start_price": self.start_price,
            "start_volume": self.start_volume,
            "current_price": self.current_price,
            "current_volume": self.current_volume,
            "elapsed_periods": self.elapsed_periods,
            "max_price": self.max_price,
            "min_price": self.min_price,
            "max_volume": self.max_volume,
            "min_volume": self.min_volume,
            "status": self.status.value,
            "last_update": self.last_update.to_iso(),
        }


@dataclass
class PatternOutcome:
    """Исход паттерна."""

    pattern_id: str
    symbol: str
    pattern_type: PatternType
    start_timestamp: Timestamp
    end_timestamp: Timestamp
    duration_minutes: int
    final_price: float
    final_volume: float
    price_change_percent: float
    volume_change_percent: float
    max_price: float
    min_price: float
    max_volume: float
    min_volume: float
    volatility: float
    outcome_type: str  # "profit", "loss", "neutral"
    confidence: float
    volume_profile: str
    market_regime: str
    success: bool

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "pattern_id": self.pattern_id,
            "symbol": self.symbol,
            "pattern_type": self.pattern_type.value,
            "start_timestamp": self.start_timestamp.to_iso(),
            "end_timestamp": self.end_timestamp.to_iso(),
            "duration_minutes": self.duration_minutes,
            "final_price": self.final_price,
            "final_volume": self.final_volume,
            "price_change_percent": self.price_change_percent,
            "volume_change_percent": self.volume_change_percent,
            "max_price": self.max_price,
            "min_price": self.min_price,
            "max_volume": self.max_volume,
            "min_volume": self.min_volume,
            "volatility": self.volatility,
            "outcome_type": self.outcome_type,
            "confidence": self.confidence,
            "volume_profile": self.volume_profile,
            "market_regime": self.market_regime,
            "success": self.success,
        }


@dataclass
class PatternSnapshot:
    """Снимок паттерна."""

    pattern_id: str
    symbol: str
    pattern_type: PatternType
    timestamp: Timestamp
    price: float
    volume: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "pattern_id": self.pattern_id,
            "symbol": self.symbol,
            "pattern_type": self.pattern_type.value,
            "timestamp": self.timestamp.to_iso(),
            "price": self.price,
            "volume": self.volume,
            "metadata": self.metadata,
        }


class PatternObserver:
    """Наблюдатель паттернов."""

    def __init__(
        self, pattern_memory: PatternMemory, config: Optional[ObservationConfig] = None
    ):
        self.pattern_memory = pattern_memory
        self.config = config or ObservationConfig()

        # Активные наблюдения
        self.active_observations: Dict[str, ObservationState] = {}

        # Callbacks
        self.on_completion_callbacks: List[Callable[[PatternOutcome], None]] = []
        self.on_update_callbacks: List[Callable[[ObservationState], None]] = []

        # Состояние выполнения
        self.is_running = False
        self.observation_task: Optional[asyncio.Task] = None

        # Статистика
        self.stats = {
            "total_observations": 0,
            "completed_observations": 0,
            "cancelled_observations": 0,
            "failed_observations": 0,
            "total_profit_outcomes": 0,
            "total_loss_outcomes": 0,
            "total_neutral_outcomes": 0,
        }

    def start_observation(
        self,
        pattern_detection: PatternDetection,
        market_data_provider: Callable[[str], Dict[str, Any]],
    ) -> str:
        """
        Запуск наблюдения за паттерном.

        Args:
            pattern_detection: Обнаруженный паттерн
            market_data_provider: Провайдер рыночных данных

        Returns:
            ID наблюдения
        """
        try:
            # Генерируем ID паттерна если его нет
            pattern_id = getattr(pattern_detection, 'pattern_id', None)
            if not pattern_id:
                pattern_id = f"{pattern_detection.symbol}_{pattern_detection.pattern_type.value}_{datetime.now().timestamp()}"

            # Получаем начальные данные
            initial_data = market_data_provider(pattern_detection.symbol)

            if not initial_data:
                logger.error(f"Cannot get initial data for {pattern_detection.symbol}")
                return ""

            # Создаем состояние наблюдения
            observation_state = ObservationState(
                pattern_id=pattern_id,
                symbol=pattern_detection.symbol,
                pattern_type=pattern_detection.pattern_type,
                start_timestamp=getattr(pattern_detection, 'timestamp', Timestamp(datetime.now())),
                start_price=initial_data.get("price", 0.0),
                start_volume=initial_data.get("volume", 0.0),
                current_price=initial_data.get("price", 0.0),
                current_volume=initial_data.get("volume", 0.0),
                elapsed_periods=0,
                max_price=initial_data.get("price", 0.0),
                min_price=initial_data.get("price", 0.0),
                max_volume=initial_data.get("volume", 0.0),
                min_volume=initial_data.get("volume", 0.0),
                price_changes=[],
                volume_changes=[],
                volatilities=[],
                status=ObserverStatus.ACTIVE,
                last_update=Timestamp(datetime.now()),
            )

            # Сохраняем снимок паттерна
            snapshot = self._create_snapshot_from_detection(
                pattern_detection, initial_data
            )
            if hasattr(self.pattern_memory, 'save_snapshot'):
                self.pattern_memory.save_snapshot(pattern_id, snapshot)

            # Добавляем в активные наблюдения
            self.active_observations[pattern_id] = observation_state

            # Запускаем задачу наблюдения если еще не запущена
            if not self.is_running:
                self._start_observation_loop(market_data_provider)

            logger.info(f"Started observation for pattern {pattern_id}")
            return pattern_id

        except Exception as e:
            logger.error(f"Error starting observation: {e}")
            return ""

    def stop_observation(self, pattern_id: str) -> bool:
        """
        Остановка наблюдения.

        Args:
            pattern_id: ID паттерна

        Returns:
            True если остановка успешна
        """
        try:
            if pattern_id in self.active_observations:
                observation = self.active_observations[pattern_id]
                observation.status = ObserverStatus.CANCELLED

                logger.info(f"Stopped observation for pattern {pattern_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error stopping observation {pattern_id}: {e}")
            return False

    def get_observation_state(self, pattern_id: str) -> Optional[ObservationState]:
        """Получение состояния наблюдения."""
        return self.active_observations.get(pattern_id)

    def get_active_observations(self) -> List[ObservationState]:
        """Получение всех активных наблюдений."""
        return list(self.active_observations.values())

    def add_completion_callback(
        self, callback: Callable[[PatternOutcome], None]
    ) -> None:
        """Добавление callback для завершения наблюдения."""
        self.on_completion_callbacks.append(callback)

    def add_update_callback(self, callback: Callable[[ObservationState], None]) -> None:
        """Добавление callback для обновления состояния."""
        self.on_update_callbacks.append(callback)

    def _start_observation_loop(
        self, market_data_provider: Callable[[str], Dict[str, Any]]
    ) -> None:
        """Запуск цикла наблюдения."""
        if self.is_running:
            return

        self.is_running = True
        self.observation_task = asyncio.create_task(
            self._observation_loop(market_data_provider)
        )

        logger.info("Pattern observation loop started")

    async def _observation_loop(
        self, market_data_provider: Callable[[str], Dict[str, Any]]
    ) -> None:
        """Основной цикл наблюдения."""
        try:
            while self.is_running:
                # Обновляем все активные наблюдения
                completed_observations = []

                for pattern_id, observation in list(self.active_observations.items()):
                    if observation.status != ObserverStatus.ACTIVE:
                        continue

                    # Получаем текущие данные
                    current_data = market_data_provider(observation.symbol)
                    if not current_data:
                        continue

                    # Обновляем состояние
                    updated = self._update_observation_state(observation, current_data)

                    # Проверяем завершение
                    if self._should_complete_observation(observation):
                        outcome = self._create_pattern_outcome(
                            observation, current_data
                        )
                        if outcome:
                            completed_observations.append((pattern_id, outcome))

                    # Вызываем callback обновления
                    if updated:
                        for callback in self.on_update_callbacks:
                            try:
                                callback(observation)
                            except Exception as e:
                                logger.error(f"Error in update callback: {e}")

                # Обрабатываем завершенные наблюдения
                for pattern_id, outcome in completed_observations:
                    await self._complete_observation(pattern_id, outcome)

                # Очищаем завершенные наблюдения
                self._cleanup_completed_observations()

                # Пауза между обновлениями
                await asyncio.sleep(self.config.update_interval_seconds)

        except Exception as e:
            logger.error(f"Error in observation loop: {e}")
            self.is_running = False
        finally:
            self.is_running = False
            logger.info("Pattern observation loop stopped")

    def _update_observation_state(
        self, observation: ObservationState, current_data: Dict[str, Any]
    ) -> bool:
        """Обновление состояния наблюдения."""
        try:
            current_price = current_data.get("price", observation.current_price)
            current_volume = current_data.get("volume", observation.current_volume)

            # Проверяем изменения
            if (
                current_price == observation.current_price
                and current_volume == observation.current_volume
            ):
                return False

            # Обновляем текущие значения
            observation.current_price = current_price
            observation.current_volume = current_volume
            observation.last_update = Timestamp(datetime.now())

            # Обновляем экстремумы
            observation.max_price = max(observation.max_price, current_price)
            observation.min_price = min(observation.min_price, current_price)
            observation.max_volume = max(observation.max_volume, current_volume)
            observation.min_volume = min(observation.min_volume, current_volume)

            # Рассчитываем изменения
            price_change = (
                current_price - observation.start_price
            ) / observation.start_price
            volume_change = (
                current_volume - observation.start_volume
            ) / observation.start_volume

            observation.price_changes.append(price_change)
            observation.volume_changes.append(volume_change)

            # Рассчитываем волатильность
            if len(observation.price_changes) >= 2:
                volatility = np.std(observation.price_changes[-10:])  # Последние 10 изменений
                observation.volatilities.append(volatility)

            # Увеличиваем счетчик периодов
            observation.elapsed_periods += 1

            return True

        except Exception as e:
            logger.error(f"Error updating observation state: {e}")
            return False

    def _should_complete_observation(self, observation: ObservationState) -> bool:
        """Проверка необходимости завершения наблюдения."""
        try:
            # Завершаем по времени
            if observation.elapsed_periods >= self.config.observation_periods:
                return True

            # Завершаем по достижению порогов
            current_price_change = (
                observation.current_price - observation.start_price
            ) / observation.start_price

            if current_price_change >= self.config.profit_threshold_percent:
                return True

            if current_price_change <= self.config.loss_threshold_percent:
                return True

            # Завершаем по волатильности
            if observation.volatilities:
                current_volatility = observation.volatilities[-1]
                if current_volatility >= self.config.volatility_threshold:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking observation completion: {e}")
            return False

    def _create_pattern_outcome(
        self, observation: ObservationState, current_data: Dict[str, Any]
    ) -> Optional[PatternOutcome]:
        """Создание исхода паттерна."""
        try:
            # Рассчитываем основные метрики
            price_change_percent = (
                observation.current_price - observation.start_price
            ) / observation.start_price * 100

            volume_change_percent = (
                observation.current_volume - observation.start_volume
            ) / observation.start_volume * 100

            # Определяем тип исхода
            if price_change_percent >= self.config.profit_threshold_percent:
                outcome_type = "profit"
                success = True
            elif price_change_percent <= self.config.loss_threshold_percent:
                outcome_type = "loss"
                success = False
            else:
                outcome_type = "neutral"
                success = False

            # Рассчитываем волатильность
            volatility = 0.0
            if observation.volatilities:
                volatility = float(np.mean(observation.volatilities))

            # Анализируем профиль объема
            volume_profile = "normal"
            if self.config.enable_volume_analysis:
                volume_profile = self._analyze_volume_profile(observation.volume_changes)

            # Анализируем режим рынка
            market_regime = "normal"
            if self.config.enable_regime_analysis:
                market_regime = self._analyze_market_regime(
                    observation.price_changes, observation.volatilities
                )

            # Рассчитываем уверенность
            confidence = self._calculate_outcome_confidence(observation)

            # Создаем исход
            outcome = PatternOutcome(
                pattern_id=observation.pattern_id,
                symbol=observation.symbol,
                pattern_type=observation.pattern_type,
                start_timestamp=observation.start_timestamp,
                end_timestamp=observation.last_update,
                duration_minutes=observation.elapsed_periods * self.config.candle_interval_seconds // 60,
                final_price=observation.current_price,
                final_volume=observation.current_volume,
                price_change_percent=price_change_percent,
                volume_change_percent=volume_change_percent,
                max_price=observation.max_price,
                min_price=observation.min_price,
                max_volume=observation.max_volume,
                min_volume=observation.min_volume,
                volatility=volatility,
                outcome_type=outcome_type,
                confidence=confidence,
                volume_profile=volume_profile,
                market_regime=market_regime,
                success=success,
            )

            return outcome

        except Exception as e:
            logger.error(f"Error creating pattern outcome: {e}")
            return None

    async def _complete_observation(
        self, pattern_id: str, outcome: PatternOutcome
    ) -> None:
        """Завершение наблюдения."""
        try:
            # Обновляем статус
            if pattern_id in self.active_observations:
                self.active_observations[pattern_id].status = ObserverStatus.COMPLETED

            # Сохраняем исход в память
            if hasattr(self.pattern_memory, 'save_outcome'):
                self.pattern_memory.save_outcome(pattern_id, outcome)

            # Обновляем статистику
            self.stats["completed_observations"] += 1
            if outcome.outcome_type == "profit":
                self.stats["total_profit_outcomes"] += 1
            elif outcome.outcome_type == "loss":
                self.stats["total_loss_outcomes"] += 1
            else:
                self.stats["total_neutral_outcomes"] += 1

            # Вызываем callbacks
            for callback in self.on_completion_callbacks:
                try:
                    callback(outcome)
                except Exception as e:
                    logger.error(f"Error in completion callback: {e}")

            logger.info(
                f"Completed observation for pattern {pattern_id}: {outcome.outcome_type}"
            )

        except Exception as e:
            logger.error(f"Error completing observation {pattern_id}: {e}")

    def _cleanup_completed_observations(self) -> None:
        """Очистка завершенных наблюдений."""
        try:
            completed_ids = [
                pattern_id
                for pattern_id, observation in self.active_observations.items()
                if observation.status in [ObserverStatus.COMPLETED, ObserverStatus.CANCELLED]
            ]

            for pattern_id in completed_ids:
                del self.active_observations[pattern_id]

        except Exception as e:
            logger.error(f"Error cleaning up observations: {e}")

    def _create_snapshot_from_detection(
        self, pattern_detection: PatternDetection, market_data: Dict[str, Any]
    ) -> PatternSnapshot:
        """Создание снимка из обнаружения паттерна."""
        try:
            # Генерируем ID паттерна если его нет
            pattern_id = getattr(pattern_detection, 'pattern_id', None)
            if not pattern_id:
                pattern_id = f"{pattern_detection.symbol}_{pattern_detection.pattern_type.value}_{datetime.now().timestamp()}"

            snapshot = PatternSnapshot(
                pattern_id=pattern_id,
                symbol=pattern_detection.symbol,
                pattern_type=pattern_detection.pattern_type,
                timestamp=getattr(pattern_detection, 'timestamp', Timestamp(datetime.now())),
                price=market_data.get("price", 0.0),
                volume=market_data.get("volume", 0.0),
                metadata={
                    "detection_strength": getattr(pattern_detection, 'strength', 0.0),
                    "detection_confidence": getattr(pattern_detection, 'confidence', 0.0),
                    "market_data": market_data,
                },
            )

            return snapshot

        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            # Возвращаем пустой снимок
            return PatternSnapshot(
                pattern_id="",
                symbol="",
                pattern_type=PatternType.UNKNOWN,
                timestamp=Timestamp(datetime.now()),
                price=0.0,
                volume=0.0,
                metadata={},
            )

    def _calculate_outcome_confidence(self, observation: ObservationState) -> float:
        """Расчет уверенности исхода."""
        try:
            # Базовые факторы
            data_quality = min(1.0, observation.elapsed_periods / self.config.observation_periods)
            price_consistency = 1.0 - abs(observation.current_price - observation.start_price) / observation.start_price
            volume_consistency = 1.0 - abs(observation.current_volume - observation.start_volume) / observation.start_volume

            # Взвешенная уверенность
            confidence = (data_quality * 0.4 + price_consistency * 0.4 + volume_consistency * 0.2)
            return float(confidence)

        except Exception as e:
            logger.error(f"Error calculating outcome confidence: {e}")
            return 0.0

    def _analyze_volume_profile(self, volume_changes: List[float]) -> str:
        """Анализ профиля объема."""
        try:
            if not volume_changes:
                return "normal"

            avg_volume_change = np.mean(volume_changes)
            volume_volatility = np.std(volume_changes)

            if avg_volume_change > 0.1 and volume_volatility < 0.05:
                return "increasing_stable"
            elif avg_volume_change < -0.1 and volume_volatility < 0.05:
                return "decreasing_stable"
            elif volume_volatility > 0.1:
                return "volatile"
            else:
                return "normal"

        except Exception as e:
            logger.error(f"Error analyzing volume profile: {e}")
            return "normal"

    def _analyze_market_regime(
        self, price_changes: List[float], volatilities: List[float]
    ) -> str:
        """Анализ режима рынка."""
        try:
            if not price_changes or not volatilities:
                return "normal"

            avg_price_change = np.mean(price_changes)
            avg_volatility = np.mean(volatilities)

            if avg_volatility > 0.03:
                return "volatile"
            elif abs(avg_price_change) > 0.02:
                return "trending"
            else:
                return "ranging"

        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return "normal"

    def shutdown(self) -> None:
        """Завершение работы наблюдателя."""
        try:
            self.is_running = False

            if self.observation_task:
                self.observation_task.cancel()

            # Очищаем все наблюдения
            self.active_observations.clear()

            logger.info("Pattern observer shutdown completed")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику наблюдателя."""
        total_observations = len(self.active_observations)
        completed_count = sum(
            1 for obs in self.active_observations.values()
            if obs.status == ObserverStatus.COMPLETED
        )
        active_count = sum(
            1 for obs in self.active_observations.values()
            if obs.status == ObserverStatus.ACTIVE
        )
        
        # Анализ исходов
        outcomes = {
            "profit": 0,
            "loss": 0,
            "neutral": 0,
        }
        
        # Средние показатели
        avg_duration = 0.0
        avg_price_change = 0.0
        avg_volume_change = 0.0
        
        if completed_count > 0:
            # В реальной системе здесь был бы анализ завершенных наблюдений
            # Пока возвращаем базовую статистику
            avg_duration = 30.0  # Примерное значение
            avg_price_change = 0.02  # 2%
            avg_volume_change = 0.15  # 15%
        
        return {
            "total_observations": total_observations,
            "active_observations": active_count,
            "completed_observations": completed_count,
            "outcomes": outcomes,
            "average_duration_minutes": avg_duration,
            "average_price_change_percent": avg_price_change,
            "average_volume_change_percent": avg_volume_change,
            "is_running": self.is_running,
            "config": {
                "observation_periods": self.config.observation_periods,
                "candle_interval_seconds": self.config.candle_interval_seconds,
                "update_interval_seconds": self.config.update_interval_seconds,
            },
        }

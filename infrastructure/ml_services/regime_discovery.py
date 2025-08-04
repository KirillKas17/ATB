"""
Модуль обнаружения рыночных режимов.
Обеспечивает автоматическое обнаружение и классификацию
различных рыночных режимов на основе анализа данных.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
from shared.numpy_utils import np
import pandas as pd
from loguru import logger
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from domain.types.messaging_types import Event as MessagingEvent
from infrastructure.messaging.event_bus import EventBus

# Type aliases
DataFrame = pd.DataFrame
Series = pd.Series


@dataclass
class RegimeConfig:
    """Конфигурация обнаружения режимов."""

    # Параметры анализа
    window_size: int = 100
    min_regime_duration: int = 300  # 5 минут
    max_regimes: int = 5
    # Параметры кластеризации
    clustering_method: str = "gmm"  # gmm, kmeans, dbscan
    n_clusters: int = 4
    min_samples: int = 10
    # Пороги
    regime_confidence_threshold: float = 0.7
    regime_change_threshold: float = 0.3
    stability_threshold: float = 0.8
    # Пути
    models_path: str = "models/regime_discovery"
    cache_path: str = "cache/regimes"


@dataclass
class MarketRegime:
    """Рыночный режим."""

    regime_id: str
    name: str
    characteristics: Dict[str, float]
    confidence: float
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    stability: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeTransition:
    """Переход между режимами."""

    from_regime: str
    to_regime: str
    transition_time: datetime
    confidence: float
    trigger_factors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RegimeDiscovery:
    """
    Система обнаружения рыночных режимов.
    Обеспечивает:
    - Автоматическое обнаружение режимов
    - Классификацию режимов
    - Отслеживание переходов
    - Прогнозирование режимов
    """

    def __init__(
        self, event_bus: EventBus, config: Optional[RegimeConfig] = None
    ) -> None:
        """Инициализация системы обнаружения режимов."""
        self.event_bus = event_bus
        self.config = config or RegimeConfig()
        # Создание директорий
        Path(self.config.models_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.cache_path).mkdir(parents=True, exist_ok=True)
        # Модели
        self.clustering_model: Optional[Union[GaussianMixture, KMeans, DBSCAN]] = None
        self.scaler = StandardScaler()
        self.is_scaler_fitted: bool = False
        # Данные
        self.feature_buffer: List[List[float]] = []
        self.regime_buffer: List[Dict[str, Any]] = []
        # Режимы
        self.current_regime: Optional[MarketRegime] = None
        self.regime_history: List[MarketRegime] = []
        self.transition_history: List[RegimeTransition] = []
        # Состояние
        self.is_discovering: bool = False
        self.discovery_task: Optional[asyncio.Task] = None
        # Статистика
        self.regime_statistics: Dict[str, Any] = {}
        # Загрузка моделей
        asyncio.create_task(self.load_models())
        logger.info("RegimeDiscovery initialized")

    async def start_discovery(self) -> None:
        """Запуск обнаружения режимов."""
        if self.is_discovering:
            logger.warning("Regime discovery is already running")
            return
        self.is_discovering = True
        self.discovery_task = asyncio.create_task(self._discovery_loop())
        logger.info("Regime discovery started")

    async def stop_discovery(self) -> None:
        """Остановка обнаружения режимов."""
        if not self.is_discovering:
            return
        self.is_discovering = False
        if self.discovery_task:
            self.discovery_task.cancel()
            try:
                await self.discovery_task
            except asyncio.CancelledError:
                pass
        logger.info("Regime discovery stopped")

    async def _discovery_loop(self) -> None:
        """Основной цикл обнаружения режимов."""
        logger.info("Starting regime discovery loop")
        while self.is_discovering:
            try:
                # Проверка достаточности данных
                if len(self.feature_buffer) >= self.config.min_samples:
                    # Обнаружение режимов
                    await self._detect_regimes()
                    # Анализ переходов
                    await self._analyze_transitions()
                    # Обновление статистики
                    await self._update_statistics()
                    # Публикация событий
                    await self._publish_regime_events()
                # Ожидание следующей итерации
                await asyncio.sleep(60)  # Каждую минуту
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(300)

    async def _detect_regimes(self) -> None:
        """Обнаружение режимов."""
        try:
            # Подготовка данных
            X = await self._prepare_features()
            if X is None or len(X) < self.config.min_samples:
                return
            # Кластеризация
            if self.clustering_model is None:
                await self._initialize_clustering_model(X)
            if self.clustering_model is None:
                return
            # Предсказание режима
            regime_label = self.clustering_model.predict(X[-1:])[0]
            confidence = self._calculate_regime_confidence(X[-1:], regime_label)
            # Проверка изменения режима
            if await self._is_regime_change(regime_label, confidence):
                await self._handle_regime_change(regime_label, confidence)
            else:
                # Обновление текущего режима
                if self.current_regime:
                    self.current_regime.confidence = confidence
                    self.current_regime.stability = self._calculate_regime_stability()
            # Сохранение в буфер
            self.regime_buffer.append(
                {
                    "timestamp": datetime.now(),
                    "regime": regime_label,
                    "confidence": confidence,
                    "features": X[-1].tolist(),
                }
            )
        except Exception as e:
            logger.error(f"Error detecting regimes: {e}")

    async def _prepare_features(self) -> Optional[np.ndarray]:
        """Подготовка признаков для анализа."""
        try:
            if len(self.feature_buffer) < self.config.min_samples:
                return None
            # Преобразование в numpy массив
            features = np.array(list(self.feature_buffer))
            # Нормализация
            if not self.is_scaler_fitted:
                features = self.scaler.fit_transform(features)
                self.is_scaler_fitted = True
            else:
                features = self.scaler.transform(features)
            return features.astype(np.float64)
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

    async def _initialize_clustering_model(self, X: np.ndarray) -> None:
        """Инициализация модели кластеризации."""
        try:
            if self.config.clustering_method == "gmm":
                self.clustering_model = GaussianMixture(
                    n_components=self.config.n_clusters,
                    random_state=42,
                    covariance_type="full",
                )
            elif self.config.clustering_method == "kmeans":
                self.clustering_model = KMeans(
                    n_clusters=self.config.n_clusters, random_state=42
                )
            elif self.config.clustering_method == "dbscan":
                self.clustering_model = DBSCAN(
                    eps=0.5, min_samples=self.config.min_samples
                )
            else:
                self.clustering_model = GaussianMixture(
                    n_components=self.config.n_clusters, random_state=42
                )
            # Обучение модели
            self.clustering_model.fit(X)
            logger.info(
                f"Clustering model initialized: {self.config.clustering_method}"
            )
        except Exception as e:
            logger.error(f"Error initializing clustering model: {e}")

    def _calculate_regime_confidence(self, X: np.ndarray, regime_label: int) -> float:
        """Расчет уверенности в определении режима."""
        try:
            if self.clustering_model is not None and hasattr(self.clustering_model, "cluster_centers_"):
                center = self.clustering_model.cluster_centers_[regime_label]
                distance = float(np.linalg.norm(X[0] - center))
                return float(max(0, 1 - distance / 10))  # Нормализация
            else:
                return 0.8  # Значение по умолчанию
        except Exception as e:
            logger.error(f"Error calculating regime confidence: {e}")
            return 0.5

    async def _is_regime_change(self, new_regime: int, confidence: float) -> bool:
        """Проверка изменения режима."""
        try:
            # Если нет текущего режима
            if self.current_regime is None:
                return True
            # Проверка уверенности
            if confidence < self.config.regime_confidence_threshold:
                return False
            # Проверка изменения метки
            current_label = int(self.current_regime.regime_id.split("_")[-1])
            if new_regime != current_label:
                return True
            # Проверка стабильности
            if self.current_regime.stability < self.config.stability_threshold:
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking regime change: {e}")
            return False

    async def _handle_regime_change(self, new_regime: int, confidence: float) -> None:
        """Обработка изменения режима."""
        try:
            # Завершение текущего режима
            if self.current_regime:
                self.current_regime.end_time = datetime.now()
                self.current_regime.duration = (
                    self.current_regime.end_time - self.current_regime.start_time
                ).total_seconds()
                # Добавление в историю
                self.regime_history.append(self.current_regime)
                # Создание записи о переходе
                transition = RegimeTransition(
                    from_regime=self.current_regime.regime_id,
                    to_regime=f"regime_{new_regime}",
                    transition_time=datetime.now(),
                    confidence=confidence,
                    trigger_factors=self._identify_trigger_factors(),
                )
                self.transition_history.append(transition)
                logger.info(
                    f"Regime transition: {self.current_regime.regime_id} -> regime_{new_regime}"
                )
            # Создание нового режима
            regime_name = self._get_regime_name(new_regime)
            characteristics = self._extract_regime_characteristics(new_regime)
            self.current_regime = MarketRegime(
                regime_id=f"regime_{new_regime}",
                name=regime_name,
                characteristics=characteristics,
                confidence=confidence,
                start_time=datetime.now(),
                stability=1.0,
            )
            logger.info(
                f"New regime detected: {regime_name} (confidence: {confidence:.3f})"
            )
        except Exception as e:
            logger.error(f"Error handling regime change: {e}")

    def _get_regime_name(self, regime_label: int) -> str:
        """Получение имени режима."""
        regime_names = {
            0: "Trending Up",
            1: "Trending Down",
            2: "Sideways",
            3: "Volatile",
            4: "Consolidation",
        }
        return regime_names.get(regime_label, f"Regime {regime_label}")

    def _extract_regime_characteristics(self, regime_label: int) -> Dict[str, float]:
        """Извлечение характеристик режима."""
        try:
            if len(self.feature_buffer) < 10:
                return {}
            # Анализ последних данных
            recent_features = np.array(list(self.feature_buffer)[-10:])
            characteristics = {
                "volatility": float(np.std(recent_features)),
                "trend_strength": float(np.mean(np.diff(recent_features, axis=0))),
                "mean_value": float(np.mean(recent_features)),
                "range": float(np.max(recent_features) - np.min(recent_features)),
            }
            return characteristics
        except Exception as e:
            logger.error(f"Error extracting regime characteristics: {e}")
            return {}

    def _identify_trigger_factors(self) -> List[str]:
        """Определение факторов, вызвавших переход."""
        try:
            factors: List[str] = []
            if len(self.feature_buffer) < 20:
                return factors
            # Анализ изменений в признаках
            recent_features = np.array(list(self.feature_buffer)[-20:])
            # Волатильность
            volatility = np.std(recent_features)
            if volatility > 0.5:
                factors.append("high_volatility")
            # Тренд
            trend = np.mean(np.diff(recent_features, axis=0))
            if abs(trend) > 0.1:
                factors.append("strong_trend")
            # Резкие изменения
            changes = np.abs(np.diff(recent_features, axis=0))
            if np.max(changes) > 0.3:
                factors.append("sudden_change")
            return factors
        except Exception as e:
            logger.error(f"Error identifying trigger factors: {e}")
            return []

    def _calculate_regime_stability(self) -> float:
        """Расчет стабильности текущего режима."""
        try:
            if len(self.regime_buffer) < 10:
                return 1.0
            # Анализ последних предсказаний
            recent_regimes = [item["regime"] for item in list(self.regime_buffer)[-10:]]
            # Доля одинаковых режимов
            current_regime = recent_regimes[-1]
            stability = recent_regimes.count(current_regime) / len(recent_regimes)
            return stability
        except Exception as e:
            logger.error(f"Error calculating regime stability: {e}")
            return 0.5

    async def _analyze_transitions(self) -> None:
        """Анализ переходов между режимами."""
        try:
            if len(self.transition_history) < 2:
                return
            # Анализ паттернов переходов
            transition_patterns: Dict[str, List[RegimeTransition]] = {}
            for transition in self.transition_history[-50:]:  # Последние 50 переходов
                key = f"{transition.from_regime}->{transition.to_regime}"
                if key not in transition_patterns:
                    transition_patterns[key] = []
                transition_patterns[key].append(transition)
            # Выявление частых переходов
            frequent_transitions = [
                pattern
                for pattern, transitions in transition_patterns.items()
                if len(transitions) >= 3
            ]
            if frequent_transitions:
                logger.info(f"Frequent transitions detected: {frequent_transitions}")
        except Exception as e:
            logger.error(f"Error analyzing transitions: {e}")

    async def _update_statistics(self) -> None:
        """Обновление статистики режимов."""
        try:
            if not self.regime_history:
                return
            # Статистика по режимам
            for regime in self.regime_history:
                regime_id = regime.regime_id
                if regime_id not in self.regime_statistics:
                    self.regime_statistics[regime_id] = {
                        "count": 0,
                        "total_duration": 0.0,
                        "avg_duration": 0.0,
                        "avg_confidence": 0.0,
                        "avg_stability": 0.0,
                        "last_seen": None,
                    }
                stats = self.regime_statistics[regime_id]
                stats["count"] += 1
                stats["total_duration"] += regime.duration or 0.0
                stats["avg_duration"] = stats["total_duration"] / stats["count"]
                stats["avg_confidence"] = (
                    stats["avg_confidence"] * (stats["count"] - 1) + regime.confidence
                ) / stats["count"]
                stats["avg_stability"] = (
                    stats["avg_stability"] * (stats["count"] - 1) + regime.stability
                ) / stats["count"]
                stats["last_seen"] = regime.start_time
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

    async def _publish_regime_events(self) -> None:
        """Публикация событий о режимах."""
        try:
            if self.current_regime:
                from domain.types.messaging_types import EventPriority as MessagingEventPriority, EventName, EventType as MessagingEventType
                event = MessagingEvent(
                    name=EventName("system.health_check"),
                    type=MessagingEventType.SYSTEM_HEALTH_CHECK,  # Исправляю тип EventType
                    data={
                        "current_regime": self.current_regime.regime_id,
                        "regime_name": self.current_regime.name,
                        "confidence": self.current_regime.confidence,
                        "stability": self.current_regime.stability,
                        "characteristics": self.current_regime.characteristics,
                        "duration": (
                            datetime.now() - self.current_regime.start_time
                        ).total_seconds(),
                    },
                    priority=MessagingEventPriority.NORMAL,
                )
                await self.event_bus.publish(event)
        except Exception as e:
            logger.error(f"Error publishing regime events: {e}")

    def add_market_data(self, features: List[float]) -> None:
        """Добавление рыночных данных."""
        try:
            self.feature_buffer.append(features)
        except Exception as e:
            logger.error(f"Error adding market data: {e}")

    def get_current_regime(self) -> Optional[Dict[str, Any]]:
        """Получение текущего режима."""
        if self.current_regime is None:
            return None
        return {
            "regime_id": self.current_regime.regime_id,
            "name": self.current_regime.name,
            "confidence": self.current_regime.confidence,
            "stability": self.current_regime.stability,
            "characteristics": self.current_regime.characteristics,
            "start_time": self.current_regime.start_time.isoformat(),
            "duration": (
                datetime.now() - self.current_regime.start_time
            ).total_seconds(),
        }

    def get_regime_history(self) -> List[Dict[str, Any]]:
        """Получение истории режимов."""
        return [
            {
                "regime_id": regime.regime_id,
                "name": regime.name,
                "confidence": regime.confidence,
                "stability": regime.stability,
                "start_time": regime.start_time.isoformat(),
                "end_time": regime.end_time.isoformat() if regime.end_time else None,
                "duration": regime.duration,
            }
            for regime in self.regime_history
        ]

    def get_transition_history(self) -> List[Dict[str, Any]]:
        """Получение истории переходов."""
        return [
            {
                "from_regime": transition.from_regime,
                "to_regime": transition.to_regime,
                "transition_time": transition.transition_time.isoformat(),
                "confidence": transition.confidence,
                "trigger_factors": transition.trigger_factors,
            }
            for transition in self.transition_history
        ]

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Получение статистики режимов."""
        return self.regime_statistics.copy()

    def get_discovery_status(self) -> Dict[str, Any]:
        """Получение статуса обнаружения."""
        return {
            "is_discovering": self.is_discovering,
            "current_regime": (
                self.current_regime.regime_id if self.current_regime else None
            ),
            "total_regimes": len(self.regime_history),
            "total_transitions": len(self.transition_history),
            "data_points": len(self.feature_buffer),
            "model_initialized": self.clustering_model is not None,
        }

    async def predict_next_regime(
        self, horizon_minutes: int = 60
    ) -> Optional[Dict[str, Any]]:
        """Прогнозирование следующего режима."""
        try:
            if len(self.transition_history) < 5:
                return None
            # Простой прогноз на основе переходов
            current_regime = (
                self.current_regime.regime_id if self.current_regime else "unknown"
            )
            # Анализ переходов из текущего режима
            transitions_from_current = [
                t
                for t in self.transition_history[-100:]
                if t.from_regime == current_regime
            ]
            if not transitions_from_current:
                return None
            # Подсчет вероятностей переходов
            transition_counts: Dict[str, int] = {}
            for transition in transitions_from_current:
                to_regime = transition.to_regime
                transition_counts[to_regime] = transition_counts.get(to_regime, 0) + 1
            # Наиболее вероятный следующий режим
            total_transitions = sum(transition_counts.values())
            probabilities = {
                regime: count / total_transitions
                for regime, count in transition_counts.items()
            }
            most_likely_regime = max(probabilities, key=lambda k: probabilities[k])
            return {
                "predicted_regime": most_likely_regime,
                "confidence": probabilities[most_likely_regime],
                "all_probabilities": probabilities,
                "horizon_minutes": horizon_minutes,
                "prediction_time": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error predicting next regime: {e}")
            return None

    async def save_models(self) -> None:
        """Сохранение моделей."""
        try:
            if self.clustering_model is not None:
                model_path = f"{self.config.models_path}/clustering_model.joblib"
                joblib.dump(self.clustering_model, model_path)
            scaler_path = f"{self.config.models_path}/scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            logger.info("Regime discovery models saved")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    async def load_models(self) -> None:
        """Загрузка моделей."""
        try:
            model_path = f"{self.config.models_path}/clustering_model.joblib"
            if Path(model_path).exists():
                self.clustering_model = joblib.load(model_path)
            scaler_path = f"{self.config.models_path}/scaler.joblib"
            if Path(scaler_path).exists():
                self.scaler = joblib.load(scaler_path)
                self.is_scaler_fitted = True
            logger.info("Regime discovery models loaded")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

"""
Конкретные реализации эволюционирующих компонентов.
Содержит реализации всех абстрактных методов из EvolvableComponent
для различных типов компонентов системы.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor

from infrastructure.core.evolution_manager import EvolvableComponent


@dataclass
class ComponentState:
    """Состояние компонента."""

    name: str
    version: str
    parameters: Dict[str, Any]
    performance_history: List[float]
    last_update: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvolvableStrategy(EvolvableComponent):
    """
    Эволюционирующая стратегия.
    Автоматически адаптируется к изменениям рынка,
    оптимизирует параметры и улучшает производительность.
    """

    def __init__(self, name: str, strategy_type: str = "trend_following"):
        """Инициализация эволюционирующей стратегии."""
        super().__init__(name)
        self.strategy_type = strategy_type
        self.parameters = self._get_default_parameters()
        self.state = ComponentState(
            name=name,
            version="1.0.0",
            parameters=self.parameters.copy(),
            performance_history=[],
            last_update=datetime.now(),
        )
        self.adaptation_rate = 0.01
        self.learning_rate = 0.001
        self.adaptation_count = 0
        logger.info(f"EvolvableStrategy {name} initialized")

    def _get_default_parameters(self) -> Dict[str, Any]:
        """Получение параметров по умолчанию."""
        if self.strategy_type == "trend_following":
            return {
                "ema_period": 14,
                "trend_threshold": 0.01,
                "position_size": 1.0,
                "stop_loss": 0.02,
                "take_profit": 0.04,
            }
        elif self.strategy_type == "mean_reversion":
            return {
                "lookback_period": 20,
                "std_multiplier": 2.0,
                "position_size": 1.0,
                "stop_loss": 0.015,
                "take_profit": 0.03,
            }
        else:
            return {
                "window_size": 10,
                "threshold": 0.005,
                "position_size": 1.0,
                "stop_loss": 0.02,
                "take_profit": 0.04,
            }

    async def adapt(self, data: Any) -> bool:
        """Быстрая адаптация к новым данным."""
        try:
            if not data or not isinstance(data, dict):
                return False
            # Быстрая адаптация параметров
            market_data = data.get("market_data")
            if market_data is not None:
                await self._quick_adapt_parameters(market_data)
            # Обновление состояния
            self.state.last_update = datetime.now()
            self.adaptation_count += 1
            logger.debug(f"Quick adaptation completed for {self.name}")
            return True
        except Exception as e:
            logger.error(f"Error in quick adaptation for {self.name}: {e}")
            return False

    async def learn(self, data: Any) -> bool:
        """Обучение на новых данных."""
        try:
            if not data or not isinstance(data, dict):
                return False
            market_data = data.get("market_data")
            performance_data = data.get("performance_data", {})
            if market_data is not None:
                await self._learn_from_data(market_data, performance_data)
                logger.debug(f"Learning completed for {self.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error in learning for {self.name}: {e}")
            return False

    async def evolve(self, data: Any) -> bool:
        """Полная эволюция компонента."""
        try:
            if not data or not isinstance(data, dict):
                return False
            market_data = data.get("market_data")
            if market_data is not None:
                await self._full_evolution(market_data)
                logger.info(f"Evolution completed for {self.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error in evolution for {self.name}: {e}")
            return False

    def get_performance(self) -> float:
        """Получение производительности."""
        if not self.state.performance_history:
            return 0.0
        return sum(self.state.performance_history) / len(self.state.performance_history)

    def get_confidence(self) -> float:
        """Получение уверенности."""
        if not self.state.performance_history:
            return 0.0
        recent_performance = self.state.performance_history[-10:]
        if not recent_performance:
            return 0.0
        return sum(recent_performance) / len(recent_performance)

    def save_state(self, path: str) -> bool:
        """Сохранение состояния."""
        try:
            state_data = {
                "name": self.name,
                "strategy_type": self.strategy_type,
                "parameters": self.parameters,
                "state": {
                    "name": self.state.name,
                    "version": self.state.version,
                    "parameters": self.state.parameters,
                    "performance_history": self.state.performance_history,
                    "last_update": self.state.last_update.isoformat(),
                    "metadata": self.state.metadata,
                },
                "adaptation_rate": self.adaptation_rate,
                "learning_rate": self.learning_rate,
                "adaptation_count": self.adaptation_count,
            }
            joblib.dump(state_data, f"{path}.joblib")
            logger.info(f"State saved for {self.name} to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving state for {self.name}: {e}")
            return False

    def load_state(self, path: str) -> bool:
        """Загрузка состояния."""
        try:
            state_file = f"{path}.joblib"
            if not os.path.exists(state_file):
                logger.warning(f"State file not found: {state_file}")
                return False
            state_data = joblib.load(state_file)
            self.name = state_data["name"]
            self.strategy_type = state_data["strategy_type"]
            self.parameters = state_data["parameters"]
            self.adaptation_rate = state_data["adaptation_rate"]
            self.learning_rate = state_data["learning_rate"]
            self.adaptation_count = state_data["adaptation_count"]
            # Восстановление состояния
            state_dict = state_data["state"]
            self.state = ComponentState(
                name=state_dict["name"],
                version=state_dict["version"],
                parameters=state_dict["parameters"],
                performance_history=state_dict["performance_history"],
                last_update=datetime.fromisoformat(state_dict["last_update"]),
                metadata=state_dict["metadata"],
            )
            logger.info(f"State loaded for {self.name} from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading state for {self.name}: {e}")
            return False

    async def _quick_adapt_parameters(self, market_data: pd.DataFrame) -> None:
        """Быстрая адаптация параметров."""
        try:
            if market_data.empty:
                return
            # Адаптация на основе волатильности
            volatility = market_data["close"].pct_change().std()
            if volatility > 0.02:
                self.parameters["trend_threshold"] *= 1.1
            elif volatility < 0.01:
                self.parameters["trend_threshold"] *= 0.9
            # Ограничения
            self.parameters["trend_threshold"] = max(0.005, min(0.05, self.parameters["trend_threshold"]))
        except Exception as e:
            logger.error(f"Error in quick parameter adaptation: {e}")

    async def _learn_from_data(
        self, market_data: pd.DataFrame, performance_data: Dict[str, Any]
    ) -> None:
        """Обучение на данных."""
        try:
            if market_data.empty:
                return
            # Обновление производительности
            performance = performance_data.get("performance", 0.0)
            self.state.performance_history.append(performance)
            # Ограничение размера истории
            if len(self.state.performance_history) > 100:
                self.state.performance_history = self.state.performance_history[-100:]
            # Адаптация параметров на основе производительности
            if performance < 0.5:
                self.learning_rate *= 1.1
            elif performance > 0.8:
                self.learning_rate *= 0.9
        except Exception as e:
            logger.error(f"Error in learning from data: {e}")

    async def _full_evolution(self, market_data: pd.DataFrame) -> None:
        """Полная эволюция стратегии."""
        try:
            if market_data.empty:
                return
            # Генетическая оптимизация параметров
            best_parameters = await self._genetic_optimization(market_data)
            if best_parameters:
                self.parameters = best_parameters
                self.state.version = self._increment_version(self.state.version)
        except Exception as e:
            logger.error(f"Error in full evolution: {e}")

    async def _genetic_optimization(
        self, market_data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Генетическая оптимизация параметров."""
        try:
            if market_data.empty:
                return None
            # Простая генетическая оптимизация
            population_size = 10
            generations = 5
            population = [self.parameters.copy() for _ in range(population_size)]
            best_fitness = 0.0
            best_parameters = self.parameters.copy()
            for generation in range(generations):
                # Оценка приспособленности
                fitness_scores = []
                for parameters in population:
                    fitness = self._evaluate_fitness(parameters, market_data)
                    fitness_scores.append(fitness)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_parameters = parameters.copy()
                # Селекция и скрещивание
                new_population = []
                for _ in range(population_size):
                    parent1 = population[np.random.choice(len(population), p=np.array(fitness_scores) / sum(fitness_scores))]
                    parent2 = population[np.random.choice(len(population), p=np.array(fitness_scores) / sum(fitness_scores))]
                    child = self._crossover(parent1, parent2)
                    new_population.append(child)
                population = new_population
            return best_parameters
        except Exception as e:
            logger.error(f"Error in genetic optimization: {e}")
            return None

    def _evaluate_fitness(
        self, parameters: Dict[str, Any], market_data: pd.DataFrame
    ) -> float:
        """Оценка приспособленности параметров."""
        try:
            if market_data.empty:
                return 0.0
            # Простая оценка на основе исторических данных
            window_size = min(100, len(market_data))
            window: pd.DataFrame = market_data.iloc[-window_size:]
            signals = []
            for i in range(window_size - 10):
                signal = self._generate_signal(window.iloc[i:i+10], parameters)
                if signal is not None:
                    signals.append(signal)
            if not signals:
                return 0.0
            # Оценка на основе прибыльности сигналов
            positive_signals = sum(1 for s in signals if s > 0)
            return positive_signals / len(signals)
        except Exception as e:
            logger.error(f"Error evaluating fitness: {e}")
            return 0.0

    def _generate_signal(
        self, window: pd.DataFrame, parameters: Dict[str, Any]
    ) -> Optional[float]:
        """Генерация торгового сигнала."""
        try:
            if window.empty or len(window) < 10:
                return None
            # Простая стратегия тренд-следования
            ema_period = parameters.get("ema_period", 14)
            trend_threshold = parameters.get("trend_threshold", 0.01)
            # Расчет EMA
            ema = window["close"].ewm(span=ema_period).mean()
            current_ema = ema.iloc[-1]
            prev_ema = ema.iloc[-2]
            # Сигнал на основе тренда
            trend = (current_ema - prev_ema) / prev_ema
            if trend > trend_threshold:
                return 1.0  # Покупка
            elif trend < -trend_threshold:
                return -1.0  # Продажа
            return 0.0  # Нет сигнала
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None

    def _crossover(
        self, parent1: Dict[str, Any], parent2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Скрещивание параметров."""
        try:
            child = {}
            for key in parent1:
                if np.random.random() < 0.5:
                    child[key] = parent1[key]
                else:
                    child[key] = parent2[key]
                # Мутация
                if np.random.random() < 0.1:
                    if isinstance(child[key], (int, float)):
                        child[key] *= np.random.uniform(0.8, 1.2)
            return child
        except Exception as e:
            logger.error(f"Error in crossover: {e}")
            return parent1.copy()

    def _increment_version(self, version: str) -> str:
        """Инкремент версии."""
        try:
            parts = version.split(".")
            if len(parts) == 3:
                major, minor, patch = parts
                patch = str(int(patch) + 1)
                return f"{major}.{minor}.{patch}"
            return version
        except Exception:
            return version


class EvolvablePredictor(EvolvableComponent):
    """
    Эволюционирующий предиктор.
    Автоматически адаптируется к изменениям в данных,
    оптимизирует модель и улучшает точность предсказаний.
    """

    def __init__(self, name: str, model_type: str = "regression"):
        """Инициализация эволюционирующего предиктора."""
        super().__init__(name)
        self.model_type = model_type
        self.model = None
        self.feature_importance: Dict[str, float] = {}
        self.prediction_history: List[float] = []
        self.evolution_count = 0
        self.last_evolution = datetime.now()
        self.is_evolving = False
        logger.info(f"EvolvablePredictor {name} initialized")

    async def adapt(self, data: Any) -> bool:
        """Быстрая адаптация к новым данным."""
        try:
            if not data or not isinstance(data, dict):
                return False
            features = data.get("features")
            if features is not None:
                await self._quick_adapt_model(features)
                logger.debug(f"Quick adaptation completed for {self.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error in quick adaptation for {self.name}: {e}")
            return False

    async def learn(self, data: Any) -> bool:
        """Обучение на новых данных."""
        try:
            if not data or not isinstance(data, dict):
                return False
            features = data.get("features")
            targets = data.get("targets")
            if features is not None and targets is not None:
                await self._learn_model(features, targets)
                logger.debug(f"Learning completed for {self.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error in learning for {self.name}: {e}")
            return False

    async def evolve(self, data: Any) -> bool:
        """Полная эволюция компонента."""
        try:
            if not data or not isinstance(data, dict):
                return False
            features = data.get("features")
            targets = data.get("targets")
            if features is not None and targets is not None:
                await self._evolve_model(features, targets)
                self.evolution_count += 1
                self.last_evolution = datetime.now()
                logger.info(f"Evolution completed for {self.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error in evolution for {self.name}: {e}")
            return False

    def get_performance(self) -> float:
        """Получение производительности."""
        if not self.prediction_history:
            return 0.0
        # Простая метрика производительности
        return sum(self.prediction_history) / len(self.prediction_history)

    def get_confidence(self) -> float:
        """Получение уверенности."""
        if not self.prediction_history:
            return 0.0
        # Уверенность на основе стабильности предсказаний
        recent_predictions = self.prediction_history[-10:]
        if not recent_predictions:
            return 0.0
        variance = np.var(recent_predictions)
        return max(0.0, 1.0 - variance)

    def save_state(self, path: str) -> bool:
        """Сохранение состояния."""
        try:
            state_data = {
                "name": self.name,
                "model_type": self.model_type,
                "feature_importance": self.feature_importance,
                "prediction_history": self.prediction_history,
                "evolution_count": self.evolution_count,
                "last_evolution": self.last_evolution.isoformat(),
                "is_evolving": self.is_evolving,
            }
            joblib.dump(state_data, f"{path}.joblib")
            if self.model is not None:
                joblib.dump(self.model, f"{path}_model.joblib")
            logger.info(f"State saved for {self.name} to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving state for {self.name}: {e}")
            return False

    def load_state(self, path: str) -> bool:
        """Загрузка состояния."""
        try:
            state_file = f"{path}.joblib"
            model_file = f"{path}_model.joblib"
            if not os.path.exists(state_file):
                logger.warning(f"State file not found: {state_file}")
                return False
            # Загрузка состояния
            state_data = joblib.load(state_file)
            self.name = state_data["name"]
            self.model_type = state_data["model_type"]
            self.feature_importance = state_data["feature_importance"]
            self.prediction_history = state_data["prediction_history"]
            self.evolution_count = state_data["evolution_count"]
            self.last_evolution = datetime.fromisoformat(state_data["last_evolution"])
            self.is_evolving = state_data["is_evolving"]
            # Загрузка модели
            if os.path.exists(model_file):
                self.model = joblib.load(model_file)
            logger.info(f"State loaded for {self.name} from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading state for {self.name}: {e}")
            return False

    async def _quick_adapt_model(self, features: np.ndarray) -> None:
        """Быстрая адаптация модели."""
        try:
            if self.model is None:
                return
            # Простая адаптация весов
            if hasattr(self.model, 'partial_fit'):
                # Для моделей с поддержкой инкрементального обучения
                pass
        except Exception as e:
            logger.error(f"Error in quick model adaptation: {e}")

    async def _learn_model(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Обучение модели."""
        try:
            if features.size == 0 or targets.size == 0:
                return
            # Создание и обучение модели
            if self.model is None:
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            if self.model is not None:
                self.model.fit(features, targets)
                # Обновление важности признаков
                if hasattr(self.model, 'feature_importances_'):
                    feature_names = [f"feature_{i}" for i in range(features.shape[1])]
                    self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        except Exception as e:
            logger.error(f"Error in model learning: {e}")

    async def _evolve_model(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Эволюция модели."""
        try:
            if features.size == 0 or targets.size == 0:
                return
            # Создание новой модели с улучшенными параметрами
            new_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42
            )
            new_model.fit(features, targets)
            # Сравнение производительности
            if self.model is not None:
                old_score = self.model.score(features, targets)
                new_score = new_model.score(features, targets)
                if new_score > old_score:
                    self.model = new_model
                    # Обновление важности признаков
                    if hasattr(self.model, 'feature_importances_'):
                        feature_names = [f"feature_{i}" for i in range(features.shape[1])]
                        self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        except Exception as e:
            logger.error(f"Error in model evolution: {e}")


class EvolvableComponentFactory:
    """Фабрика для создания эволюционирующих компонентов."""

    @staticmethod
    def create_strategy(
        name: str, strategy_type: str = "trend_following"
    ) -> EvolvableStrategy:
        """Создание эволюционирующей стратегии."""
        return EvolvableStrategy(name, strategy_type)

    @staticmethod
    def create_predictor(
        name: str, model_type: str = "regression"
    ) -> EvolvablePredictor:
        """Создание эволюционирующего предиктора."""
        return EvolvablePredictor(name, model_type)

    @staticmethod
    def create_component(
        component_type: str, name: str, **kwargs
    ) -> EvolvableComponent:
        """Создание компонента по типу."""
        if component_type == "strategy":
            strategy_type = kwargs.get("strategy_type", "trend_following")
            return EvolvableComponentFactory.create_strategy(name, strategy_type)
        elif component_type == "predictor":
            model_type = kwargs.get("model_type", "regression")
            return EvolvableComponentFactory.create_predictor(name, model_type)
        else:
            raise ValueError(f"Unknown component type: {component_type}")

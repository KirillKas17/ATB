"""
Модуль метаобучения.
Обеспечивает обучение моделей на множестве задач и
адаптацию к новым задачам с минимальными данными.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Type aliases
DataFrame = pd.DataFrame
Series = pd.Series


@dataclass
class MetaLearningConfig:
    """Конфигурация метаобучения."""

    # Параметры обучения
    num_tasks: int = 10
    samples_per_task: int = 100
    adaptation_steps: int = 5
    learning_rate: float = 0.01
    # Параметры моделей
    base_model_type: str = "random_forest"
    meta_model_type: str = "linear"
    # Пороги
    performance_threshold: float = 0.7
    adaptation_threshold: float = 0.05
    # Пути
    models_path: str = "models/meta_learning"
    task_data_path: str = "data/meta_tasks"


@dataclass
class Task:
    """Задача для метаобучения."""

    task_id: str
    features: np.ndarray
    targets: np.ndarray
    task_type: str
    difficulty: float
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelState:
    """Состояние модели."""

    model_id: str
    task_id: str
    parameters: Dict[str, Any]
    performance: float
    adaptation_steps: int
    last_update: datetime
    is_active: bool = True


class MetaLearning:
    """
    Система метаобучения.
    Обеспечивает:
    - Обучение на множестве задач
    - Быструю адаптацию к новым задачам
    - Перенос знаний между задачами
    - Оптимизацию гиперпараметров
    """

    def __init__(self, config: Optional[MetaLearningConfig] = None) -> None:
        """Инициализация системы метаобучения."""
        self.config = config or MetaLearningConfig()
        # Создание директорий
        Path(self.config.models_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.task_data_path).mkdir(parents=True, exist_ok=True)
        # Модели
        self.base_models: Dict[str, Any] = {}
        self.meta_models: Dict[str, Any] = {}
        self.model_states: Dict[str, ModelState] = {}
        # Задачи
        self.tasks: Dict[str, Task] = {}
        self.task_performance: Dict[str, List[float]] = {}
        # Мета-знания
        self.meta_knowledge: Dict[str, Any] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
        # Состояние
        self.is_learning: bool = False
        self.learning_task: Optional[asyncio.Task] = None
        # Предобработка
        self.scaler = StandardScaler()
        logger.info("MetaLearning initialized")

    async def start_learning(self) -> None:
        """Запуск метаобучения."""
        if self.is_learning:
            logger.warning("Meta-learning is already running")
            return
        self.is_learning = True
        self.learning_task = asyncio.create_task(self._learning_loop())
        logger.info("Meta-learning started")

    async def stop_learning(self) -> None:
        """Остановка метаобучения."""
        if not self.is_learning:
            return
        self.is_learning = False
        if self.learning_task:
            self.learning_task.cancel()
            try:
                await self.learning_task
            except asyncio.CancelledError:
                pass
        logger.info("Meta-learning stopped")

    async def _learning_loop(self) -> None:
        """Основной цикл метаобучения."""
        logger.info("Starting meta-learning loop")
        while self.is_learning:
            try:
                # Создание новых задач
                await self._create_new_tasks()
                # Обучение на существующих задачах
                await self._learn_on_tasks()
                # Обновление мета-знаний
                await self._update_meta_knowledge()
                # Очистка старых задач
                await self._cleanup_old_tasks()
                # Ожидание следующей итерации
                await asyncio.sleep(3600)  # Каждый час
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in meta-learning loop: {e}")
                await asyncio.sleep(300)

    async def _create_new_tasks(self) -> None:
        """Создание новых задач для обучения."""
        try:
            # Генерация синтетических задач
            for i in range(3):  # Создаем 3 новые задачи
                task = await self._generate_synthetic_task()
                if task:
                    self.tasks[task.task_id] = task
                    logger.info(f"Created new task: {task.task_id}")
        except Exception as e:
            logger.error(f"Error creating new tasks: {e}")

    async def _generate_synthetic_task(self) -> Optional[Task]:
        """Генерация синтетической задачи."""
        try:
            task_id = f"task_{len(self.tasks)}_{int(time.time())}"
            # Генерация признаков
            n_samples = self.config.samples_per_task
            n_features = 10
            # Различные паттерны для разных задач
            task_type = np.random.choice(["trend", "mean_reversion", "volatility"])
            if task_type == "trend":
                # Трендовые данные
                features = np.random.randn(n_samples, n_features)
                targets = np.sum(
                    features * np.random.randn(n_features), axis=1
                ) + np.cumsum(np.random.randn(n_samples) * 0.1)
            elif task_type == "mean_reversion":
                # Данные с возвратом к среднему
                features = np.random.randn(n_samples, n_features)
                targets = np.random.randn(n_samples) * 0.5 + np.mean(features, axis=1)
            else:  # volatility
                # Волатильные данные
                features = np.random.randn(n_samples, n_features)
                targets = np.random.randn(n_samples) * np.std(features, axis=1)
            # Сложность задачи
            difficulty = np.random.uniform(0.1, 0.9)
            return Task(
                task_id=task_id,
                features=features,
                targets=targets,
                task_type=task_type,
                difficulty=difficulty,
                created_at=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Error generating synthetic task: {e}")
            return None

    async def _learn_on_tasks(self) -> None:
        """Обучение на задачах."""
        try:
            for task_id, task in self.tasks.items():
                # Проверка необходимости обучения
                if await self._should_learn_task(task_id):
                    await self._learn_task(task)
        except Exception as e:
            logger.error(f"Error learning on tasks: {e}")

    async def _should_learn_task(self, task_id: str) -> bool:
        """Проверка необходимости обучения на задаче."""
        try:
            # Проверка производительности
            if task_id in self.task_performance:
                recent_performance = np.mean(self.task_performance[task_id][-5:])
                if recent_performance < self.config.performance_threshold:
                    return True
            # Проверка времени последнего обучения
            if task_id not in self.model_states:
                return True
            last_update = self.model_states[task_id].last_update
            if datetime.now() - last_update > timedelta(hours=6):
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking task learning need: {e}")
            return True

    async def _learn_task(self, task: Task) -> None:
        """Обучение на конкретной задаче."""
        try:
            logger.info(f"Learning on task: {task.task_id}")
            # Разделение данных
            split_idx = int(len(task.features) * 0.8)
            X_train = task.features[:split_idx]
            y_train = task.targets[:split_idx]
            X_test = task.features[split_idx:]
            y_test = task.targets[split_idx:]
            # Создание базовой модели
            base_model = self._create_base_model()
            base_model.fit(X_train, y_train)
            # Оценка производительности
            y_pred = base_model.predict(X_test)
            performance = 1.0 / (1.0 + mean_squared_error(y_test, y_pred))
            # Адаптация с мета-обучением
            adapted_model = await self._adapt_model(base_model, task)
            # Оценка адаптированной модели
            adapted_pred = adapted_model.predict(X_test)
            adapted_performance = 1.0 / (1.0 + mean_squared_error(y_test, adapted_pred))
            # Сохранение результатов
            model_id = f"{task.task_id}_model"
            self.base_models[model_id] = adapted_model
            self.model_states[model_id] = ModelState(
                model_id=model_id,
                task_id=task.task_id,
                parameters=self._extract_parameters(adapted_model),
                performance=adapted_performance,
                adaptation_steps=self.config.adaptation_steps,
                last_update=datetime.now(),
            )
            self.task_performance[task.task_id].append(adapted_performance)
            # Ограничение истории производительности
            if len(self.task_performance[task.task_id]) > 100:
                self.task_performance[task.task_id] = self.task_performance[
                    task.task_id
                ][-100:]
            logger.info(
                f"Task {task.task_id} learned. Performance: {adapted_performance:.3f}"
            )
        except Exception as e:
            logger.error(f"Error learning task {task.task_id}: {e}")

    def _create_base_model(self) -> Any:
        """Создание базовой модели."""
        if self.config.base_model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
        elif self.config.base_model_type == "linear":
            return LinearRegression()
        else:
            return RandomForestRegressor(n_estimators=100, random_state=42)

    async def _adapt_model(self, base_model: Any, task: Task) -> Any:
        """Адаптация модели с использованием мета-знаний."""
        try:
            adapted_model = base_model
            # Применение мета-знаний
            if self.meta_knowledge:
                adapted_model = await self._apply_meta_knowledge(adapted_model, task)
            # Несколько шагов адаптации
            for step in range(self.config.adaptation_steps):
                adapted_model = await self._adapt_step(adapted_model, task, step)
            return adapted_model
        except Exception as e:
            logger.error(f"Error adapting model: {e}")
            return base_model

    async def _apply_meta_knowledge(self, model: Any, task: Task) -> Any:
        """Применение мета-знаний к модели."""
        try:
            # Здесь должна быть логика применения мета-знаний
            # Пока возвращаем модель без изменений
            return model
        except Exception as e:
            logger.error(f"Error applying meta knowledge: {e}")
            return model

    async def _adapt_step(self, model: Any, task: Task, step: int) -> Any:
        """Один шаг адаптации модели."""
        try:
            # Простая адаптация через дообучение на части данных
            subset_size = int(len(task.features) * 0.2)
            subset_idx = np.random.choice(
                len(task.features), subset_size, replace=False
            )
            X_subset = task.features[subset_idx]
            y_subset = task.targets[subset_idx]
            # Дообучение модели
            if hasattr(model, "partial_fit"):
                model.partial_fit(X_subset, y_subset)
            else:
                # Для моделей без partial_fit создаем новую
                model = self._create_base_model()
                model.fit(X_subset, y_subset)
            return model
        except Exception as e:
            logger.error(f"Error in adaptation step {step}: {e}")
            return model

    def _extract_parameters(self, model: Any) -> Dict[str, Any]:
        """Извлечение параметров модели."""
        try:
            if hasattr(model, "get_params"):
                return model.get_params()
            else:
                return {"model_type": type(model).__name__}
        except Exception:
            return {"model_type": "unknown"}

    async def _update_meta_knowledge(self) -> None:
        """Обновление мета-знаний."""
        try:
            if len(self.model_states) < 5:
                return
            # Анализ производительности моделей
            performances = []
            for model_id, state in self.model_states.items():
                performances.append(
                    {
                        "model_id": model_id,
                        "performance": state.performance,
                        "task_type": (
                            self.tasks[state.task_id].task_type
                            if state.task_id in self.tasks
                            else "unknown"
                        ),
                        "difficulty": (
                            self.tasks[state.task_id].difficulty
                            if state.task_id in self.tasks
                            else 0.5
                        ),
                    }
                )
            # Обновление мета-знаний
            valid_performances = [float(p["performance"]) for p in performances if p["performance"] is not None and isinstance(p["performance"], (int, float))]
            self.meta_knowledge = {
                "average_performance": np.mean(valid_performances) if valid_performances else 0.0,
                "best_performance": max(valid_performances) if valid_performances else 0.0,
                "task_type_performance": self._aggregate_by_task_type(performances),
                "difficulty_performance": self._aggregate_by_difficulty(performances),
                "last_update": datetime.now(),
            }
            logger.info("Meta-knowledge updated")
        except Exception as e:
            logger.error(f"Error updating meta knowledge: {e}")

    def _aggregate_by_task_type(
        self, performances: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Агрегация производительности по типу задач."""
        try:
            type_performance = defaultdict(list)
            for p in performances:
                type_performance[p["task_type"]].append(p["performance"])
            return {
                task_type: np.mean(perfs)
                for task_type, perfs in type_performance.items()
            }
        except Exception:
            return {}

    def _aggregate_by_difficulty(
        self, performances: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Агрегация производительности по сложности."""
        try:
            difficulty_ranges = {
                "easy": (0.0, 0.3),
                "medium": (0.3, 0.7),
                "hard": (0.7, 1.0),
            }
            range_performance = defaultdict(list)
            for p in performances:
                difficulty = p["difficulty"]
                for range_name, (min_val, max_val) in difficulty_ranges.items():
                    if min_val <= difficulty < max_val:
                        range_performance[range_name].append(p["performance"])
                        break
            return {
                range_name: np.mean(perfs)
                for range_name, perfs in range_performance.items()
            }
        except Exception:
            return {}

    async def _cleanup_old_tasks(self) -> None:
        """Очистка старых задач."""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)
            tasks_to_remove = []
            for task_id, task in self.tasks.items():
                if task.created_at < cutoff_time:
                    tasks_to_remove.append(task_id)
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
                if task_id in self.task_performance:
                    del self.task_performance[task_id]
            if tasks_to_remove:
                logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
        except Exception as e:
            logger.error(f"Error cleaning up old tasks: {e}")

    async def adapt_to_new_task(
        self, features: np.ndarray, targets: np.ndarray, task_type: str = "unknown"
    ) -> Any:
        """Быстрая адаптация к новой задаче."""
        try:
            logger.info(f"Adapting to new task of type: {task_type}")
            # Создание новой задачи
            task = Task(
                task_id=f"new_task_{int(time.time())}",
                features=features,
                targets=targets,
                task_type=task_type,
                difficulty=0.5,  # Средняя сложность
                created_at=datetime.now(),
            )
            # Создание базовой модели
            base_model = self._create_base_model()
            base_model.fit(features, targets)
            # Быстрая адаптация с мета-знаниями
            adapted_model = await self._adapt_model(base_model, task)
            # Сохранение задачи и модели
            self.tasks[task.task_id] = task
            model_id = f"{task.task_id}_model"
            self.base_models[model_id] = adapted_model
            self.model_states[model_id] = ModelState(
                model_id=model_id,
                task_id=task.task_id,
                parameters=self._extract_parameters(adapted_model),
                performance=0.8,  # Предполагаемая производительность
                adaptation_steps=self.config.adaptation_steps,
                last_update=datetime.now(),
            )
            logger.info(f"Successfully adapted to new task: {task.task_id}")
            return adapted_model
        except Exception as e:
            logger.error(f"Error adapting to new task: {e}")
            return None

    def get_meta_knowledge(self) -> Dict[str, Any]:
        """Получение мета-знаний."""
        return self.meta_knowledge.copy()

    def get_task_performance(self) -> Dict[str, List[float]]:
        """Получение производительности задач."""
        return dict(self.task_performance)

    def get_learning_status(self) -> Dict[str, Any]:
        """Получение статуса обучения."""
        return {
            "is_learning": self.is_learning,
            "total_tasks": len(self.tasks),
            "total_models": len(self.model_states),
            "active_models": len(
                [s for s in self.model_states.values() if s.is_active]
            ),
            "average_performance": (
                np.mean([s.performance for s in self.model_states.values()])
                if self.model_states
                else 0.0
            ),
            "meta_knowledge_updated": self.meta_knowledge.get("last_update", "never"),
        }

    async def save_models(self) -> None:
        """Сохранение моделей."""
        try:
            import os

            os.makedirs(self.config.models_path, exist_ok=True)
            # Сохранение базовых моделей
            for model_id, model in self.base_models.items():
                model_path = f"{self.config.models_path}/{model_id}.joblib"
                joblib.dump(model, model_path)
            # Сохранение мета-знаний
            meta_path = f"{self.config.models_path}/meta_knowledge.joblib"
            joblib.dump(self.meta_knowledge, meta_path)
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    async def load_models(self) -> None:
        """Загрузка моделей."""
        try:
            import os

            # Загрузка базовых моделей
            for filename in os.listdir(self.config.models_path):
                if filename.endswith(".joblib") and not filename.startswith("meta"):
                    model_id = filename.replace(".joblib", "")
                    model_path = f"{self.config.models_path}/{filename}"
                    self.base_models[model_id] = joblib.load(model_path)
            # Загрузка мета-знаний
            meta_path = f"{self.config.models_path}/meta_knowledge.joblib"
            if os.path.exists(meta_path):
                self.meta_knowledge = joblib.load(meta_path)
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

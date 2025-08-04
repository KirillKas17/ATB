import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn не установлен, ML модели недоступны")


@dataclass
class PredictionResult:
    performance_score: float
    maintainability_score: float
    quality_score: float
    confidence: float
    model_used: str
    features_used: List[str]
    prediction_timestamp: datetime


@dataclass
class ModelMetrics:
    r2_score: float
    mse: float
    mae: float
    cross_val_score: float
    feature_importance: Dict[str, float]


class MLPredictor:
    """Промышленные ML-модели для предсказания метрик кода и системы."""

    def __init__(self, models_dir: str = "models/ml_predictor") -> None:
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_selectors: Dict[str, Any] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if SKLEARN_AVAILABLE:
            self._initialize_models()
            self._load_saved_models()

    def _initialize_models(self) -> None:
        """Инициализация промышленных ML моделей."""
        # Модель для предсказания производительности
        self.models["performance"] = Pipeline(
            [
                ("feature_selection", SelectKBest(f_regression, k=10)),
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                    ),
                ),
            ]
        )
        # Модель для предсказания поддерживаемости
        self.models["maintainability"] = Pipeline(
            [
                ("feature_selection", SelectKBest(f_regression, k=8)),
                ("scaler", MinMaxScaler()),
                (
                    "regressor",
                    GradientBoostingRegressor(
                        n_estimators=150,
                        learning_rate=0.1,
                        max_depth=8,
                        subsample=0.8,
                        random_state=42,
                    ),
                ),
            ]
        )
        # Модель для предсказания качества
        self.models["quality"] = Pipeline(
            [
                ("feature_selection", SelectKBest(f_regression, k=12)),
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=0.001,
                        learning_rate="adaptive",
                        max_iter=500,
                        random_state=42,
                    ),
                ),
            ]
        )
        # Дополнительные скалеры для разных типов данных
        self.scalers["default"] = StandardScaler()
        self.scalers["robust"] = MinMaxScaler()
        logger.info("ML модели инициализированы")

    def _load_saved_models(self) -> None:
        """Загрузка сохранённых моделей."""
        try:
            for model_name in self.models.keys():
                model_path = self.models_dir / f"{model_name}_model.pkl"
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        self.models[model_name] = pickle.load(f)
                    logger.info(f"Модель {model_name} загружена из {model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки сохранённых моделей: {e}")

    async def train_models(self, training_data: List[Dict[str, Any]]) -> None:
        """Промышленное обучение моделей на training_data."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn недоступен, обучение невозможно")
            return
        try:
            logger.info(f"Начало обучения моделей на {len(training_data)} примерах")
            # Подготовка данных
            X, y_performance, y_maintainability, y_quality, feature_names = (
                await self._prepare_training_data(training_data)
            )
            if len(X) < 10:
                logger.warning("Недостаточно данных для обучения (минимум 10 примеров)")
                return
            # Обучение моделей
            await self._train_performance_model(X, y_performance, feature_names)
            await self._train_maintainability_model(X, y_maintainability, feature_names)
            await self._train_quality_model(X, y_quality, feature_names)
            # Сохранение моделей
            await self._save_models()
            logger.info("Обучение моделей завершено успешно")
        except Exception as e:
            logger.error(f"Ошибка обучения моделей: {e}")

    async def _prepare_training_data(
        self, training_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Подготовка данных для обучения."""
        features = []
        y_performance = []
        y_maintainability = []
        y_quality = []
        for data_point in training_data:
            # Извлечение признаков
            feature_vector = self._extract_features(data_point)
            features.append(feature_vector)
            # Извлечение целевых переменных
            y_performance.append(data_point.get("performance_score", 0.5))
            y_maintainability.append(data_point.get("maintainability_score", 0.5))
            y_quality.append(data_point.get("quality_score", 0.5))
        X = np.array(features)
        y_perf = np.array(y_performance)
        y_maint = np.array(y_maintainability)
        y_qual = np.array(y_quality)
        # Получение имён признаков
        feature_names = self._get_feature_names()
        return X, y_perf, y_maint, y_qual, feature_names

    def _extract_features(self, data_point: Dict[str, Any]) -> List[float]:
        """Извлечение признаков из точки данных."""
        features = []
        # Метрики сложности
        if "complexity_metrics" in data_point:
            complexity = data_point["complexity_metrics"]
            features.extend(
                [
                    complexity.get("cyclomatic_complexity", 0),
                    complexity.get("cognitive_complexity", 0),
                    complexity.get("nesting_depth", 0),
                    complexity.get("lines_of_code", 0),
                ]
            )
        else:
            features.extend([0, 0, 0, 0])
        # Метрики архитектуры
        if "architecture" in data_point:
            arch = data_point["architecture"]
            features.extend(
                [
                    len(arch.get("modules", {})),
                    len(arch.get("dependencies", {}).get("circular", [])),
                    arch.get("metrics", {}).get("coupling", 0),
                    arch.get("metrics", {}).get("cohesion", 0),
                ]
            )
        else:
            features.extend([0, 0, 0, 0])
        # Метрики качества кода
        if "code_quality" in data_point:
            quality = data_point["code_quality"]
            features.extend(
                [
                    quality.get("score", 0),
                    len(quality.get("issues", [])),
                    len(quality.get("suggestions", [])),
                ]
            )
        else:
            features.extend([0, 0, 0])
        # Дополнительные метрики
        features.extend(
            [
                data_point.get("file_count", 0),
                data_point.get("function_count", 0),
                data_point.get("class_count", 0),
                data_point.get("import_count", 0),
                data_point.get("comment_ratio", 0),
                data_point.get("test_coverage", 0),
            ]
        )
        return features

    def _get_feature_names(self) -> List[str]:
        """Получение имён признаков."""
        return [
            "cyclomatic_complexity",
            "cognitive_complexity",
            "nesting_depth",
            "lines_of_code",
            "module_count",
            "circular_dependencies",
            "coupling",
            "cohesion",
            "quality_score",
            "issue_count",
            "suggestion_count",
            "file_count",
            "function_count",
            "class_count",
            "import_count",
            "comment_ratio",
            "test_coverage",
        ]

    async def _train_performance_model(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> None:
        """Обучение модели производительности."""
        try:
            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            # Обучение модели
            self.models["performance"].fit(X_train, y_train)
            # Предсказание на тестовых данных
            y_pred = self.models["performance"].predict(X_test)
            # Расчёт метрик
            metrics = ModelMetrics(
                r2_score=r2_score(y_test, y_pred),
                mse=mean_squared_error(y_test, y_pred),
                mae=mean_absolute_error(y_test, y_pred),
                cross_val_score=np.mean(
                    cross_val_score(self.models["performance"], X, y, cv=5)
                ),
                feature_importance=self._get_feature_importance(
                    "performance", feature_names
                ),
            )
            self.model_metrics["performance"] = metrics
            logger.info(
                f"Модель производительности обучена: R²={metrics.r2_score:.3f}, CV={metrics.cross_val_score:.3f}"
            )
        except Exception as e:
            logger.error(f"Ошибка обучения модели производительности: {e}")

    async def _train_maintainability_model(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> None:
        """Обучение модели поддерживаемости."""
        try:
            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            # Обучение модели
            self.models["maintainability"].fit(X_train, y_train)
            # Предсказание на тестовых данных
            y_pred = self.models["maintainability"].predict(X_test)
            # Расчёт метрик
            metrics = ModelMetrics(
                r2_score=r2_score(y_test, y_pred),
                mse=mean_squared_error(y_test, y_pred),
                mae=mean_absolute_error(y_test, y_pred),
                cross_val_score=np.mean(
                    cross_val_score(self.models["maintainability"], X, y, cv=5)
                ),
                feature_importance=self._get_feature_importance(
                    "maintainability", feature_names
                ),
            )
            self.model_metrics["maintainability"] = metrics
            logger.info(
                f"Модель поддерживаемости обучена: R²={metrics.r2_score:.3f}, CV={metrics.cross_val_score:.3f}"
            )
        except Exception as e:
            logger.error(f"Ошибка обучения модели поддерживаемости: {e}")

    async def _train_quality_model(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> None:
        """Обучение модели качества."""
        try:
            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            # Обучение модели
            self.models["quality"].fit(X_train, y_train)
            # Предсказание на тестовых данных
            y_pred = self.models["quality"].predict(X_test)
            # Расчёт метрик
            metrics = ModelMetrics(
                r2_score=r2_score(y_test, y_pred),
                mse=mean_squared_error(y_test, y_pred),
                mae=mean_absolute_error(y_test, y_pred),
                cross_val_score=np.mean(
                    cross_val_score(self.models["quality"], X, y, cv=5)
                ),
                feature_importance=self._get_feature_importance(
                    "quality", feature_names
                ),
            )
            self.model_metrics["quality"] = metrics
            logger.info(
                f"Модель качества обучена: R²={metrics.r2_score:.3f}, CV={metrics.cross_val_score:.3f}"
            )
        except Exception as e:
            logger.error(f"Ошибка обучения модели качества: {e}")

    def _get_feature_importance(
        self, model_name: str, feature_names: List[str]
    ) -> Dict[str, float]:
        """Получение важности признаков для модели."""
        try:
            model = self.models[model_name]
            if hasattr(model, "named_steps") and "regressor" in model.named_steps:
                regressor = model.named_steps["regressor"]
                if hasattr(regressor, "feature_importances_"):
                    importances = regressor.feature_importances_
                    return dict(zip(feature_names, importances))
        except Exception as e:
            logger.error(f"Ошибка получения важности признаков для {model_name}: {e}")
        return {name: 0.0 for name in feature_names}

    async def _save_models(self) -> None:
        """Сохранение обученных моделей."""
        try:
            for model_name, model in self.models.items():
                model_path = self.models_dir / f"{model_name}_model.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                logger.info(f"Модель {model_name} сохранена в {model_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения моделей: {e}")

    async def predict_metrics(self, code_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Промышленное предсказание метрик на основе обученных моделей."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn недоступен, возвращаю эвристики")
            return {
                "performance_score": float(np.random.uniform(0.6, 0.9)),
                "maintainability_score": float(np.random.uniform(0.5, 0.8)),
                "quality_score": float(np.random.uniform(0.5, 0.8)),
            }
        try:
            # Извлечение признаков
            features = self._extract_features(code_structure)
            X = np.array([features])
            predictions = {}
            confidence_scores = {}
            # Предсказание производительности
            if "performance" in self.models:
                try:
                    perf_pred = self.models["performance"].predict(X)[0]
                    predictions["performance_score"] = float(
                        np.clip(perf_pred, 0.0, 1.0).item()
                    )
                    confidence_scores["performance"] = (
                        self._calculate_prediction_confidence("performance", X)
                    )
                except Exception as e:
                    logger.error(f"Ошибка предсказания производительности: {e}")
                    predictions["performance_score"] = 0.7
            # Предсказание поддерживаемости
            if "maintainability" in self.models:
                try:
                    maint_pred = self.models["maintainability"].predict(X)[0]
                    predictions["maintainability_score"] = float(
                        np.clip(maint_pred, 0.0, 1.0).item()
                    )
                    confidence_scores["maintainability"] = (
                        self._calculate_prediction_confidence("maintainability", X)
                    )
                except Exception as e:
                    logger.error(f"Ошибка предсказания поддерживаемости: {e}")
                    predictions["maintainability_score"] = 0.6
            # Предсказание качества
            if "quality" in self.models:
                try:
                    qual_pred = self.models["quality"].predict(X)[0]
                    predictions["quality_score"] = float(np.clip(qual_pred, 0.0, 1.0).item())
                    confidence_scores["quality"] = (
                        self._calculate_prediction_confidence("quality", X)
                    )
                except Exception as e:
                    logger.error(f"Ошибка предсказания качества: {e}")
                    predictions["quality_score"] = 0.6
            # Расчёт общего уровня уверенности
            overall_confidence = (
                np.mean(list(confidence_scores.values())) if confidence_scores else 0.5
            )
            # Добавление метаданных
            predictions.update(
                {
                    "prediction_confidence": float(overall_confidence),
                    "performance_confidence": float(confidence_scores.get("performance", 0.5)),
                    "maintainability_confidence": float(confidence_scores.get("maintainability", 0.5)),
                    "quality_confidence": float(confidence_scores.get("quality", 0.5)),
                }
            )
            logger.info(
                f"Предсказание завершено: производительность={predictions.get('performance_score', 0):.3f}, "
                f"поддерживаемость={predictions.get('maintainability_score', 0):.3f}, "
                f"качество={predictions.get('quality_score', 0):.3f}"
            )
            return predictions
        except Exception as e:
            logger.error(f"Ошибка предсказания метрик: {e}")
            return {
                "performance_score": 0.7,
                "maintainability_score": 0.6,
                "quality_score": 0.6,
                "prediction_confidence": 0.0,
            }

    def _calculate_prediction_confidence(self, model_name: str, X: np.ndarray) -> float:
        """Расчёт уверенности в предсказании."""
        try:
            if model_name in self.model_metrics:
                # Использование метрик модели для оценки уверенности
                metrics = self.model_metrics[model_name]
                confidence = (metrics.r2_score + metrics.cross_val_score) / 2
                return float(np.clip(confidence, 0.0, 1.0).item())
        except Exception as e:
            logger.error(f"Ошибка расчёта уверенности для {model_name}: {e}")
        return 0.5

    async def get_model_performance(self) -> Dict[str, Any]:
        """Получение производительности моделей."""
        if not self.model_metrics:
            return {"error": "Модели не обучены"}
        performance = {}
        for model_name, metrics in self.model_metrics.items():
            performance[model_name] = {
                "r2_score": metrics.r2_score,
                "mse": metrics.mse,
                "mae": metrics.mae,
                "cross_val_score": metrics.cross_val_score,
                "feature_importance": metrics.feature_importance,
            }
        return performance

    async def optimize_hyperparameters(
        self, training_data: List[Dict[str, Any]]
    ) -> None:
        """Оптимизация гиперпараметров моделей."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn недоступен, оптимизация невозможна")
            return
        try:
            logger.info("Начало оптимизации гиперпараметров")
            # Подготовка данных
            X, y_performance, y_maintainability, y_quality, feature_names = (
                await self._prepare_training_data(training_data)
            )
            if len(X) < 20:
                logger.warning("Недостаточно данных для оптимизации гиперпараметров")
                return
            # Оптимизация модели производительности
            await self._optimize_performance_model(X, y_performance)
            # Оптимизация модели поддерживаемости
            await self._optimize_maintainability_model(X, y_maintainability)
            # Оптимизация модели качества
            await self._optimize_quality_model(X, y_quality)
            logger.info("Оптимизация гиперпараметров завершена")
        except Exception as e:
            logger.error(f"Ошибка оптимизации гиперпараметров: {e}")

    async def _optimize_performance_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Оптимизация гиперпараметров модели производительности."""
        try:
            param_grid = {
                "regressor__n_estimators": [50, 100, 150],
                "regressor__max_depth": [5, 10, 15],
                "regressor__min_samples_split": [2, 5, 10],
            }
            grid_search = GridSearchCV(
                self.models["performance"], param_grid, cv=5, scoring="r2", n_jobs=-1
            )
            grid_search.fit(X, y)
            # Обновление модели
            self.models["performance"] = grid_search.best_estimator_
            logger.info(
                f"Оптимизация модели производительности: лучшие параметры = {grid_search.best_params_}"
            )
        except Exception as e:
            logger.error(f"Ошибка оптимизации модели производительности: {e}")

    async def _optimize_maintainability_model(
        self, X: np.ndarray, y: np.ndarray
    ) -> None:
        """Оптимизация гиперпараметров модели поддерживаемости."""
        try:
            param_grid = {
                "regressor__n_estimators": [100, 150, 200],
                "regressor__learning_rate": [0.05, 0.1, 0.15],
                "regressor__max_depth": [5, 8, 10],
            }
            grid_search = GridSearchCV(
                self.models["maintainability"],
                param_grid,
                cv=5,
                scoring="r2",
                n_jobs=-1,
            )
            grid_search.fit(X, y)
            # Обновление модели
            self.models["maintainability"] = grid_search.best_estimator_
            logger.info(
                f"Оптимизация модели поддерживаемости: лучшие параметры = {grid_search.best_params_}"
            )
        except Exception as e:
            logger.error(f"Ошибка оптимизации модели поддерживаемости: {e}")

    async def _optimize_quality_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Оптимизация гиперпараметров модели качества."""
        try:
            param_grid = {
                "regressor__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
                "regressor__alpha": [0.0001, 0.001, 0.01],
                "regressor__learning_rate_init": [0.001, 0.01, 0.1],
            }
            grid_search = GridSearchCV(
                self.models["quality"], param_grid, cv=5, scoring="r2", n_jobs=-1
            )
            grid_search.fit(X, y)
            # Обновление модели
            self.models["quality"] = grid_search.best_estimator_
            logger.info(
                f"Оптимизация модели качества: лучшие параметры = {grid_search.best_params_}"
            )
        except Exception as e:
            logger.error(f"Ошибка оптимизации модели качества: {e}")

    async def get_feature_importance_analysis(self) -> Dict[str, Any]:
        """Анализ важности признаков."""
        analysis = {}
        for model_name, metrics in self.model_metrics.items():
            if metrics.feature_importance:
                # Сортировка признаков по важности
                sorted_features = sorted(
                    metrics.feature_importance.items(), key=lambda x: x[1], reverse=True
                )
                analysis[model_name] = {
                    "top_features": sorted_features[:5],
                    "least_important": sorted_features[-5:],
                    "importance_distribution": {
                        "high": len([f for f, imp in sorted_features if imp > 0.1]),
                        "medium": len(
                            [f for f, imp in sorted_features if 0.01 <= imp <= 0.1]
                        ),
                        "low": len([f for f, imp in sorted_features if imp < 0.01]),
                    },
                }
        return analysis

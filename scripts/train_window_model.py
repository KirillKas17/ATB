import asyncio
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
from shared.numpy_utils import np
import optuna
import pandas as pd
from loguru import logger
from ml.window_optimizer import WindowConfig, WindowSizeOptimizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

warnings.filterwarnings("ignore")

# Создание необходимых директорий
for dir_path in ["models", "datasets", "logs", "backups"]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Проверка наличия файла данных
data_path = Path("datasets/window_training_data.csv")
if not data_path.exists():
    logger.error(f"Файл {data_path} не найден")
    sys.exit(1)


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""

    test_size: float = 0.2
    random_state: int = 42
    n_trials: int = 100
    cv_splits: int = 5
    n_jobs: int = -1
    early_stopping_rounds: int = 10
    min_samples: int = 100
    max_samples: int = 10000
    metrics_window: int = 24
    model_dir: str = "models"
    data_dir: str = "datasets"
    log_dir: str = "logs"
    backup_dir: str = "backups"


@dataclass
class TrainingMetrics:
    """Метрики обучения"""

    mse: float
    rmse: float
    r2: float
    cv_scores: List[float]
    feature_importance: Dict[str, float]
    training_time: float
    samples_count: int
    last_update: datetime


class WindowModelTrainer:
    """Тренировка модели окна"""

    def __init__(self, config: Optional[TrainingConfig] = None):
        """Инициализация тренера"""
        self.config = config or TrainingConfig()
        self._setup_directories()
        self.metrics_history: List[TrainingMetrics] = []
        self._model_lock = asyncio.Lock()

    def _setup_directories(self):
        """Создание необходимых директорий"""
        for dir_path in [
            self.config.model_dir,
            self.config.data_dir,
            self.config.log_dir,
            self.config.backup_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    async def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Загрузка данных для обучения"""
        try:
            data_path = Path(f"{self.config.data_dir}/window_training_data.csv")
            if not data_path.exists():
                raise FileNotFoundError(f"Файл {data_path} не найден")

            # Загрузка данных
            df = pd.read_csv(data_path)

            # Проверка наличия колонок
            required_columns = [
                "volatility",
                "trend_strength",
                "regime_encoded",
                "atr",
                "adx",
                "rsi",
                "bollinger_width",
                "optimal_window",
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Отсутствуют колонки: {missing_columns}")

            # Обработка пропущенных значений
            df = self._handle_missing_values(df)

            # Ограничение размера выборки
            if len(df) > self.config.max_samples:
                # Явно приводим к DataFrame для mypy
                df_dataframe = pd.DataFrame(df)
                # Проверяем, что df_dataframe не является callable перед индексированием
                if callable(df_dataframe):
                    df_dataframe = df_dataframe()
                if hasattr(df_dataframe, 'iloc'):
                    df = df_dataframe.iloc[:self.config.max_samples]  # type: ignore
                else:
                    # Альтернативный способ обрезки данных
                    df = df_dataframe.head(self.config.max_samples)  # type: ignore

            X = df.drop(columns=["optimal_window"])
            y = df["optimal_window"]

            logger.info(f"Загружено {len(df)} записей для обучения")
            return X, y

        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {str(e)}")
            raise

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка пропущенных значений"""
        # Заполнение пропусков медианой для числовых колонок
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_columns].fillna(df[numeric_columns].median())
        df = df.copy()
        df[numeric_columns] = df_numeric

        # Заполнение пропусков модой для категориальных колонок
        categorical_columns = df.select_dtypes(include=["object"]).columns
        if len(categorical_columns) > 0:
            mode_values = df[categorical_columns].mode()
            if not mode_values.empty:
                df_categorical = df[categorical_columns].fillna(mode_values.iloc[0])
            else:
                df_categorical = df[categorical_columns].fillna("unknown")
            df[categorical_columns] = df_categorical

        return df

    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Оптимизация гиперпараметров"""
        try:

            def objective(trial):
                # Параметры для оптимизации
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                }

                # Создание модели
                model = WindowSizeOptimizer(WindowConfig(**params))

                # Кросс-валидация
                scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=self.config.cv_splits,
                    scoring="neg_mean_squared_error",
                    n_jobs=self.config.n_jobs,
                )

                return -np.mean(scores)  # Отрицательный MSE для максимизации

            # Оптимизация
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=self.config.n_trials)

            return study.best_params

        except Exception as e:
            logger.error(f"Ошибка оптимизации гиперпараметров: {str(e)}")
            return {}

    async def train_model(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[WindowSizeOptimizer, TrainingMetrics]:
        """Обучение модели"""
        try:
            start_time = datetime.now()

            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )

            # Оптимизация гиперпараметров
            best_params = self._optimize_hyperparameters(X_train, y_train)

            # Создание и обучение модели
            model = WindowSizeOptimizer(WindowConfig(**best_params))
            model.fit(X_train, y_train)

            # Оценка модели
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Кросс-валидация
            cv_scores = cross_val_score(
                model,
                X,
                y,
                cv=self.config.cv_splits,
                scoring="neg_mean_squared_error",
                n_jobs=self.config.n_jobs,
            )

            # Получение важности признаков
            feature_importance = model.get_feature_importance()

            # Расчет времени обучения
            training_time = (datetime.now() - start_time).total_seconds()

            # Создание метрик
            metrics = TrainingMetrics(
                mse=float(mse),
                rmse=float(rmse),
                r2=float(r2),
                cv_scores=cv_scores.tolist(),
                feature_importance=feature_importance,
                training_time=training_time,
                samples_count=len(X),
                last_update=datetime.now(),
            )

            return model, metrics

        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {str(e)}")
            raise

    async def save_model(self, model: WindowSizeOptimizer, metrics: TrainingMetrics):
        """Сохранение модели и метрик"""
        try:
            async with self._model_lock:
                # Сохранение модели
                model_path = Path(f"{self.config.model_dir}/window_model.joblib")
                joblib.dump(model, model_path)

                # Сохранение метрик
                metrics_path = Path(f"{self.config.model_dir}/window_metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(metrics.__dict__, f, indent=2, default=str)

                # Создание бэкапа
                backup_dir = Path(
                    f"{self.config.backup_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                backup_dir.mkdir(parents=True, exist_ok=True)

                joblib.dump(model, backup_dir / "window_model.joblib")
                with open(backup_dir / "window_metrics.json", "w") as f:
                    json.dump(metrics.__dict__, f, indent=2, default=str)

                logger.info(f"Модель и метрики сохранены в {self.config.model_dir}")

        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {str(e)}")
            raise


async def main():
    """Основная функция"""
    try:
        # Инициализация тренера
        trainer = WindowModelTrainer()

        # Загрузка данных
        X, y = await trainer.load_training_data()

        # Обучение модели
        model, metrics = await trainer.train_model(X, y)

        # Сохранение модели
        await trainer.save_model(model, metrics)

        logger.info("✅ Модель успешно обучена и сохранена")

    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}")
        raise


if __name__ == "__main__":
    # Настройка логирования
    logger.add(
        "logs/train_window_model_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
    )

    # Запуск асинхронного main
    asyncio.run(main())

"""
Интеграция с машинным обучением для анализа рынка.
"""

import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from shared.numpy_utils import np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def generate_features(market_data: Dict[str, Any]) -> Dict[str, float]:
    """Генерация признаков из рыночных данных."""
    return {}


@dataclass
class MLConfig:
    """Конфигурация ML интеграции"""

    # Параметры модели
    model_type: str = "random_forest"  # Тип модели
    n_estimators: int = 100  # Количество деревьев
    max_depth: int = 10  # Максимальная глубина
    min_samples_split: int = 2  # Минимальное количество образцов для разделения
    min_samples_leaf: int = 1  # Минимальное количество образцов в листе
    # Параметры обучения
    train_size: float = 0.8  # Размер обучающей выборки
    test_size: float = 0.2  # Размер тестовой выборки
    random_state: int = 42  # Seed для воспроизводимости
    # Параметры признаков
    feature_window: int = 20  # Окно для расчета признаков
    target_window: int = 5  # Окно для расчета целевой переменной
    min_samples: int = 100  # Минимальное количество образцов для обучения
    # Параметры индикаторов
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14
    # Параметры логирования
    log_dir: str = "logs"


class MLIntegration:
    """Интеграция с машинным обучением."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация ML интеграции.
        Args:
            config: Конфигурация
        """
        self.config = MLConfig(**(config or {}))
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Настройка логирования."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        log_file = os.path.join(self.config.log_dir, "ml_integration.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def load_model(self, model_name: str, model_path: str) -> None:
        """
        Загрузка модели.
        Args:
            model_name: Имя модели
            model_path: Путь к модели
        """
        try:
            # Загружаем модель
            with open(model_path, "rb") as f:
                self.models[model_name] = pickle.load(f)
            # Загружаем scaler, если есть
            scaler_path = f"{model_path}_scaler.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    self.scalers[model_name] = pickle.load(f)
            logger.info(f"Model {model_name} loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise

    def predict(self, model_name: str, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Предсказание с помощью модели.
        Args:
            model_name: Имя модели
            features: Признаки для предсказания
        Returns:
            Dict[str, Any]: Результат предсказания
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not loaded")
            # Подготавливаем признаки
            X = self._prepare_features(features)
            # Масштабируем признаки, если есть scaler
            if model_name in self.scalers:
                X = self.scalers[model_name].transform(X)
            # Делаем предсказание
            model = self.models[model_name]
            prediction = model.predict(X)
            probability = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            # Рассчитываем уверенность
            confidence = 0.8  # Заглушка
            return {
                "prediction": (
                    float(prediction[0])
                    if hasattr(prediction, "__len__") and len(prediction) > 0
                    else float(prediction)
                ),
                "probability": (
                    probability.tolist() if probability is not None else None
                ),
                "confidence": confidence,
                "timestamp": datetime.now(),
            }
        except Exception as e:
            logger.error(f"Error making prediction with {model_name}: {str(e)}")
            raise

    def update_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Обновление признаков.
        Args:
            market_data: Рыночные данные
        Returns:
            Dict[str, float]: Обновленные признаки
        """
        try:
            features = generate_features(market_data)
            # Добавляем технические индикаторы
            features.update(self._calculate_technical_indicators(market_data))
            return features
        except Exception as e:
            logger.error(f"Error updating features: {str(e)}")
            raise

    def _calculate_technical_indicators(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Расчет технических индикаторов.
        Args:
            data: Рыночные данные
        Returns:
            Dict[str, float]: Значения индикаторов
        """
        try:
            # Преобразуем данные в DataFrame
            df = pd.DataFrame([data])
            # RSI
            delta: pd.Series[float] = df["close"].diff()
            gain = (
                (delta.where(delta > 0, 0))
                .rolling(window=self.config.rsi_period)
                .mean()
            )
            loss = (
                (-delta.where(delta < 0, 0))
                .rolling(window=self.config.rsi_period)
                .mean()
            )
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            # MACD
            exp1 = df["close"].ewm(span=self.config.macd_fast, adjust=False).mean()
            exp2 = df["close"].ewm(span=self.config.macd_slow, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=self.config.macd_signal, adjust=False).mean()
            hist = macd - signal
            # Bollinger Bands
            ma = df["close"].rolling(window=self.config.bollinger_period).mean()
            std = df["close"].rolling(window=self.config.bollinger_period).std()
            upper_band = ma + (std * self.config.bollinger_std)
            lower_band = ma - (std * self.config.bollinger_std)
            # ATR
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())
            ranges = pd.DataFrame(
                {"high_low": high_low, "high_close": high_close, "low_close": low_close}
            )
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=self.config.atr_period).mean()
            # Получаем последние значения
            rsi_result = self._iloc_series(rsi, -1)
            macd_result = self._iloc_series(macd, -1)
            signal_result = self._iloc_series(signal, -1)
            hist_result = self._iloc_series(hist, -1)
            upper_band_result = self._iloc_series(upper_band, -1)
            lower_band_result = self._iloc_series(lower_band, -1)
            atr_result = self._iloc_series(atr, -1)
            
            return {
                "rsi": rsi_result,
                "macd": macd_result,
                "macd_signal": signal_result,
                "macd_hist": hist_result,
                "bollinger_upper": upper_band_result,
                "bollinger_lower": lower_band_result,
                "atr": atr_result,
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return {}

    def _iloc_series(self, series: pd.Series, index: int) -> float:
        """Безопасное получение значения из Series."""
        try:
            # Проверяем, является ли series pandas Series
            if not isinstance(series, pd.Series):
                if hasattr(series, 'iloc'):
                    # Если у объекта есть iloc, используем его
                    return float(series.iloc[index])
                else:
                    # Иначе пытаемся преобразовать в Series
                    series = pd.Series(series)
            
            # Получаем значение по индексу
            if isinstance(series, pd.Series) and len(series) > 0 and index < len(series):
                return float(series.iloc[index])
            else:
                return 0.0
        except Exception:
            return 0.0

    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Подготовка признаков для модели.
        Args:
            features: Словарь с признаками
        Returns:
            np.ndarray: Массив признаков
        """
        try:
            # Преобразуем словарь в массив
            feature_names = sorted(features.keys())
            X = np.array([[features[name] for name in feature_names]])
            return X
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """
        Получение важности признаков.
        Args:
            model_name: Имя модели
        Returns:
            Dict[str, float]: Важность признаков
        """
        try:
            if model_name not in self.feature_importance:
                return {}
            return self.feature_importance[model_name]
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise

    def save_model(self, model_name: str, path: str) -> None:
        """
        Сохранение модели.
        Args:
            model_name: Имя модели
            path: Путь для сохранения
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not loaded")
            # Создаем директорию, если не существует
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Сохраняем модель
            with open(path, "wb") as f:
                pickle.dump(self.models[model_name], f)
            # Сохраняем scaler, если есть
            if model_name in self.scalers:
                scaler_path = f"{path}_scaler.pkl"
                with open(scaler_path, "wb") as f:
                    pickle.dump(self.scalers[model_name], f)
            logger.info(f"Model {model_name} saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
            raise

    def train(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Обучение модели.
        Args:
            model_name: Имя модели
            X: Признаки
            y: Целевая переменная
        """
        try:
            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                train_size=self.config.train_size,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
            # Масштабирование признаков
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            scaler.transform(X_test)
            # Создание и обучение модели
            if self.config.model_type == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    min_samples_split=self.config.min_samples_split,
                    min_samples_leaf=self.config.min_samples_leaf,
                    random_state=self.config.random_state,
                )
                model.fit(X_train_scaled, y_train)
                # Сохранение модели и скейлера
                self.models[model_name] = model
                self.scalers[model_name] = scaler
                # Сохранение важности признаков
                self.feature_importance[model_name] = dict(
                    zip(X.columns, model.feature_importances_)
                )
            else:
                raise ValueError(f"Unknown model type: {self.config.model_type}")
        except Exception as e:
            logger.error(f"Error training model {model_name}: {str(e)}")
            raise

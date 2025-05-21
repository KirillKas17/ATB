from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .base_strategy import BaseStrategy, Signal, StrategyMetrics


@dataclass
class RandomForestConfig:
    """Конфигурация стратегии случайного леса"""

    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    bootstrap: bool = True
    random_state: int = 42
    n_jobs: int = -1
    lookback_period: int = 20
    prediction_threshold: float = 0.7
    min_samples_for_training: int = 100
    retrain_interval: int = 100
    feature_columns: List[str] = field(
        default_factory=lambda: [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr",
            "adx",
            "plus_di",
            "minus_di",
        ]
    )
    target_column: str = "target"
    risk_per_trade: float = 0.02
    max_position_size: float = 0.1
    trailing_stop: bool = True
    trailing_step: float = 0.002
    partial_close: bool = True
    partial_close_levels: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    partial_close_sizes: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.4])
    max_trades: Optional[int] = None
    min_volume: float = 0.0
    min_volatility: float = 0.0
    max_spread: float = 0.01
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    log_dir: str = "logs"


class RandomForestStrategy(BaseStrategy):
    """Стратегия торговли на основе случайного леса (расширенная)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация стратегии.

        Args:
            config: Словарь с параметрами стратегии
        """
        super().__init__(config)
        self.config = (
            RandomForestConfig(**config)
            if config and not isinstance(config, RandomForestConfig)
            else (config or RandomForestConfig())
        )
        self.model = None
        self.scaler = StandardScaler()
        self.position = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.partial_closes = []
        self.last_retrain = 0
        self._setup_logger()

    def _setup_logger(self):
        """Настройка логгера"""
        logger.add(
            f"{self.config.log_dir}/random_forest_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ рыночных данных.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с результатами анализа
        """
        try:
            is_valid, error = self.validate_data(data)
            if not is_valid:
                raise ValueError(f"Invalid data: {error}")

            # Подготовка данных
            features = self._prepare_features(data)

            # Обучение модели при необходимости
            if self._should_retrain(data):
                self._train_model(data)

            # Получение предсказания
            prediction = self._get_prediction(features)

            # Расчет метрик
            risk_metrics = self.calculate_risk_metrics(data)

            return {
                "prediction": prediction,
                "features": features,
                "risk_metrics": risk_metrics,
                "timestamp": data.index[-1],
            }
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            return {}

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Генерация торгового сигнала.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            analysis = self.analyze(data)
            if not analysis:
                return None

            prediction = analysis["prediction"]
            features = analysis["features"]

            # Проверяем базовые условия
            if not self._check_basic_conditions(data, features):
                return None

            # Генерируем сигнал
            signal = self._generate_trading_signal(data, prediction, features)
            if signal:
                self._update_position_state(signal, data)

            return signal

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None

    def _check_basic_conditions(self, data: pd.DataFrame, features: pd.DataFrame) -> bool:
        """
        Проверка базовых условий для торговли.

        Args:
            data: DataFrame с OHLCV данными
            features: DataFrame с признаками

        Returns:
            bool: Результат проверки
        """
        try:
            # Проверка объема
            if data["volume"].iloc[-1] < self.config.min_volume:
                return False

            # Проверка волатильности
            if features["atr"].iloc[-1] < self.config.min_volatility:
                return False

            # Проверка спреда
            spread = (data["high"].iloc[-1] - data["low"].iloc[-1]) / data["close"].iloc[-1]
            if spread > self.config.max_spread:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking basic conditions: {str(e)}")
            return False

    def _generate_trading_signal(
        self, data: pd.DataFrame, prediction: Dict[str, float], features: pd.DataFrame
    ) -> Optional[Signal]:
        """
        Генерация торгового сигнала.

        Args:
            data: DataFrame с OHLCV данными
            prediction: Предсказание модели
            features: DataFrame с признаками

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            if self.position is None:
                return self._generate_entry_signal(data, prediction, features)
            else:
                return self._generate_exit_signal(data, prediction, features)

        except Exception as e:
            logger.error(f"Error generating trading signal: {str(e)}")
            return None

    def _generate_entry_signal(
        self, data: pd.DataFrame, prediction: Dict[str, float], features: pd.DataFrame
    ) -> Optional[Signal]:
        """
        Генерация сигнала на вход в позицию.

        Args:
            data: DataFrame с OHLCV данными
            prediction: Предсказание модели
            features: DataFrame с признаками

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]

            # Проверяем условия для длинной позиции
            if (
                prediction["probability"] > self.config.prediction_threshold
                and prediction["direction"] == "up"
            ):

                # Рассчитываем размер позиции
                volume = self._calculate_position_size(current_price, features["atr"].iloc[-1])

                # Устанавливаем стоп-лосс и тейк-профит
                stop_loss = current_price - features["atr"].iloc[-1] * 2
                take_profit = current_price + features["atr"].iloc[-1] * 3

                return Signal(
                    direction="long",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    confidence=prediction["probability"],
                    timestamp=data.index[-1],
                    metadata={"prediction": prediction, "features": features.iloc[-1].to_dict()},
                )

            # Проверяем условия для короткой позиции
            elif (
                prediction["probability"] > self.config.prediction_threshold
                and prediction["direction"] == "down"
            ):

                # Рассчитываем размер позиции
                volume = self._calculate_position_size(current_price, features["atr"].iloc[-1])

                # Устанавливаем стоп-лосс и тейк-профит
                stop_loss = current_price + features["atr"].iloc[-1] * 2
                take_profit = current_price - features["atr"].iloc[-1] * 3

                return Signal(
                    direction="short",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    confidence=prediction["probability"],
                    timestamp=data.index[-1],
                    metadata={"prediction": prediction, "features": features.iloc[-1].to_dict()},
                )

            return None

        except Exception as e:
            logger.error(f"Error generating entry signal: {str(e)}")
            return None

    def _generate_exit_signal(
        self, data: pd.DataFrame, prediction: Dict[str, float], features: pd.DataFrame
    ) -> Optional[Signal]:
        """
        Генерация сигнала на выход из позиции.

        Args:
            data: DataFrame с OHLCV данными
            prediction: Предсказание модели
            features: DataFrame с признаками

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]

            # Проверяем стоп-лосс и тейк-профит
            if self.position == "long":
                if current_price <= self.stop_loss or current_price >= self.take_profit:
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "stop_loss_or_take_profit",
                            "prediction": prediction,
                            "features": features.iloc[-1].to_dict(),
                        },
                    )

            elif self.position == "short":
                if current_price >= self.stop_loss or current_price <= self.take_profit:
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "stop_loss_or_take_profit",
                            "prediction": prediction,
                            "features": features.iloc[-1].to_dict(),
                        },
                    )

            # Проверяем трейлинг-стоп
            if self.config.trailing_stop and self.trailing_stop:
                if self.position == "long" and current_price > self.trailing_stop:
                    self.trailing_stop = (
                        current_price - features["atr"].iloc[-1] * self.config.trailing_step
                    )
                elif self.position == "short" and current_price < self.trailing_stop:
                    self.trailing_stop = (
                        current_price + features["atr"].iloc[-1] * self.config.trailing_step
                    )

                if (self.position == "long" and current_price <= self.trailing_stop) or (
                    self.position == "short" and current_price >= self.trailing_stop
                ):
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "trailing_stop",
                            "prediction": prediction,
                            "features": features.iloc[-1].to_dict(),
                        },
                    )

            # Проверяем частичное закрытие
            if self.config.partial_close and self.partial_closes:
                for level, size in zip(
                    self.config.partial_close_levels, self.config.partial_close_sizes
                ):
                    if self.position == "long" and current_price >= self.take_profit * level:
                        return Signal(
                            direction="partial_close",
                            entry_price=current_price,
                            volume=size,
                            timestamp=data.index[-1],
                            confidence=1.0,
                            metadata={
                                "reason": "partial_close",
                                "level": level,
                                "size": size,
                                "prediction": prediction,
                                "features": features.iloc[-1].to_dict(),
                            },
                        )
                    elif self.position == "short" and current_price <= self.take_profit * level:
                        return Signal(
                            direction="partial_close",
                            entry_price=current_price,
                            volume=size,
                            timestamp=data.index[-1],
                            confidence=1.0,
                            metadata={
                                "reason": "partial_close",
                                "level": level,
                                "size": size,
                                "prediction": prediction,
                                "features": features.iloc[-1].to_dict(),
                            },
                        )

            return None

        except Exception as e:
            logger.error(f"Error generating exit signal: {str(e)}")
            return None

    def _update_position_state(self, signal: Signal, data: pd.DataFrame):
        """
        Обновление состояния позиции.

        Args:
            signal: Торговый сигнал
            data: DataFrame с OHLCV данными
        """
        try:
            if signal.direction in ["long", "short"]:
                self.position = signal.direction
                self.stop_loss = signal.stop_loss
                self.take_profit = signal.take_profit
                self.trailing_stop = signal.entry_price
                self.partial_closes = []
            elif signal.direction == "close":
                self.position = None
                self.stop_loss = None
                self.take_profit = None
                self.trailing_stop = None
                self.partial_closes = []
            elif signal.direction == "partial_close":
                self.partial_closes.append(signal.volume)

        except Exception as e:
            logger.error(f"Error updating position state: {str(e)}")

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка признаков для модели.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            DataFrame с признаками
        """
        try:
            features = pd.DataFrame(index=data.index)

            # Базовые признаки
            features["open"] = data["open"]
            features["high"] = data["high"]
            features["low"] = data["low"]
            features["close"] = data["close"]
            features["volume"] = data["volume"]

            # RSI
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features["rsi"] = 100 - (100 / (1 + rs))

            # MACD
            ema_fast = data["close"].ewm(span=12).mean()
            ema_slow = data["close"].ewm(span=26).mean()
            features["macd"] = ema_fast - ema_slow
            features["macd_signal"] = features["macd"].ewm(span=9).mean()

            # Bollinger Bands
            bb_middle = data["close"].rolling(window=20).mean()
            bb_std = data["close"].rolling(window=20).std()
            features["bb_upper"] = bb_middle + (bb_std * 2)
            features["bb_middle"] = bb_middle
            features["bb_lower"] = bb_middle - (bb_std * 2)

            # ATR
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            features["atr"] = true_range.rolling(window=14).mean()

            # ADX
            plus_dm = data["high"].diff()
            minus_dm = data["low"].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            tr = true_range
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            features["adx"] = dx.rolling(window=14).mean()
            features["plus_di"] = plus_di
            features["minus_di"] = minus_di

            # Целевая переменная (направление движения цены)
            features[self.config.target_column] = np.where(
                data["close"].shift(-1) > data["close"], 1, 0  # Вверх  # Вниз
            )

            return features

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()

    def _should_retrain(self, data: pd.DataFrame) -> bool:
        """
        Проверка необходимости переобучения модели.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            bool: Нужно ли переобучать модель
        """
        try:
            if self.model is None:
                return True

            if len(data) - self.last_retrain >= self.config.retrain_interval:
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking retrain condition: {str(e)}")
            return False

    def _train_model(self, data: pd.DataFrame):
        """
        Обучение модели.

        Args:
            data: DataFrame с OHLCV данными
        """
        try:
            # Подготовка данных
            features = self._prepare_features(data)

            # Проверка достаточности данных
            if len(features) < self.config.min_samples_for_training:
                logger.warning(
                    f"Not enough data for training: {len(features)} < {self.config.min_samples_for_training}"
                )
                return

            # Разделение на признаки и целевую переменную
            X = features[self.config.feature_columns]
            y = features[self.config.target_column]

            # Нормализация признаков
            X_scaled = self.scaler.fit_transform(X)

            # Разделение на обучающую и тестовую выборки
            tscv = TimeSeriesSplit(n_splits=5)
            for train_index, test_index in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Обучение модели
                self.model = RandomForestClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    min_samples_split=self.config.min_samples_split,
                    min_samples_leaf=self.config.min_samples_leaf,
                    max_features=self.config.max_features,
                    bootstrap=self.config.bootstrap,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                )
                self.model.fit(X_train, y_train)

                # Оценка качества
                y_pred = self.model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                logger.info(
                    f"Model metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
                )

            self.last_retrain = len(data)

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")

    def _get_prediction(self, features: pd.DataFrame) -> Dict[str, float]:
        """
        Получение предсказания от модели.

        Args:
            features: DataFrame с признаками

        Returns:
            Dict с предсказанием
        """
        try:
            if self.model is None:
                return {"direction": None, "probability": 0.0}

            # Подготовка данных
            X = features[self.config.feature_columns].iloc[-1:].values
            X_scaled = self.scaler.transform(X)

            # Получение предсказания
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0][1]

            return {"direction": "up" if prediction == 1 else "down", "probability": probability}

        except Exception as e:
            logger.error(f"Error getting prediction: {str(e)}")
            return {"direction": None, "probability": 0.0}

    def _calculate_position_size(self, price: float, atr: float) -> float:
        """
        Расчет размера позиции.

        Args:
            price: Текущая цена
            atr: ATR

        Returns:
            float: Размер позиции
        """
        try:
            # Базовый размер позиции
            base_size = self.config.risk_per_trade

            # Корректировка на волатильность
            volatility_factor = 1 / (1 + atr / price)

            # Корректировка на максимальный размер
            position_size = base_size * volatility_factor
            position_size = min(position_size, self.config.max_position_size)

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

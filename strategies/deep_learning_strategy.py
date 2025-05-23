from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from .base_strategy import BaseStrategy, Signal


@dataclass
class DeepLearningConfig:
    """Конфигурация стратегии глубокого обучения"""

    # Параметры модели
    lstm_units: List[int] = field(
        default_factory=lambda: [64, 32]
    )  # Количество нейронов в LSTM слоях
    dense_units: List[int] = field(
        default_factory=lambda: [32, 16]
    )  # Количество нейронов в полносвязных слоях
    dropout_rate: float = 0.2  # Коэффициент прореживания
    learning_rate: float = 0.001  # Скорость обучения
    batch_size: int = 32  # Размер батча
    epochs: int = 100  # Количество эпох
    validation_split: float = 0.2  # Доля данных для валидации

    # Параметры обучения
    sequence_length: int = 60  # Длина последовательности
    prediction_horizon: int = 5  # Горизонт предсказания
    min_samples: int = 1000  # Минимальное количество образцов для обучения

    # Параметры индикаторов
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14

    # Параметры управления рисками
    risk_per_trade: float = 0.02
    max_position_size: float = 0.2
    trailing_stop: bool = True
    trailing_step: float = 0.002
    partial_close: bool = True
    partial_close_levels: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    partial_close_sizes: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.4])

    # Общие параметры
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    log_dir: str = "logs"


class DeepLearningStrategy(BaseStrategy):
    """Стратегия глубокого обучения (расширенная)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация стратегии.

        Args:
            config: Словарь с параметрами стратегии
        """
        super().__init__(config)
        self.config = (
            DeepLearningConfig(**config)
            if config and not isinstance(config, DeepLearningConfig)
            else (config or DeepLearningConfig())
        )
        self.model = None
        self.scaler = StandardScaler()
        self.position = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.partial_closes = []
        self._setup_logger()

    def _setup_logger(self):
        """Настройка логгера"""
        logger.add(
            f"{self.config.log_dir}/deep_learning_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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

            # Расчет признаков
            features = self._calculate_features(data)

            # Расчет индикаторов
            indicators = self._calculate_indicators(data)

            # Анализ состояния рынка
            market_state = self._analyze_market_state(data, indicators)

            # Расчет метрик риска
            risk_metrics = self.calculate_risk_metrics(data)

            return {
                "features": features,
                "indicators": indicators,
                "market_state": market_state,
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

            features = analysis["features"]
            indicators = analysis["indicators"]
            market_state = analysis["market_state"]

            # Проверяем базовые условия
            if not self._check_basic_conditions(data, features, indicators):
                return None

            # Генерируем сигнал
            signal = self._generate_trading_signal(
                data, features, indicators, market_state
            )
            if signal:
                self._update_position_state(signal, data)

            return signal

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None

    def _check_basic_conditions(
        self,
        data: pd.DataFrame,
        features: Dict[str, float],
        indicators: Dict[str, float],
    ) -> bool:
        """
        Проверка базовых условий для торговли.

        Args:
            data: DataFrame с OHLCV данными
            features: Признаки
            indicators: Значения индикаторов

        Returns:
            bool: Результат проверки
        """
        try:
            # Проверка наличия модели
            if self.model is None:
                return False

            # Проверка достаточности данных
            if len(data) < self.config.min_samples:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking basic conditions: {str(e)}")
            return False

    def _calculate_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Расчет признаков для модели.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с признаками
        """
        try:
            # Ценовые признаки
            returns = data["close"].pct_change()
            log_returns = np.log(data["close"] / data["close"].shift(1))
            volatility = returns.rolling(window=self.config.sequence_length).std()

            # Объемные признаки
            volume_ma = (
                data["volume"].rolling(window=self.config.sequence_length).mean()
            )
            volume_std = (
                data["volume"].rolling(window=self.config.sequence_length).std()
            )
            volume_ratio = data["volume"] / volume_ma

            # Технические индикаторы
            rsi = self._calculate_rsi(data)
            macd, signal, hist = self._calculate_macd(data)
            upper_band, lower_band = self._calculate_bollinger_bands(data)
            atr = self._calculate_atr(data)

            # Моментум признаки
            momentum = (
                data["close"] / data["close"].shift(self.config.sequence_length) - 1
            )
            roc = (
                data["close"] - data["close"].shift(self.config.sequence_length)
            ) / data["close"].shift(self.config.sequence_length)

            # Волатильностные признаки
            high_low_ratio = data["high"] / data["low"]
            close_open_ratio = data["close"] / data["open"]

            return {
                "returns": returns.iloc[-1],
                "log_returns": log_returns.iloc[-1],
                "volatility": volatility.iloc[-1],
                "volume_ma": volume_ma.iloc[-1],
                "volume_std": volume_std.iloc[-1],
                "volume_ratio": volume_ratio.iloc[-1],
                "rsi": rsi.iloc[-1],
                "macd": macd.iloc[-1],
                "macd_signal": signal.iloc[-1],
                "macd_hist": hist.iloc[-1],
                "bollinger_upper": upper_band.iloc[-1],
                "bollinger_lower": lower_band.iloc[-1],
                "atr": atr.iloc[-1],
                "momentum": momentum.iloc[-1],
                "roc": roc.iloc[-1],
                "high_low_ratio": high_low_ratio.iloc[-1],
                "close_open_ratio": close_open_ratio.iloc[-1],
            }

        except Exception as e:
            logger.error(f"Error calculating features: {str(e)}")
            return {}

    def _calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """
        Расчет RSI.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            pd.Series с RSI
        """
        try:
            delta = data["close"].diff()
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
            return 100 - (100 / (1 + rs))

        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series()

    def _calculate_macd(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Расчет MACD.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Tuple с MACD, сигнальной линией и гистограммой
        """
        try:
            exp1 = data["close"].ewm(span=self.config.macd_fast, adjust=False).mean()
            exp2 = data["close"].ewm(span=self.config.macd_slow, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=self.config.macd_signal, adjust=False).mean()
            hist = macd - signal
            return macd, signal, hist

        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return pd.Series(), pd.Series(), pd.Series()

    def _calculate_bollinger_bands(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Расчет полос Боллинджера.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Tuple с верхней и нижней полосами
        """
        try:
            sma = data["close"].rolling(window=self.config.bollinger_period).mean()
            std = data["close"].rolling(window=self.config.bollinger_period).std()
            upper_band = sma + (std * self.config.bollinger_std)
            lower_band = sma - (std * self.config.bollinger_std)
            return upper_band, lower_band

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return pd.Series(), pd.Series()

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        Расчет ATR.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            pd.Series с ATR
        """
        try:
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            return true_range.rolling(window=self.config.atr_period).mean()

        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series()

    def _generate_trading_signal(
        self,
        data: pd.DataFrame,
        features: Dict[str, float],
        indicators: Dict[str, float],
        market_state: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерация торгового сигнала.

        Args:
            data: DataFrame с OHLCV данными
            features: Признаки
            indicators: Значения индикаторов
            market_state: Состояние рынка

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            data["close"].iloc[-1]

            if self.position is None:
                return self._generate_entry_signal(
                    data, features, indicators, market_state
                )
            else:
                return self._generate_exit_signal(
                    data, features, indicators, market_state
                )

        except Exception as e:
            logger.error(f"Error generating trading signal: {str(e)}")
            return None

    def _generate_entry_signal(
        self,
        data: pd.DataFrame,
        features: Dict[str, float],
        indicators: Dict[str, float],
        market_state: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на вход в позицию.

        Args:
            data: DataFrame с OHLCV данными
            features: Признаки
            indicators: Значения индикаторов
            market_state: Состояние рынка

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]

            # Подготовка последовательности для модели
            sequence = self._prepare_sequence(data)

            # Получение предсказания модели
            prediction = self.model.predict(sequence)[0]

            # Проверяем условия для длинной позиции
            if prediction[0] > 0.7:  # Вероятность роста > 70%
                # Рассчитываем размер позиции
                volume = self._calculate_position_size(current_price, indicators["atr"])

                # Устанавливаем стоп-лосс и тейк-профит
                stop_loss = current_price - indicators["atr"] * 2
                take_profit = current_price + (current_price - stop_loss) * 2

                return Signal(
                    direction="long",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    confidence=prediction[0],
                    timestamp=data.index[-1],
                    metadata={
                        "features": features,
                        "indicators": indicators,
                        "market_state": market_state,
                        "prediction": prediction,
                    },
                )

            # Проверяем условия для короткой позиции
            elif prediction[1] > 0.7:  # Вероятность падения > 70%
                # Рассчитываем размер позиции
                volume = self._calculate_position_size(current_price, indicators["atr"])

                # Устанавливаем стоп-лосс и тейк-профит
                stop_loss = current_price + indicators["atr"] * 2
                take_profit = current_price - (stop_loss - current_price) * 2

                return Signal(
                    direction="short",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    confidence=prediction[1],
                    timestamp=data.index[-1],
                    metadata={
                        "features": features,
                        "indicators": indicators,
                        "market_state": market_state,
                        "prediction": prediction,
                    },
                )

            return None

        except Exception as e:
            logger.error(f"Error generating entry signal: {str(e)}")
            return None

    def _generate_exit_signal(
        self,
        data: pd.DataFrame,
        features: Dict[str, float],
        indicators: Dict[str, float],
        market_state: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на выход из позиции.

        Args:
            data: DataFrame с OHLCV данными
            features: Признаки
            indicators: Значения индикаторов
            market_state: Состояние рынка

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]

            # Проверяем стоп-лосс и тейк-профит
            if self.position == "long":
                if current_price <= self.stop_loss:
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "stop_loss",
                            "features": features,
                            "indicators": indicators,
                            "market_state": market_state,
                        },
                    )
                elif current_price >= self.take_profit:
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "take_profit",
                            "features": features,
                            "indicators": indicators,
                            "market_state": market_state,
                        },
                    )

            elif self.position == "short":
                if current_price >= self.stop_loss:
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "stop_loss",
                            "features": features,
                            "indicators": indicators,
                            "market_state": market_state,
                        },
                    )
                elif current_price <= self.take_profit:
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "take_profit",
                            "features": features,
                            "indicators": indicators,
                            "market_state": market_state,
                        },
                    )

            # Проверяем трейлинг-стоп
            if self.config.trailing_stop and self.trailing_stop:
                if self.position == "long" and current_price > self.trailing_stop:
                    self.trailing_stop = (
                        current_price - indicators["atr"] * self.config.trailing_step
                    )
                elif self.position == "short" and current_price < self.trailing_stop:
                    self.trailing_stop = (
                        current_price + indicators["atr"] * self.config.trailing_step
                    )

                if (
                    self.position == "long" and current_price <= self.trailing_stop
                ) or (self.position == "short" and current_price >= self.trailing_stop):
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "trailing_stop",
                            "features": features,
                            "indicators": indicators,
                            "market_state": market_state,
                        },
                    )

            # Проверяем частичное закрытие
            if self.config.partial_close and self.partial_closes:
                for level, size in zip(
                    self.config.partial_close_levels, self.config.partial_close_sizes
                ):
                    if (
                        self.position == "long"
                        and current_price >= self.take_profit * level
                    ):
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
                                "features": features,
                                "indicators": indicators,
                                "market_state": market_state,
                            },
                        )
                    elif (
                        self.position == "short"
                        and current_price <= self.take_profit * level
                    ):
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
                                "features": features,
                                "indicators": indicators,
                                "market_state": market_state,
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

    def _prepare_sequence(self, data: pd.DataFrame) -> np.ndarray:
        """
        Подготовка последовательности для модели.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            np.ndarray с подготовленной последовательностью
        """
        try:
            # Расчет признаков для всей последовательности
            features = []
            for i in range(len(data) - self.config.sequence_length + 1):
                window_data = data.iloc[i : i + self.config.sequence_length]
                window_features = self._calculate_features(window_data)
                features.append(list(window_features.values()))

            # Преобразование в массив
            X = np.array(features)

            # Масштабирование признаков
            X = self.scaler.transform(X)

            # Изменение формы для LSTM
            X = X.reshape((1, self.config.sequence_length, X.shape[1]))

            return X

        except Exception as e:
            logger.error(f"Error preparing sequence: {str(e)}")
            return np.array([])

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Расчет индикаторов.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с значениями индикаторов
        """
        try:
            # RSI
            rsi = self._calculate_rsi(data)

            # MACD
            macd, signal, hist = self._calculate_macd(data)

            # Bollinger Bands
            upper_band, lower_band = self._calculate_bollinger_bands(data)

            # ATR
            atr = self._calculate_atr(data)

            return {
                "rsi": rsi.iloc[-1],
                "macd": macd.iloc[-1],
                "macd_signal": signal.iloc[-1],
                "macd_hist": hist.iloc[-1],
                "bollinger_upper": upper_band.iloc[-1],
                "bollinger_lower": lower_band.iloc[-1],
                "atr": atr.iloc[-1],
            }

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return {}

    def _analyze_market_state(
        self, data: pd.DataFrame, indicators: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Анализ состояния рынка.

        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов

        Returns:
            Dict с состоянием рынка
        """
        try:
            # Направление тренда
            trend_direction = "up" if indicators["macd_hist"] > 0 else "down"

            # Волатильность
            volatility = (
                "high" if indicators["atr"] > data["close"].iloc[-1] * 0.01 else "low"
            )

            # Объем
            volume_state = (
                "high"
                if data["volume"].iloc[-1]
                > data["volume"].rolling(window=20).mean().iloc[-1]
                else "low"
            )

            return {
                "trend_direction": trend_direction,
                "volatility": volatility,
                "volume_state": volume_state,
            }

        except Exception as e:
            logger.error(f"Error analyzing market state: {str(e)}")
            return {}

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

    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Построение модели LSTM.

        Args:
            input_shape: Форма входных данных

        Returns:
            Sequential: Модель LSTM
        """
        try:
            model = Sequential()

            # LSTM слои
            for i, units in enumerate(self.config.lstm_units):
                if i == 0:
                    model.add(
                        LSTM(units, return_sequences=True, input_shape=input_shape)
                    )
                else:
                    model.add(LSTM(units, return_sequences=True))
                model.add(Dropout(self.config.dropout_rate))

            # Полносвязные слои
            for units in self.config.dense_units:
                model.add(Dense(units, activation="relu"))
                model.add(Dropout(self.config.dropout_rate))

            # Выходной слой
            model.add(Dense(2, activation="softmax"))

            # Компиляция модели
            model.compile(
                optimizer=Adam(learning_rate=self.config.learning_rate),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            return model

        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            return None

    def train(self, data: pd.DataFrame):
        """
        Обучение модели.

        Args:
            data: DataFrame с OHLCV данными
        """
        try:
            # Расчет признаков
            features = []
            for i in range(len(data) - self.config.sequence_length):
                window_data = data.iloc[i : i + self.config.sequence_length]
                window_features = self._calculate_features(window_data)
                features.append(list(window_features.values()))

            # Создание DataFrame с признаками
            features_df = pd.DataFrame(features)

            # Расчет целевой переменной
            future_returns = (
                data["close"]
                .pct_change(self.config.prediction_horizon)
                .shift(-self.config.prediction_horizon)
            )
            target = pd.get_dummies((future_returns > 0).astype(int))

            # Удаление NaN значений
            valid_idx = ~(features_df.isna().any(axis=1) | target.isna().any(axis=1))
            X = features_df[valid_idx]
            y = target[valid_idx]

            # Масштабирование признаков
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            # Изменение формы для LSTM
            X_scaled = X_scaled.reshape(
                (
                    X_scaled.shape[0],
                    self.config.sequence_length,
                    X_scaled.shape[1] // self.config.sequence_length,
                )
            )

            # Построение модели
            self.model = self._build_model(
                (self.config.sequence_length, X_scaled.shape[2])
            )

            if self.model is None:
                raise ValueError("Failed to build model")

            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                ),
                ModelCheckpoint(
                    f"{self.config.log_dir}/best_model.h5",
                    monitor="val_loss",
                    save_best_only=True,
                ),
            ]

            # Обучение модели
            history = self.model.fit(
                X_scaled,
                y,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=1,
            )

            logger.info(
                f"Model trained. Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}"
            )

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")

    def save_model(self, path: str):
        """
        Сохранение модели.

        Args:
            path: Путь для сохранения
        """
        try:
            if self.model is not None:
                self.model.save(path)
                logger.info(f"Model saved to {path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self, path: str):
        """
        Загрузка модели.

        Args:
            path: Путь к файлу модели
        """
        try:
            self.model = load_model(path)
            logger.info(f"Model loaded from {path}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

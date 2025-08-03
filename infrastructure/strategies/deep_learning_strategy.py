from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

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

    def __init__(
        self, config: Optional[Union[Dict[str, Any], DeepLearningConfig]] = None
    ):
        """
        Инициализация стратегии.
        Args:
            config: Словарь с параметрами стратегии или объект конфигурации
        """
        # Преобразуем конфигурацию в словарь для базового класса
        if isinstance(config, DeepLearningConfig):
            config_dict = {
                "lstm_units": config.lstm_units,
                "dense_units": config.dense_units,
                "dropout_rate": config.dropout_rate,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "validation_split": config.validation_split,
                "sequence_length": config.sequence_length,
                "prediction_horizon": config.prediction_horizon,
                "min_samples": config.min_samples,
                "rsi_period": config.rsi_period,
                "macd_fast": config.macd_fast,
                "macd_slow": config.macd_slow,
                "macd_signal": config.macd_signal,
                "bollinger_period": config.bollinger_period,
                "bollinger_std": config.bollinger_std,
                "atr_period": config.atr_period,
                "risk_per_trade": config.risk_per_trade,
                "max_position_size": config.max_position_size,
                "trailing_stop": config.trailing_stop,
                "trailing_step": config.trailing_step,
                "partial_close": config.partial_close,
                "partial_close_levels": config.partial_close_levels,
                "partial_close_sizes": config.partial_close_sizes,
                "symbols": config.symbols,
                "timeframes": config.timeframes,
                "log_dir": config.log_dir,
            }
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {}
        
        super().__init__(config_dict)
        
        # Устанавливаем конфигурацию
        if isinstance(config, DeepLearningConfig):
            self._config = config
        elif isinstance(config, dict):
            self._config = DeepLearningConfig(**config)
        else:
            self._config = DeepLearningConfig()
            
        self.model: Optional[Sequential] = None
        self.scaler = StandardScaler()
        self.position: Optional[str] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.trailing_stop: Optional[float] = None
        self.partial_closes: List[Dict[str, Any]] = []
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Настройка логгера"""
        logger.add(
            f"{self._config.log_dir}/deep_learning_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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
            if len(data) < self._config.min_samples:
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
            volatility = returns.rolling(window=self._config.sequence_length).std()
            # Объемные признаки
            volume_ma = (
                data["volume"].rolling(window=self._config.sequence_length).mean()
            )
            volume_std = (
                data["volume"].rolling(window=self._config.sequence_length).std()
            )
            volume_ratio = data["volume"] / volume_ma
            # Технические индикаторы
            rsi = self._calculate_rsi(data)
            macd, signal, hist = self._calculate_macd(data)
            upper_band, lower_band = self._calculate_bollinger_bands(data)
            atr = self._calculate_atr(data)
            # Нормализация признаков
            features = {
                "returns": float(returns.iloc[-1]) if returns.iloc[-1] is not None and not pd.isna(returns.iloc[-1]) else 0.0,
                "log_returns": float(log_returns.iloc[-1]) if log_returns.iloc[-1] is not None and not pd.isna(log_returns.iloc[-1]) else 0.0,  # type: ignore
                "volatility": float(volatility.iloc[-1]) if volatility.iloc[-1] is not None and not pd.isna(volatility.iloc[-1]) else 0.0,  # type: ignore
                "volume_ratio": float(volume_ratio.iloc[-1]) if volume_ratio.iloc[-1] is not None and not pd.isna(volume_ratio.iloc[-1]) else 1.0,  # type: ignore
                "rsi": float(rsi.iloc[-1]) if len(rsi) > 0 and rsi.iloc[-1] is not None and not pd.isna(rsi.iloc[-1]) else 50.0,  # type: ignore
                "macd": float(macd.iloc[-1]) if len(macd) > 0 and macd.iloc[-1] is not None and not pd.isna(macd.iloc[-1]) else 0.0,  # type: ignore
                "macd_signal": float(signal.iloc[-1]) if len(signal) > 0 and signal.iloc[-1] is not None and not pd.isna(signal.iloc[-1]) else 0.0,  # type: ignore
                "macd_hist": float(hist.iloc[-1]) if len(hist) > 0 and hist.iloc[-1] is not None and not pd.isna(hist.iloc[-1]) else 0.0,  # type: ignore
                "bollinger_position": (
                    float((data["close"].iloc[-1] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1]))  # type: ignore
                    if len(upper_band) > 0 and len(lower_band) > 0 and upper_band.iloc[-1] is not None and not pd.isna(upper_band.iloc[-1]) and lower_band.iloc[-1] is not None and not pd.isna(lower_band.iloc[-1]) and (upper_band.iloc[-1] - lower_band.iloc[-1]) != 0  # type: ignore
                    else 0.5
                ),
                "atr": float(atr.iloc[-1]) if len(atr) > 0 and atr.iloc[-1] is not None and not pd.isna(atr.iloc[-1]) else 0.0,  # type: ignore
            }
            return features
        except Exception as e:
            logger.error(f"Error calculating features: {str(e)}")
            return {}

    def _calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """
        Расчет RSI.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            pd.Series: RSI
        """
        try:
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self._config.rsi_period).mean()  # type: ignore
            loss = (-delta.where(delta < 0, 0)).rolling(window=self._config.rsi_period).mean()  # type: ignore
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series()  # type: ignore

    def _calculate_macd(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Расчет MACD.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: MACD, Signal, Histogram
        """
        try:
            ema_fast = data["close"].ewm(span=self._config.macd_fast).mean()
            ema_slow = data["close"].ewm(span=self._config.macd_slow).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=self._config.macd_signal).mean()
            histogram = macd - signal
            return macd, signal, histogram
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return pd.Series(), pd.Series(), pd.Series()  # type: ignore

    def _calculate_bollinger_bands(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Расчет Bollinger Bands.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Tuple[pd.Series, pd.Series]: Upper Band, Lower Band
        """
        try:
            sma = data["close"].rolling(window=self._config.bollinger_period).mean()
            std = data["close"].rolling(window=self._config.bollinger_period).std()
            upper_band = sma + (std * self._config.bollinger_std)
            lower_band = sma - (std * self._config.bollinger_std)
            return upper_band, lower_band
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return pd.Series(), pd.Series()  # type: ignore

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        Расчет ATR.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            pd.Series: ATR
        """
        try:
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)  # type: ignore
            atr = true_range.rolling(window=self._config.atr_period).mean()
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series()  # type: ignore

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
            if self.model is None:
                return None
                
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
                if self.stop_loss and current_price <= self.stop_loss:
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
                elif self.take_profit and current_price >= self.take_profit:
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
                if self.stop_loss and current_price >= self.stop_loss:
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
                elif self.take_profit and current_price <= self.take_profit:
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
            if self._config.trailing_stop and self.trailing_stop:
                if self.position == "long" and current_price > self.trailing_stop:
                    self.trailing_stop = (
                        current_price - indicators["atr"] * self._config.trailing_step
                    )
                elif self.position == "short" and current_price < self.trailing_stop:
                    self.trailing_stop = (
                        current_price + indicators["atr"] * self._config.trailing_step
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
            if self._config.partial_close and self.partial_closes and self.take_profit:
                for level, size in zip(
                    self._config.partial_close_levels, self._config.partial_close_sizes
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
            return None
        except Exception as e:
            logger.error(f"Error generating exit signal: {str(e)}")
            return None

    def _update_position_state(self, signal: Signal, data: pd.DataFrame) -> None:
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
                self.partial_closes.append({"volume": signal.volume})
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
            for i in range(len(data) - self._config.sequence_length):
                window_data = data.iloc[i : i + self._config.sequence_length]  # type: ignore
                window_features = self._calculate_features(window_data)
                features.append(list(window_features.values()))
            # Преобразование в массив
            X = np.array(features)
            # Масштабирование признаков
            X = self.scaler.transform(X)
            # Изменение формы для LSTM
            X = X.reshape((1, self._config.sequence_length, X.shape[1]))
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
                "rsi": rsi.iloc[-1] if len(rsi) > 0 else 50.0,  # type: ignore
                "macd": macd.iloc[-1] if len(macd) > 0 else 0.0,  # type: ignore
                "macd_signal": signal.iloc[-1] if len(signal) > 0 else 0.0,  # type: ignore
                "macd_hist": hist.iloc[-1] if len(hist) > 0 else 0.0,  # type: ignore
                "bollinger_upper": upper_band.iloc[-1] if len(upper_band) > 0 else 0.0,  # type: ignore
                "bollinger_lower": lower_band.iloc[-1] if len(lower_band) > 0 else 0.0,  # type: ignore
                "atr": atr.iloc[-1] if len(atr) > 0 else 0.0,  # type: ignore
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
                "high" if indicators["atr"] > data["close"].iloc[-1] * 0.01 else "low"  # type: ignore
            )
            # Объем
            volume_state = (
                "high"
                if data["volume"].iloc[-1]  # type: ignore
                > data["volume"].rolling(window=20).mean().iloc[-1]  # type: ignore
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
            base_size = self._config.risk_per_trade
            # Корректировка на волатильность
            volatility_factor = 1 / (1 + atr / price)
            # Корректировка на максимальный размер
            position_size = base_size * volatility_factor
            position_size = min(position_size, self._config.max_position_size)
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def _build_model(self, input_shape: Tuple[int, int]) -> Optional[Sequential]:
        """
        Построение модели LSTM.
        Args:
            input_shape: Форма входных данных
        Returns:
            Optional[Sequential]: Модель LSTM или None
        """
        try:
            model = Sequential()
            # LSTM слои
            for i, units in enumerate(self._config.lstm_units):
                if i == 0:
                    model.add(
                        LSTM(units, return_sequences=True, input_shape=input_shape)
                    )
                else:
                    model.add(LSTM(units, return_sequences=True))
                model.add(Dropout(self._config.dropout_rate))
            # Полносвязные слои
            for units in self._config.dense_units:
                model.add(Dense(units, activation="relu"))
                model.add(Dropout(self._config.dropout_rate))
            # Выходной слой
            model.add(Dense(2, activation="softmax"))
            # Компиляция модели
            model.compile(
                optimizer=Adam(learning_rate=self._config.learning_rate),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            return model
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            return None

    def train(self, data: pd.DataFrame) -> None:
        """
        Обучение модели.
        Args:
            data: DataFrame с OHLCV данными
        """
        try:
            # Расчет признаков
            features = []
            for i in range(len(data) - self._config.sequence_length):
                window_data = data.iloc[i : i + self._config.sequence_length]  # type: ignore
                window_features = self._calculate_features(window_data)
                features.append(list(window_features.values()))
            # Создание DataFrame с признаками
            features_df = pd.DataFrame(features)
            # Расчет целевой переменной
            future_returns = (
                data["close"]
                .pct_change(self._config.prediction_horizon)  # type: ignore
                .shift(-self._config.prediction_horizon)  # type: ignore
            )
            target = pd.get_dummies((future_returns > 0).astype(int))  # type: ignore
            # Удаление NaN значений
            valid_indices = ~(features_df.isna().any(axis=1) | target.isna().any(axis=1))  # type: ignore
            X = features_df[valid_indices].values
            y = target[valid_indices].values
            # Масштабирование признаков
            X = self.scaler.fit_transform(X)
            # Изменение формы для LSTM
            X = X.reshape((X.shape[0], self._config.sequence_length, X.shape[1]))
            # Построение модели
            self.model = self._build_model((self._config.sequence_length, X.shape[2]))
            if self.model is None:
                logger.error("Failed to build model")
                return
            # Обучение модели
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    f"{self._config.log_dir}/best_model.h5",
                    save_best_only=True,
                ),
            ]
            self.model.fit(
                X,
                y,
                batch_size=self._config.batch_size,
                epochs=self._config.epochs,
                validation_split=self._config.validation_split,
                callbacks=callbacks,
                verbose=0,
            )
            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")

    def save_model(self, path: str) -> None:
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

    def load_model(self, path: str) -> None:
        """
        Загрузка модели.
        Args:
            path: Путь к модели
        """
        try:
            self.model = load_model(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .types import MarketData, Signal


@dataclass
class DataProcessorConfig:
    symbols: Optional[List[str]] = None
    timeframes: Optional[List[str]] = None
    indicators: Optional[List[str]] = None
    normalize: bool = True
    scale: bool = True
    cache_size: int = 128
    feature_selection: bool = False
    random_seed: Optional[int] = None
    log_dir: str = "logs"


@dataclass
class DataProcessorMetrics:
    processed_rows: int = 0
    missing_values: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    features: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)


class DataProcessor:
    """Процессор данных (расширенный)"""

    def __init__(self, config: Dict):
        self.config = (
            DataProcessorConfig(**config) if not isinstance(config, DataProcessorConfig) else config
        )
        self.metrics = DataProcessorMetrics()
        self._setup_logger()
        self._cache = {}
        np.random.seed(self.config.random_seed or 42)

    def _setup_logger(self):
        logger.add(
            f"{self.config.log_dir}/data_processor_{{time}}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def process_market_data(
        self, data: pd.DataFrame, symbol: Optional[str] = None, timeframe: Optional[str] = None
    ) -> List[MarketData]:
        """
        Обработка рыночных данных (поддержка мульти-символов и таймфреймов)
        """
        try:
            required_columns = ["open", "high", "low", "close", "volume"]
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")
            market_data = []
            for idx, row in data.iterrows():
                market_data.append(
                    MarketData(
                        symbol=symbol or row.get("symbol", ""),
                        timeframe=timeframe or row.get("timeframe", ""),
                        timestamp=idx,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        metadata={"index": idx, "row": row.to_dict()},
                    )
                )
            self.metrics.processed_rows += len(market_data)
            return market_data
        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error processing market data: {str(e)}")
            return []

    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Расширенная валидация данных
        """
        try:
            if data.empty:
                return False, "Empty dataset"
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return False, f"Missing columns: {missing_columns}"
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    return False, f"Column {col} is not numeric"
            if data.isnull().any().any():
                self.metrics.missing_values += int(data.isnull().sum().sum())
                return False, "Dataset contains missing values"
            if (data[["open", "high", "low", "close", "volume"]] < 0).any().any():
                return False, "Dataset contains negative values"
            if not (data["high"] >= data["low"]).all():
                return False, "High price is less than low price"
            if (
                not (data["high"] >= data["open"]).all()
                or not (data["high"] >= data["close"]).all()
            ):
                return False, "High price is less than open or close price"
            if not (data["low"] <= data["open"]).all() or not (data["low"] <= data["close"]).all():
                return False, "Low price is less than open or close price"
            return True, None
        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error validating data: {str(e)}")
            return False, str(e)

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Предобработка данных: сортировка, удаление дубликатов, интерполяция, индикаторы, нормализация, масштабирование
        """
        try:
            processed_data = data.copy()
            processed_data.sort_index(inplace=True)
            processed_data.drop_duplicates(inplace=True)
            processed_data.interpolate(method="time", inplace=True)
            processed_data = self._add_technical_indicators(processed_data)
            if self.config.normalize:
                processed_data = self._normalize(processed_data)
            if self.config.scale:
                processed_data = self._scale(processed_data)
            return processed_data
        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error preprocessing data: {str(e)}")
            return data

    @lru_cache(maxsize=32)
    def _cached_indicators(self, data_hash: int) -> pd.DataFrame:
        return self._add_technical_indicators(self._cache[data_hash])

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление расширенного набора технических индикаторов
        """
        try:
            # Скользящие средние
            data["sma_20"] = data["close"].rolling(window=20).mean()
            data["sma_50"] = data["close"].rolling(window=50).mean()
            data["sma_200"] = data["close"].rolling(window=200).mean()
            # EMA
            data["ema_20"] = data["close"].ewm(span=20, adjust=False).mean()
            data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()
            # Волатильность
            data["volatility"] = data["close"].rolling(window=20).std()
            # RSI
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data["rsi"] = 100 - (100 / (1 + rs))
            # MACD
            exp1 = data["close"].ewm(span=12, adjust=False).mean()
            exp2 = data["close"].ewm(span=26, adjust=False).mean()
            data["macd"] = exp1 - exp2
            data["signal"] = data["macd"].ewm(span=9, adjust=False).mean()
            # Bollinger Bands
            data["bb_middle"] = data["close"].rolling(window=20).mean()
            data["bb_std"] = data["close"].rolling(window=20).std()
            data["bb_upper"] = data["bb_middle"] + (data["bb_std"] * 2)
            data["bb_lower"] = data["bb_middle"] - (data["bb_std"] * 2)
            # ATR
            data["tr"] = np.maximum.reduce(
                [
                    data["high"] - data["low"],
                    abs(data["high"] - data["close"].shift()),
                    abs(data["low"] - data["close"].shift()),
                ]
            )
            data["atr_14"] = data["tr"].rolling(window=14).mean()
            # ADX
            up_move = data["high"].diff()
            down_move = data["low"].diff()
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            tr = data["tr"]
            plus_di = (
                100 * pd.Series(plus_dm).rolling(window=14).sum() / tr.rolling(window=14).sum()
            )
            minus_di = (
                100 * pd.Series(minus_dm).rolling(window=14).sum() / tr.rolling(window=14).sum()
            )
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            data["adx_14"] = dx.rolling(window=14).mean()
            # OBV
            data["obv"] = (np.sign(data["close"].diff()) * data["volume"]).fillna(0).cumsum()
            # CCI
            tp = (data["high"] + data["low"] + data["close"]) / 3
            data["cci_20"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
            # Stochastic
            data["stoch_k"] = (
                100
                * (data["close"] - data["low"].rolling(14).min())
                / (data["high"].rolling(14).max() - data["low"].rolling(14).min())
            )
            data["stoch_d"] = data["stoch_k"].rolling(3).mean()
            # VWAP
            data["vwap"] = (data["close"] * data["volume"]).cumsum() / data["volume"].cumsum()
            # Momentum
            data["momentum_10"] = data["close"].diff(10)
            self.metrics.features = list(data.columns)
            return data
        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error adding technical indicators: {str(e)}")
            return data

    def _normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / (
                data[numeric_cols].std() + 1e-8
            )
            return data
        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            return data

    def _scale(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].min()) / (
                data[numeric_cols].max() - data[numeric_cols].min() + 1e-8
            )
            return data
        except Exception as e:
            logger.error(f"Error scaling data: {str(e)}")
            return data

    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Подготовка данных для обучения: генерация фичей, удаление пропусков, разделение на признаки и метки
        """
        try:
            processed_data = self.preprocess_data(data)
            processed_data.dropna(inplace=True)
            features = processed_data.drop(
                ["open", "high", "low", "close", "volume"], axis=1, errors="ignore"
            )
            labels = processed_data[["close"]].shift(-1)
            features = features[:-1]
            labels = labels[:-1]
            return features, labels
        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error preparing training data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

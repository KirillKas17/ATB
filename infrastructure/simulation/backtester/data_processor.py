from shared.numpy_utils import np
import pandas as pd
from pandas import DataFrame, Series
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache
import logging
from dataclasses import dataclass, field
from loguru import logger

from domain.entities.market import MarketData
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.type_definitions import TimestampValue
from domain.value_objects.currency import Currency


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
            DataProcessorConfig(**config)
            if not isinstance(config, DataProcessorConfig)
            else config
        )
        self.metrics = DataProcessorMetrics()
        self._setup_logger()
        self._cache: Dict[int, pd.DataFrame] = {}
        np.random.seed(self.config.random_seed or 42)

    def _setup_logger(self) -> None:
        logger.add(
            f"{self.config.log_dir}/data_processor_{{time}}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def process_market_data(
        self,
        data: pd.DataFrame,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
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
                # Преобразуем timestamp в datetime
                if isinstance(idx, (str, int, float, bool)):
                    from datetime import datetime
                    if isinstance(idx, str):
                        try:
                            timestamp = datetime.fromisoformat(idx)
                        except ValueError:
                            timestamp = datetime.now()
                    else:
                        timestamp = datetime.now()
                else:
                    timestamp = idx if isinstance(idx, datetime) else datetime.now()
                
                from domain.type_definitions import Symbol, MetadataDict
                from domain.value_objects.currency import Currency
                from decimal import Decimal
                
                market_data.append(
                    MarketData(
                        symbol=Symbol(symbol or row.get("symbol", "")),
                        timestamp=TimestampValue(timestamp),
                        open=Price(Decimal(str(row["open"])), Currency.USD),
                        high=Price(Decimal(str(row["high"])), Currency.USD),
                        low=Price(Decimal(str(row["low"])), Currency.USD),
                        close=Price(Decimal(str(row["close"])), Currency.USD),
                        volume=Volume(Decimal(str(row["volume"])), Currency.USD),
                        metadata=MetadataDict({"index": str(idx), "row": row.to_dict() if hasattr(row, 'to_dict') else dict(row.items() if hasattr(row, 'items') else [])}),
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
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                return False, f"Missing columns: {missing_columns}"
            for col in required_columns:
                if hasattr(pd, 'api') and hasattr(pd.api, 'types') and hasattr(pd.api.types, 'is_numeric_dtype'):
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        return False, f"Column {col} is not numeric"
                else:
                    # Fallback проверка
                    if not np.issubdtype(data[col].dtype, np.number):  # type: ignore[arg-type]
                        return False, f"Column {col} is not numeric"
            # Проверяем наличие NaN значений
            if hasattr(data, 'isna'):
                isna_result: pd.DataFrame = data.isna()
                if isna_result.any().any():
                    sum_result: pd.Series = isna_result.sum()
                    self.metrics.missing_values += int(sum_result.sum())
                    return False, "Dataset contains missing values"
            # Проверяем отрицательные значения
            negative_check: pd.DataFrame = data[["open", "high", "low", "close", "volume"]] < 0
            if negative_check.any().any():
                return False, "Dataset contains negative values"
            if not (data["high"] >= data["low"]).all():
                return False, "High price is less than low price"
            if (
                not (data["high"] >= data["open"]).all()
                or not (data["high"] >= data["close"]).all()
            ):
                return False, "High price is less than open or close price"
            if (
                not (data["low"] <= data["open"]).all()
                or not (data["low"] <= data["close"]).all()
            ):
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
            if hasattr(processed_data, 'sort_index'):
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
            # Создаем копию данных для модификации
            result_data = data.copy()
            
            # Скользящие средние
            if "close" in result_data.columns:
                result_data["sma_20"] = result_data["close"].rolling(window=20).mean()
                result_data["sma_50"] = result_data["close"].rolling(window=50).mean()
                result_data["sma_200"] = result_data["close"].rolling(window=200).mean()
                # EMA
                result_data["ema_20"] = result_data["close"].ewm(span=20, adjust=False).mean()
                result_data["ema_50"] = result_data["close"].ewm(span=50, adjust=False).mean()
                # Волатильность
                result_data["volatility"] = result_data["close"].rolling(window=20).std()
            # RSI
            if "close" in result_data.columns:
                delta = result_data["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                result_data["rsi"] = 100 - (100 / (1 + rs))
                # MACD
                exp1 = result_data["close"].ewm(span=12, adjust=False).mean()
                exp2 = result_data["close"].ewm(span=26, adjust=False).mean()
                result_data["macd"] = exp1 - exp2
                result_data["signal"] = result_data["macd"].ewm(span=9, adjust=False).mean()
                # Bollinger Bands
                result_data["bb_middle"] = result_data["close"].rolling(window=20).mean()
                result_data["bb_std"] = result_data["close"].rolling(window=20).std()
                result_data["bb_upper"] = result_data["bb_middle"] + (result_data["bb_std"] * 2)
                result_data["bb_lower"] = result_data["bb_middle"] - (result_data["bb_std"] * 2)
            # ATR
            if all(col in result_data.columns for col in ["high", "low", "close"]):
                result_data["tr"] = np.maximum.reduce(
                    [
                        result_data["high"] - result_data["low"],
                        abs(result_data["high"] - result_data["close"].shift()),
                        abs(result_data["low"] - result_data["close"].shift()),
                    ]
                )
                result_data["atr_14"] = result_data["tr"].rolling(window=14).mean()
                # ADX
                up_move = result_data["high"].diff()
                down_move = result_data["low"].diff()
                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
                tr = result_data["tr"]
                plus_di = (
                    100
                    * pd.Series(plus_dm).rolling(window=14).sum()
                    / tr.rolling(window=14).sum()
                )
                minus_di = (
                    100
                    * pd.Series(minus_dm).rolling(window=14).sum()
                    / tr.rolling(window=14).sum()
                )
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                result_data["adx_14"] = dx.rolling(window=14).mean()
            # OBV
            if "close" in result_data.columns and "volume" in result_data.columns:
                result_data["obv"] = (
                    (np.sign(result_data["close"].diff()) * result_data["volume"]).fillna(0).cumsum()
                )
            # CCI
            if all(col in result_data.columns for col in ["high", "low", "close"]):
                tp = (result_data["high"] + result_data["low"] + result_data["close"]) / 3
                result_data["cci_20"] = (tp - tp.rolling(20).mean()) / (
                    0.015 * tp.rolling(20).std()
                )
            # Stochastic
            if all(col in result_data.columns for col in ["high", "low", "close"]):
                result_data["stoch_k"] = (
                    100
                    * (result_data["close"] - result_data["low"].rolling(14).min())
                    / (result_data["high"].rolling(14).max() - result_data["low"].rolling(14).min())
                )
                result_data["stoch_d"] = result_data["stoch_k"].rolling(3).mean()
            # VWAP
            if "close" in result_data.columns and "volume" in result_data.columns:
                result_data["vwap"] = (result_data["close"] * result_data["volume"]).cumsum() / result_data[
                    "volume"
                ].cumsum()
            # Momentum
            if "close" in result_data.columns:
                result_data["momentum_10"] = result_data["close"].diff(10)
            # Проверяем и инициализируем features если нужно
            if not hasattr(self.metrics, 'features') or self.metrics.features is None:
                self.metrics.features = []
            if isinstance(self.metrics.features, list):
                # Приводим к числовому типу для корректной работы
                numeric_features = [col for col in result_data.columns if pd.api.types.is_numeric_dtype(result_data[col])]
                self.metrics.features.extend(numeric_features)
            return result_data
        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error adding technical indicators: {str(e)}")
            return data

    def _normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            numeric_cols: pd.Index = data.select_dtypes(include=[np.number]).columns
            result_data: pd.DataFrame = data.copy()
            # Приводим к float для корректных операций
            result_data.loc[:, numeric_cols] = result_data[numeric_cols].astype(float)
            result_data.loc[:, numeric_cols] = (result_data[numeric_cols] - result_data[numeric_cols].mean()) / (
                result_data[numeric_cols].std() + 1e-8
            )
            return result_data
        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            return data

    def _scale(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            numeric_cols: pd.Index = data.select_dtypes(include=[np.number]).columns
            result_data: pd.DataFrame = data.copy()
            # Приводим к float для корректных операций
            result_data.loc[:, numeric_cols] = result_data[numeric_cols].astype(float)
            result_data.loc[:, numeric_cols] = (result_data[numeric_cols] - result_data[numeric_cols].min()) / (
                result_data[numeric_cols].max() - result_data[numeric_cols].min() + 1e-8
            )
            return result_data
        except Exception as e:
            logger.error(f"Error scaling data: {str(e)}")
            return data

    def prepare_training_data(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Подготовка данных для обучения
        """
        try:
            # Удаляем строки с пропущенными значениями
            clean_data = data.dropna()
            # Удаляем ненужные колонки
            feature_cols = [col for col in clean_data.columns if col not in ["open", "high", "low", "close", "volume"]]
            target_cols = ["close"]
            
            X = clean_data[feature_cols]
            y = clean_data[target_cols]
            
            return X, y
        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            logger.error(f"Error preparing training data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

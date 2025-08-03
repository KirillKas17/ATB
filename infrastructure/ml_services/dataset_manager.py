"""
DatasetManager - управление датасетами для ML моделей
Загрузка, предобработка, валидация и кэширование данных
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore

from shared.unified_cache import get_cache_manager

# Type aliases


class DatasetManager:
    """
    Менеджер датасетов для ML моделей.
    Отвечает за:
    - Загрузку и предобработку данных
    - Валидацию качества данных
    - Кэширование датасетов
    - Создание признаков для ML моделей
    - Разделение на train/validation/test
    """

    def __init__(
        self, cache_dir: str = "cache/datasets", max_cache_size: int = 1000
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        self.cache_manager = get_cache_manager()
        # Кэш для предобработанных датасетов
        self._dataset_cache: Dict[str, pd.DataFrame] = {}
        self._scaler_cache: Dict[str, Union[StandardScaler, MinMaxScaler]] = {}
        # Конфигурация предобработки
        self.preprocessing_config: Dict[str, Any] = {
            "fill_method": "ffill",  # forward fill для пропусков
            "outlier_method": "iqr",  # межквартильный размах для выбросов
            "normalization": "standard",  # стандартизация
            "feature_engineering": True,
            "window_sizes": [5, 10, 20, 50],
            "target_column": "target",
            "sequence_length": 60,
        }

    async def load_dataset(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 10000,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Загрузка датасета для символа.
        Args:
            symbol: Торговая пара
            timeframe: Временной интервал
            limit: Количество записей
            force_reload: Принудительная перезагрузка
        Returns:
            DataFrame с данными
        """
        cache_key = f"{symbol}_{timeframe}_{limit}"
        # Проверяем кэш
        if not force_reload and cache_key in self._dataset_cache:
            logger.info(f"Loading dataset from cache: {cache_key}")
            return self._dataset_cache[cache_key]
        try:
            # Загружаем данные из внешнего источника
            # В реальной системе здесь будет вызов к exchange API или БД
            data = await self._fetch_market_data(symbol, timeframe, limit)
            if data is None or data.empty:
                raise ValueError(f"No data available for {symbol}")
            # Предобработка данных
            processed_data = await self._preprocess_data(data, symbol)
            # Кэшируем результат
            self._dataset_cache[cache_key] = processed_data
            # Ограничиваем размер кэша
            if len(self._dataset_cache) > self.max_cache_size:
                self._cleanup_cache()
            logger.info(
                f"Loaded and processed dataset for {symbol}: {len(processed_data)} records"
            )
            return processed_data
        except Exception as e:
            logger.error(f"Error loading dataset for {symbol}: {e}")
            raise

    async def _fetch_market_data(
        self, symbol: str, timeframe: str, limit: int
    ) -> Optional[pd.DataFrame]:
        """
        Загрузка рыночных данных из внешнего источника.
        Args:
            symbol: Торговая пара
            timeframe: Временной интервал
            limit: Количество записей
        Returns:
            DataFrame с рыночными данными
        """
        try:
            # Временная заглушка - генерируем синтетические данные
            # В реальной системе здесь будет вызов к exchange API
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=limit)
            # Генерируем временные метки
            timestamps = pd.date_range(start=start_time, end=end_time, periods=limit)
            # Генерируем синтетические OHLCV данные
            np.random.seed(42)  # Для воспроизводимости
            base_price = 50000.0  # Базовая цена
            returns = np.random.normal(0, 0.02, limit)  # 2% волатильность
            prices = [base_price]
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(new_price)
            # Создаем OHLCV данные
            data: List[Dict[str, Any]] = []
            for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
                # Генерируем OHLC на основе цены
                high = price * (1 + abs(np.random.normal(0, 0.005)))
                low = price * (1 - abs(np.random.normal(0, 0.005)))
                open_price = price * (1 + np.random.normal(0, 0.002))
                close_price = price
                volume = np.random.uniform(1000, 10000)
                data.append(
                    {
                        "timestamp": timestamp,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close_price,
                        "volume": volume,
                        "symbol": symbol,
                    }
                )
            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None

    async def _preprocess_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Предобработка данных.
        Args:
            data: Исходные данные
            symbol: Торговая пара
        Returns:
            Обработанные данные
        """
        try:
            df = data.copy()
            # Убеждаемся, что индекс является DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            # 1. Обработка пропусков
            df = self._handle_missing_values(df)
            # 2. Обработка выбросов
            df = self._handle_outliers(df)
            # 3. Создание технических индикаторов
            df = self._create_technical_features(df)
            # 4. Создание целевой переменной
            df = self._create_target_variable(df)
            # 5. Нормализация данных
            df = self._normalize_features(df, symbol)
            # 6. Создание временных признаков
            df = self._create_temporal_features(df)
            # 7. Удаление строк с пропусками после создания признаков
            df = df.dropna()
            logger.info(
                f"Preprocessed data for {symbol}: {len(df)} records, {len(df.columns)} features"
            )
            return df
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка пропущенных значений."""
        # Заполняем пропуски методом forward fill
        df = df.fillna(method="ffill")
        # Если остались пропуски, заполняем backward fill
        df = df.fillna(method="bfill")
        # Если все еще есть пропуски, заполняем средним
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка выбросов."""
        if self.preprocessing_config["outlier_method"] == "iqr":
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                # Заменяем выбросы на граничные значения
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        return df

    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание технических индикаторов."""
        if not self.preprocessing_config["feature_engineering"]:
            return df
        # Простые скользящие средние
        for window in self.preprocessing_config["window_sizes"]:
            df[f"sma_{window}"] = df["close"].rolling(window=window).mean()
            df[f"ema_{window}"] = df["close"].ewm(span=window).mean()
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        # Bollinger Bands
        for window in [20]:
            sma = df["close"].rolling(window=window).mean()  # type: ignore
            std = df["close"].rolling(window=window).std()  # type: ignore
            df[f"bb_upper_{window}"] = sma + (std * 2)
            df[f"bb_lower_{window}"] = sma - (std * 2)
        # MACD
        ema12 = df["close"].ewm(span=12).mean()  # type: ignore
        ema26 = df["close"].ewm(span=26).mean()  # type: ignore
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()  # type: ignore
        # Объемные индикаторы
        df["volume_sma"] = df["volume"].rolling(window=20).mean()  # type: ignore
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        return df

    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание целевой переменной."""
        # Простая целевая переменная - изменение цены через N периодов
        target_periods = 5
        df[self.preprocessing_config["target_column"]] = (
            df["close"].shift(-target_periods) / df["close"] - 1  # type: ignore
        )
        return df

    def _normalize_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Нормализация признаков."""
        cache_key = f"scaler_{symbol}"
        if cache_key in self._scaler_cache:
            scaler = self._scaler_cache[cache_key]
        else:
            if self.preprocessing_config["normalization"] == "standard":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            self._scaler_cache[cache_key] = scaler
        # Выбираем числовые колонки для нормализации
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        # Исключаем целевую переменную из нормализации
        feature_columns = [col for col in numeric_columns if col != self.preprocessing_config["target_column"]]
        if feature_columns:
            # Приводим df к pandas DataFrame для индексирования
            if hasattr(df, 'iloc'):
                df_pandas = df
            else:
                df_pandas = pd.DataFrame(df)
            # Безопасное индексирование
            if hasattr(df_pandas, 'iloc'):
                df_pandas.loc[:, feature_columns] = scaler.fit_transform(df_pandas[feature_columns])  # type: ignore
                # Обновляем исходный df
                for col in feature_columns:
                    if hasattr(df, 'loc'):
                        df.loc[:, col] = df_pandas[col]  # type: ignore
                    else:
                        df[col] = df_pandas[col]  # type: ignore
        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание временных признаков."""
        if hasattr(df, 'index') and isinstance(df.index, pd.DatetimeIndex):
            df["hour"] = df.index.hour  # type: ignore
            df["day_of_week"] = df.index.dayofweek  # type: ignore
            df["day_of_month"] = df.index.day  # type: ignore
            df["month"] = df.index.month  # type: ignore
            df["quarter"] = df.index.quarter  # type: ignore
        return df

    async def create_sequences(
        self, df: pd.DataFrame, sequence_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Создание последовательностей для LSTM/RNN моделей.
        Args:
            df: DataFrame с данными
            sequence_length: Длина последовательности
        Returns:
            Кортеж (X, y) с последовательностями
        """
        if sequence_length is None:
            sequence_length = self.preprocessing_config["sequence_length"]
        # Выбираем признаки (исключаем целевую переменную)
        feature_columns = [col for col in df.columns if col != self.preprocessing_config["target_column"]]
        target_column = self.preprocessing_config["target_column"]
        # Создаем последовательности
        X, y = [], []
        for i in range(len(df) - sequence_length):
            X.append(df[feature_columns].iloc[i:i + sequence_length].values)  # type: ignore
            y.append(df[target_column].iloc[i + sequence_length])  # type: ignore
        return np.array(X), np.array(y)

    async def split_dataset(
        self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Разделение датасета на train/validation/test.
        Args:
            df: DataFrame с данными
            train_ratio: Доля тренировочных данных
            val_ratio: Доля валидационных данных
        Returns:
            Кортеж (train, validation, test)
        """
        total_size = len(df)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        train_df: pd.DataFrame = df.iloc[:train_size]  # type: ignore[index]
        val_df: pd.DataFrame = df.iloc[train_size:train_size + val_size]  # type: ignore[index]
        test_df: pd.DataFrame = df.iloc[train_size + val_size:]  # type: ignore[index]
        return train_df, val_df, test_df

    async def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Валидация качества датасета.
        Args:
            df: DataFrame для валидации
        Returns:
            Словарь с метриками качества
        """
        validation_results = {
            "total_records": len(df),
            "missing_values": {},
            "duplicates": len(df.duplicated()),  # type: ignore[attr-defined]
            "data_types": {},
            "numeric_stats": {},
            "quality_score": 0.0,
        }
        # Проверка пропущенных значений
        for col in df.columns:
            missing_count = df[col].isnull().sum()  # type: ignore
            validation_results["missing_values"][col] = missing_count
        # Проверка типов данных
        for col in df.columns:
            validation_results["data_types"][col] = str(df[col].dtype)  # type: ignore
        # Статистика числовых колонок
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            col_data: pd.Series = df[col]
            # Проверяем, есть ли атрибут values
            if hasattr(col_data, 'values'):
                col_values: np.ndarray = col_data.values
            else:
                col_values: np.ndarray = np.array(col_data)
            validation_results["numeric_stats"][col] = {
                "mean": float(np.mean(col_values)),
                "std": float(np.std(col_values)),
                "min": float(np.min(col_values)),
                "max": float(np.max(col_values)),
            }
        # Вычисление общего качества
        total_cells = len(df) * len(df.columns)
        missing_cells = sum(validation_results["missing_values"].values())
        quality_score = 1.0 - (missing_cells / total_cells)
        validation_results["quality_score"] = quality_score
        return validation_results

    def _cleanup_cache(self):
        """Очистка кэша при превышении лимита."""
        if len(self._dataset_cache) > self.max_cache_size:
            # Удаляем самые старые записи
            keys_to_remove = list(self._dataset_cache.keys())[:len(self._dataset_cache) - self.max_cache_size]
            for key in keys_to_remove:
                del self._dataset_cache[key]
            logger.info(f"Cleaned up cache, removed {len(keys_to_remove)} entries")

    async def get_dataset_info(self, symbol: str) -> Dict[str, Any]:
        """
        Получение информации о датасете.
        Args:
            symbol: Торговая пара
        Returns:
            Словарь с информацией
        """
        try:
            df = await self.load_dataset(symbol)
            info = {
                "symbol": symbol,
                "total_records": len(df),
                "columns": list(df.columns),
                "date_range": {
                    "start": str(df.index.min()) if len(df) > 0 else None,
                    "end": str(df.index.max()) if len(df) > 0 else None,
                },
                "memory_usage": df.memory_usage(deep=True).sum(),  # type: ignore
                "cached": symbol in self._dataset_cache,
            }
            return info
        except Exception as e:
            logger.error(f"Error getting dataset info for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

    async def clear_cache(self):
        """Очистка всего кэша."""
        self._dataset_cache.clear()
        self._scaler_cache.clear()
        logger.info("Dataset cache cleared")

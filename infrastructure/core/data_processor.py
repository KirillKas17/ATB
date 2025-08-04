# -*- coding: utf-8 -*-
"""Процессор данных для infrastructure слоя."""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from shared.numpy_utils import np
import pandas as pd
from shared.logging import LoggerMixin

# Type aliases для pandas
DataFrame = pd.DataFrame
Series = pd.Series


class DataProcessor(LoggerMixin):
    """Процессор для обработки рыночных данных."""

    def __init__(self) -> None:
        super().__init__()
        self.processors: Dict[str, Callable] = {}
        self._register_default_processors()

    def _register_default_processors(self) -> None:
        """Регистрация стандартных процессоров."""
        self.processors.update(
            {
                "sma": self._calculate_sma,
                "ema": self._calculate_ema,
                "rsi": self._calculate_rsi,
                "macd": self._calculate_macd,
                "bollinger_bands": self._calculate_bollinger_bands,
                "atr": self._calculate_atr,
                "volume_sma": self._calculate_volume_sma,
                "price_change": self._calculate_price_change,
                "volatility": self._calculate_volatility,
                "momentum": self._calculate_momentum,
            }
        )

    async def process_market_data(
        self,
        data: pd.DataFrame,
        indicators: List[str],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Обработка рыночных данных с вычислением индикаторов."""
        try:
            result = data.copy()
            for indicator in indicators:
                if indicator in self.processors:
                    result = await self.processors[indicator](result, parameters or {})
                else:
                    self.log_warning(f"Unknown indicator: {indicator}")
            self.log_info(f"Processed market data with {len(indicators)} indicators")
            return result
        except Exception as e:
            self.log_error(f"Failed to process market data: {str(e)}")
            raise

    async def _calculate_sma(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Вычисление простой скользящей средней."""
        period = params.get("period", 20)
        column = params.get("column", "close")
        if column in data.columns:
            data[f"sma_{period}"] = data[column].rolling(window=period).mean()
        return data

    async def _calculate_ema(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Вычисление экспоненциальной скользящей средней."""
        period = params.get("period", 20)
        column = params.get("column", "close")
        if column in data.columns:
            data[f"ema_{period}"] = data[column].ewm(span=period).mean()
        return data

    async def _calculate_rsi(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Вычисление RSI."""
        period = params.get("period", 14)
        column = params.get("column", "close")
        if column in data.columns:
            delta = data[column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            data[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        return data

    async def _calculate_macd(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Вычисление MACD."""
        fast_period = params.get("fast_period", 12)
        slow_period = params.get("slow_period", 26)
        signal_period = params.get("signal_period", 9)
        column = params.get("column", "close")
        if column in data.columns:
            ema_fast = data[column].ewm(span=fast_period).mean()
            ema_slow = data[column].ewm(span=slow_period).mean()
            data["macd"] = ema_fast - ema_slow
            data["macd_signal"] = data["macd"].ewm(span=signal_period).mean()
            data["macd_histogram"] = data["macd"] - data["macd_signal"]
        return data

    async def _calculate_bollinger_bands(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Вычисление полос Боллинджера."""
        period = params.get("period", 20)
        std_dev = params.get("std_dev", 2)
        column = params.get("column", "close")
        if column in data.columns:
            sma = data[column].rolling(window=period).mean()
            std = data[column].rolling(window=period).std()
            data[f"bb_upper_{period}"] = sma + (std * std_dev)
            data[f"bb_middle_{period}"] = sma
            data[f"bb_lower_{period}"] = sma - (std * std_dev)
        return data

    async def _calculate_atr(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Вычисление Average True Range."""
        period = params.get("period", 14)
        if all(col in data.columns for col in ["high", "low", "close"]):
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            ranges_df = pd.DataFrame({
                'high_low': high_low,
                'high_close': high_close,
                'low_close': low_close
            })
            true_range = ranges_df.max(axis=1)
            data[f"atr_{period}"] = true_range.rolling(window=period).mean()
        return data

    async def _calculate_volume_sma(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Вычисление скользящей средней объема."""
        period = params.get("period", 20)
        column = params.get("column", "volume")
        if column in data.columns:
            data[f"volume_sma_{period}"] = data[column].rolling(window=period).mean()
        return data

    async def _calculate_price_change(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Вычисление изменения цены."""
        column = params.get("column", "close")
        periods = params.get("periods", [1, 5, 10])
        if column in data.columns:
            for period in periods:
                data[f"price_change_{period}"] = data[column].pct_change(periods=period)
        return data

    async def _calculate_volatility(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Вычисление волатильности."""
        period = params.get("period", 20)
        column = params.get("column", "close")
        if column in data.columns:
            returns = data[column].pct_change()
            data[f"volatility_{period}"] = returns.rolling(window=period).std()
        return data

    async def _calculate_momentum(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Вычисление моментума."""
        period = params.get("period", 10)
        column = params.get("column", "close")
        if column in data.columns:
            data[f"momentum_{period}"] = data[column] - data[column].shift(period)
        return data

    async def normalize_data(
        self, data: pd.DataFrame, method: str = "minmax"
    ) -> pd.DataFrame:
        """Нормализация данных."""
        try:
            result = data.copy()
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                if method == "minmax":
                    min_val = data[column].min()
                    max_val = data[column].max()
                    if max_val > min_val:
                        result[column] = (data[column] - min_val) / (max_val - min_val)
                elif method == "zscore":
                    mean_val = data[column].mean()
                    std_val = data[column].std()
                    if std_val > 0:
                        result[column] = (data[column] - mean_val) / std_val
                elif method == "robust":
                    median_val = data[column].median()
                    mad_val = data[column].mad()
                    if mad_val > 0:
                        result[column] = (data[column] - median_val) / mad_val
            self.log_info(f"Normalized data using {method} method")
            return result
        except Exception as e:
            self.log_error(f"Failed to normalize data: {str(e)}")
            raise

    async def remove_outliers(
        self, data: pd.DataFrame, method: str = "iqr", threshold: float = 1.5
    ) -> pd.DataFrame:
        """Удаление выбросов из данных."""
        try:
            result = data.copy()
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                if method == "iqr":
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    mask: pd.Series[bool] = (data[column] >= lower_bound) & (data[column] <= upper_bound)
                    result = result[mask]
                elif method == "zscore":
                    z_scores = np.abs(
                        (data[column] - data[column].mean()) / data[column].std()
                    )
                    mask = z_scores < threshold
                    result = result[mask]
            self.log_info(f"Removed outliers using {method} method")
            return result
        except Exception as e:
            self.log_error(f"Failed to remove outliers: {str(e)}")
            raise

    async def resample_data(
        self, data: pd.DataFrame, freq: str, method: str = "last"
    ) -> pd.DataFrame:
        """Изменение частоты данных."""
        try:
            if method == "last":
                result = data.resample(freq).last()
            elif method == "first":
                result = data.resample(freq).first()
            elif method == "mean":
                result = data.resample(freq).mean()
            elif method == "sum":
                # Безопасная альтернатива для sum
                result = data.resample(freq).agg(lambda x: x.sum() if hasattr(x, 'sum') else x)
            else:
                result = data.resample(freq).last()
            result = result.dropna()
            self.log_info(f"Resampled data to {freq} frequency using {method} method")
            return result
        except Exception as e:
            self.log_error(f"Failed to resample data: {str(e)}")
            raise

    async def add_custom_indicator(self, name: str, processor: Callable) -> None:
        """Добавление пользовательского индикатора."""
        self.processors[name] = processor
        self.log_info(f"Added custom indicator: {name}")

    async def get_available_indicators(self) -> List[str]:
        """Получение списка доступных индикаторов."""
        return list(self.processors.keys())

    async def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Валидация данных."""
        validation_result: Dict[str, Any] = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
        }
        try:
            # Проверка на пустые данные
            if data.empty:
                validation_result["is_valid"] = False
                validation_result["errors"].append("Data is empty")
                return validation_result
            # Проверка на дубликаты
            try:
                duplicates = data.index.duplicated().sum()
            except AttributeError:
                duplicates = 0
            if duplicates > 0:
                validation_result["warnings"].append(
                    f"Found {duplicates} duplicate timestamps"
                )
            # Проверка на пропущенные значения
            try:
                missing_values = data.isnull().sum()
                total_missing = int(missing_values.sum())
            except AttributeError:
                missing_values = pd.Series()
                total_missing = 0
            if total_missing > 0:
                validation_result["warnings"].append(
                    f"Found {total_missing} missing values"
                )
            # Статистика данных
            validation_result["stats"] = {
                "rows": len(data),
                "columns": len(data.columns),
                "date_range": {"start": data.index.min(), "end": data.index.max()},
                "missing_values": missing_values.to_dict(),
                "data_types": data.dtypes.to_dict(),
            }
            self.log_info("Data validation completed")
            return validation_result
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Validation failed: {str(e)}")
            return validation_result

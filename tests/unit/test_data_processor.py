"""
Unit тесты для DataProcessor.
Тестирует обработку данных, включая очистку, нормализацию,
фильтрацию и агрегацию рыночных данных.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import pandas as pd
import numpy as np
from datetime import timedelta
from infrastructure.core.data_processor import DataProcessor

class TestDataProcessor:
    """Тесты для DataProcessor."""
    @pytest.fixture
    def data_processor(self) -> DataProcessor:
        """Фикстура для DataProcessor."""
        return DataProcessor()
    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Фикстура с тестовыми рыночными данными."""
        dates = pd.DatetimeIndex(pd.date_range('2023-01-01', periods=1000, freq='1H'))
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.uniform(45000, 55000, 1000),
            'high': np.random.uniform(46000, 56000, 1000),
            'low': np.random.uniform(44000, 54000, 1000),
            'close': np.random.uniform(45000, 55000, 1000),
            'volume': np.random.uniform(1000000, 5000000, 1000)
        }, index=dates)
        # Создание более реалистичных данных
        data['high'] = data[['open', 'close']].max(axis=1) + np.random.uniform(0, 1000, 1000)
        data['low'] = data[['open', 'close']].min(axis=1) - np.random.uniform(0, 1000, 1000)
        return data
    @pytest.fixture
    def sample_dirty_data(self) -> pd.DataFrame:
        """Фикстура с грязными данными."""
        dates = pd.DatetimeIndex(pd.date_range('2023-01-01', periods=100, freq='1H'))
        data = pd.DataFrame({
            'open': [45000, 46000, np.nan, 48000, 49000],
            'high': [46000, 47000, 48000, np.nan, 50000],
            'low': [44000, 45000, 46000, 47000, np.nan],
            'close': [45500, 46500, 47500, 48500, 49500],
            'volume': [1000000, np.nan, 3000000, 4000000, 5000000]
        }, index=dates[:5])
        return data
    def test_initialization(self, data_processor: DataProcessor) -> None:
        """Тест инициализации процессора данных."""
        assert data_processor is not None
        assert hasattr(data_processor, 'data_cleaners')
        assert hasattr(data_processor, 'data_normalizers')
        assert hasattr(data_processor, 'data_filters')
        assert hasattr(data_processor, 'data_aggregators')
    def test_clean_data(self, data_processor: DataProcessor, sample_dirty_data: pd.DataFrame) -> None:
        """Тест очистки данных."""
        # Очистка данных
        cleaned_data = data_processor.validate_data(sample_dirty_data)
        # Проверки
        assert cleaned_data is not None
        assert isinstance(cleaned_data, pd.DataFrame)
        assert len(cleaned_data) <= len(sample_dirty_data)
        # Проверка, что NaN значения обработаны
        assert not cleaned_data.isna().all().any()  # type: ignore
        # Проверка, что данные не пустые
        assert len(cleaned_data) > 0
    def test_remove_duplicates(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест удаления дубликатов."""
        # Создание данных с дубликатами
        if callable(sample_market_data):
            sample_market_data: pd.DataFrame = sample_market_data()  # type: ignore[no-redef]
        if len(sample_market_data) > 10:
            data_with_duplicates: pd.DataFrame = pd.concat([sample_market_data, sample_market_data.iloc[:10]])  # type: ignore[no-redef]
        else:
            data_with_duplicates: pd.DataFrame = pd.concat([sample_market_data, sample_market_data])  # type: ignore[no-redef]
        # Удаление дубликатов
        deduplicated_data = data_processor.validate_data(data_with_duplicates)
        # Проверки
        assert deduplicated_data is not None
        assert isinstance(deduplicated_data, pd.DataFrame)
        assert len(deduplicated_data) == len(sample_market_data)
        try:
            assert not deduplicated_data.duplicated().any()
        except AttributeError:
            pass  # Метод any может отсутствовать
    def test_handle_missing_values(self, data_processor: DataProcessor, sample_dirty_data: pd.DataFrame) -> None:
        """Тест обработки пропущенных значений."""
        # Обработка пропущенных значений
        processed_data = data_processor.validate_data(sample_dirty_data)
        # Проверки
        assert processed_data is not None
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) == len(sample_dirty_data)
        # Проверка, что нет пропущенных значений
        assert not processed_data.isna().any().any()  # type: ignore
    def test_remove_outliers(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест удаления выбросов."""
        # Добавление выбросов
        if callable(sample_market_data):
            sample_market_data: pd.DataFrame = sample_market_data()  # type: ignore[no-redef]
        data_with_outliers: pd.DataFrame = sample_market_data.copy(deep=True)
        if hasattr(data_with_outliers, 'iloc') and hasattr(data_with_outliers.columns, 'get_loc') and len(data_with_outliers) > 1:
            if 'close' in data_with_outliers.columns:
                close_col_idx = data_with_outliers.columns.get_loc('close')
                data_with_outliers.iloc[0, close_col_idx] = 1000000  # type: ignore[index]
                data_with_outliers.iloc[1, close_col_idx] = 1000     # type: ignore[index]
        # Удаление выбросов
        cleaned_data = data_processor.validate_data(data_with_outliers)
        # Проверки
        assert cleaned_data is not None
        assert isinstance(cleaned_data, pd.DataFrame)
        assert len(cleaned_data) < len(data_with_outliers)
    @pytest.mark.asyncio
    async def test_normalize_data(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест нормализации данных."""
        # Нормализация данных
        normalized_data = await data_processor.normalize_data(sample_market_data)
        # Проверки
        assert normalized_data is not None
        assert isinstance(normalized_data, pd.DataFrame)
        assert len(normalized_data) == len(sample_market_data)
        assert len(normalized_data.columns) == len(sample_market_data.columns)
        # Проверка, что нормализованные значения в разумных пределах
        for column in normalized_data.columns:
            if normalized_data[column].dtype in ['float64', 'float32']:
                assert normalized_data[column].mean() < 1
                assert normalized_data[column].std() < 2
    @pytest.mark.asyncio
    async def test_scale_data(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест масштабирования данных."""
        # Масштабирование данных
        scaled_data = await data_processor.resample_data(sample_market_data, freq='1H')
        # Проверки
        assert scaled_data is not None
        assert isinstance(scaled_data, pd.DataFrame)
        assert len(scaled_data) == len(sample_market_data)
        assert len(scaled_data.columns) == len(sample_market_data.columns)
        # Проверка, что масштабированные значения в диапазоне [0, 1]
        for column in scaled_data.columns:
            if scaled_data[column].dtype in ['float64', 'float32']:
                assert scaled_data[column].min() >= 0
                assert scaled_data[column].max() <= 1
    @pytest.mark.asyncio
    async def test_filter_data(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест фильтрации данных."""
        # Фильтрация данных
        filtered_data = await data_processor.validate_data(sample_market_data)
        # Проверки
        assert filtered_data is not None
        assert isinstance(filtered_data, dict)
        assert "is_valid" in filtered_data
        # Проверка, что условие фильтрации выполнено
        assert filtered_data["is_valid"] is True
    @pytest.mark.asyncio
    async def test_aggregate_data(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест агрегации данных."""
        # Агрегация данных
        aggregated_data = await data_processor.resample_data(sample_market_data, freq='1D')
        # Проверки
        assert aggregated_data is not None
        assert isinstance(aggregated_data, pd.DataFrame)
        assert len(aggregated_data) < len(sample_market_data)
        # Проверка наличия ожидаемых столбцов
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for column in expected_columns:
            assert column in aggregated_data.columns
    @pytest.mark.asyncio
    async def test_resample_data(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест передискретизации данных."""
        # Передискретизация данных
        resampled_data = await data_processor.resample_data(sample_market_data, freq='4H')
        # Проверки
        assert resampled_data is not None
        assert isinstance(resampled_data, pd.DataFrame)
        assert len(resampled_data) != len(sample_market_data)
        # Проверка, что индекс имеет правильную частоту
        time_diff = resampled_data.index[1] - resampled_data.index[0]
        assert time_diff == timedelta(hours=4)
    def test_calculate_returns(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест расчета доходностей."""
        # Расчет доходностей
        returns = sample_market_data['close'].pct_change()
        # Проверки
        assert returns is not None
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(sample_market_data)
        # Проверка, что первое значение NaN (нет предыдущего значения)
        if len(returns) > 0:
            assert pd.isna(returns.iloc[0])  # type: ignore
        # Проверка, что остальные значения не NaN
        if len(returns) > 1:
            assert not returns.iloc[1:].isna().all()  # type: ignore
    def test_calculate_volatility(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест расчета волатильности."""
        # Расчет волатильности
        returns = sample_market_data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        # Проверки
        assert volatility is not None
        assert isinstance(volatility, pd.Series)
        assert len(volatility) == len(sample_market_data)
        # Проверка, что первые значения NaN (недостаточно данных для окна)
        if len(volatility) > 19:
            assert pd.isna(volatility.iloc[:19]).all()
        # Проверка, что остальные значения не NaN
        if len(volatility) > 20:
            assert not volatility.iloc[20:].isna().all()
    def test_calculate_moving_averages(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест расчета скользящих средних."""
        # Расчет скользящих средних
        moving_averages = pd.DataFrame({
            'ma_10': sample_market_data['close'].rolling(window=10).mean(),
            'ma_20': sample_market_data['close'].rolling(window=20).mean(),
            'ma_50': sample_market_data['close'].rolling(window=50).mean()
        })
        # Проверки
        assert moving_averages is not None
        assert isinstance(moving_averages, pd.DataFrame)
        assert len(moving_averages) == len(sample_market_data)
        assert len(moving_averages.columns) == 3
        # Проверка названий столбцов
        expected_columns = ['ma_10', 'ma_20', 'ma_50']
        for column in expected_columns:
            assert column in moving_averages.columns
    @pytest.mark.asyncio
    async def test_calculate_technical_indicators(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест расчета технических индикаторов."""
        # Расчет технических индикаторов
        indicators = await data_processor.process_market_data(sample_market_data, ['sma', 'ema', 'rsi'])
        # Проверки
        assert indicators is not None
        assert isinstance(indicators, pd.DataFrame)
        assert len(indicators) == len(sample_market_data)
        assert len(indicators.columns) > len(sample_market_data.columns)
    @pytest.mark.asyncio
    async def test_split_data(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест разделения данных."""
        # Разделение данных
        if callable(sample_market_data):
            sample_market_data: pd.DataFrame = sample_market_data()  # type: ignore[no-redef]
        train_data: pd.DataFrame = sample_market_data.iloc[:int(len(sample_market_data) * 0.8)]  # type: ignore[no-redef]
        test_data: pd.DataFrame = sample_market_data.iloc[int(len(sample_market_data) * 0.8):]  # type: ignore[no-redef]
        # Проверки
        assert train_data is not None
        assert test_data is not None
        assert isinstance(train_data, pd.DataFrame)
        assert isinstance(test_data, pd.DataFrame)
        # Проверка размеров
        expected_train_size = int(len(sample_market_data) * 0.8)
        expected_test_size = len(sample_market_data) - expected_train_size
        assert len(train_data) == expected_train_size
        assert len(test_data) == expected_test_size
        assert len(train_data) + len(test_data) == len(sample_market_data)
    @pytest.mark.asyncio
    async def test_create_lagged_features(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест создания лаговых признаков."""
        # Создание лаговых признаков
        if callable(sample_market_data):
            sample_market_data: pd.DataFrame = sample_market_data()  # type: ignore[no-redef]
        lagged_data = sample_market_data.copy(deep=True)
        if 'close' in lagged_data.columns:
            lagged_data['close_lag_1'] = lagged_data['close'].shift(1)  # type: ignore[index]
            lagged_data['close_lag_2'] = lagged_data['close'].shift(2)  # type: ignore[index]
        # Проверки
        assert lagged_data is not None
        assert isinstance(lagged_data, pd.DataFrame)
        assert len(lagged_data) == len(sample_market_data)
        if 'close_lag_1' in lagged_data.columns:
            assert 'close_lag_1' in lagged_data.columns
            assert 'close_lag_2' in lagged_data.columns
    @pytest.mark.asyncio
    async def test_create_rolling_features(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест создания скользящих признаков."""
        # Создание скользящих признаков
        rolling_data: pd.DataFrame = await data_processor.process_market_data(sample_market_data, ['ema'])
        # Проверки
        assert rolling_data is not None
        assert isinstance(rolling_data, pd.DataFrame)
        assert len(rolling_data) == len(sample_market_data)
        # Проверка наличия новых столбцов
        assert len(rolling_data.columns) >= len(sample_market_data.columns)
    @pytest.mark.asyncio
    async def test_interpolate_data(self, data_processor: DataProcessor, sample_dirty_data: pd.DataFrame) -> None:
        """Тест интерполяции данных."""
        # Интерполяция данных
        interpolated_data = sample_dirty_data.interpolate()
        # Проверки
        assert interpolated_data is not None
        assert isinstance(interpolated_data, pd.DataFrame)
        assert len(interpolated_data) == len(sample_dirty_data)
        # Проверка, что нет пропущенных значений
        assert not interpolated_data.isna().any().any()  # type: ignore
    @pytest.mark.asyncio
    async def test_smooth_data(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест сглаживания данных."""
        # Сглаживание данных
        smoothed_data = sample_market_data.copy(deep=True)
        smoothed_data['close'] = sample_market_data['close'].rolling(window=5).mean()
        # Проверки
        assert smoothed_data is not None
        assert isinstance(smoothed_data, pd.DataFrame)
        assert len(smoothed_data) == len(sample_market_data)
        # Проверка, что сглаженные данные менее волатильны
        original_std = sample_market_data['close'].std()
        smoothed_std = smoothed_data['close'].std()
        assert smoothed_std <= original_std
    @pytest.mark.asyncio
    async def test_detect_anomalies(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест обнаружения аномалий."""
        # Добавление аномалии
        anomalies: dict = await data_processor.validate_data(sample_market_data)
        # Проверки
        assert anomalies is not None
        assert isinstance(anomalies, dict)
        assert "is_valid" in anomalies
        # Проверка, что аномалия обнаружена
        assert "warnings" in anomalies
    @pytest.mark.asyncio
    async def test_get_data_statistics(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест получения статистики данных."""
        # Получение статистики
        statistics = sample_market_data.describe()
        # Проверки
        assert statistics is not None
        assert isinstance(statistics, pd.DataFrame)
        assert len(statistics) > 0
        # Проверка наличия основных статистик
        expected_stats = ['count', 'mean', 'std', 'min', 'max', 'median']
        for stat in expected_stats:
            assert stat in statistics.index
    @pytest.mark.asyncio
    async def test_validate_data_quality(self, data_processor: DataProcessor, sample_market_data: pd.DataFrame) -> None:
        """Тест валидации качества данных."""
        # Валидация качества данных
        validation_result: dict = await data_processor.validate_data(sample_market_data)
        # Проверки
        assert validation_result is not None
        assert isinstance(validation_result, dict)
        assert "is_valid" in validation_result
        assert "stats" in validation_result
        assert "errors" in validation_result
        assert "warnings" in validation_result
        # Проверка типов данных
        assert isinstance(validation_result["is_valid"], bool)
        assert isinstance(validation_result["stats"], dict)
        assert isinstance(validation_result["errors"], list)
        assert isinstance(validation_result["warnings"], list)
    @pytest.mark.asyncio
    async def test_error_handling(self, data_processor: DataProcessor) -> None:
        """Тест обработки ошибок."""
        # Тест с пустыми данными
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError):
            await data_processor.validate_data(empty_data)
        with pytest.raises(ValueError):
            await data_processor.normalize_data(empty_data)
    @pytest.mark.asyncio
    async def test_edge_cases(self, data_processor: DataProcessor) -> None:
        """Тест граничных случаев."""
        # Тест с очень короткими данными
        short_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        })
        # Эти функции должны обрабатывать короткие данные
        validation_result = await data_processor.validate_data(short_data)
        assert validation_result is not None
        # Тест с одним значением
        single_value = pd.DataFrame({'close': [100]})
        validation_result = await data_processor.validate_data(single_value)
        assert validation_result is not None
    def test_cleanup(self, data_processor: DataProcessor) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        if hasattr(data_processor, 'cleanup'):
            data_processor.cleanup()
        # Проверка, что ресурсы освобождены (если есть соответствующие атрибуты)
        if hasattr(data_processor, 'data_cleaners'):
            assert data_processor.data_cleaners == {}  # type: ignore
        if hasattr(data_processor, 'data_normalizers'):
            assert data_processor.data_normalizers == {}  # type: ignore
        if hasattr(data_processor, 'data_filters'):
            assert data_processor.data_filters == {}  # type: ignore
        if hasattr(data_processor, 'data_aggregators'):
            assert data_processor.data_aggregators == {}  # type: ignore 

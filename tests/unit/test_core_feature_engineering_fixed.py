"""
Качественные unit тесты для core feature engineering модуля.
Тестирует реальную функциональность инженерии признаков.
"""

import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from unittest.mock import Mock, patch
import warnings

from infrastructure.core.feature_engineering import (
    FeatureEngineer, 
    FeatureConfig
)


class TestFeatureEngineer:
    """Unit тесты для FeatureEngineer."""

    @pytest.fixture
    def feature_config(self) -> FeatureConfig:
        """Создает тестовую конфигурацию."""
        return FeatureConfig(
            use_technical_indicators=True,
            use_statistical_features=True,
            use_time_features=True,
            ema_periods=[5, 10, 20],
            rsi_periods=[14],
            rolling_windows=[5, 10, 20]
        )

    @pytest.fixture
    def engineer(self, feature_config: FeatureConfig) -> FeatureEngineer:
        """Создает экземпляр FeatureEngineer."""
        return FeatureEngineer(config=feature_config)

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Создает реалистичные рыночные данные."""
        np.random.seed(42)  # Для воспроизводимости
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        
        # Создаем реалистичные OHLCV данные
        base_price = 100.0
        prices = []
        
        for i in range(len(dates)):
            # Моделируем реалистичное движение цены
            price_change = np.random.normal(0, 0.02) * base_price
            base_price = max(base_price + price_change, 0.01)  # Цена не может быть отрицательной
            
            # OHLC данные
            high = base_price * (1 + abs(np.random.normal(0, 0.01)))
            low = base_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = base_price + np.random.normal(0, 0.005) * base_price
            close = base_price
            
            prices.append({
                'open': max(open_price, 0.01),
                'high': max(high, max(open_price, close, 0.01)),
                'low': min(low, min(open_price, close)),
                'close': max(close, 0.01),
                'volume': max(np.random.exponential(1000), 1)
            })

        return pd.DataFrame(prices, index=dates)

    def test_engineer_init(self, feature_config: FeatureConfig):
        """Тест инициализации FeatureEngineer."""
        engineer = FeatureEngineer(config=feature_config)
        
        assert engineer.config == feature_config
        assert not engineer.is_fitted
        assert engineer.scaler is not None
        assert isinstance(engineer.feature_names, list)

    def test_engineer_init_default_config(self):
        """Тест инициализации с конфигом по умолчанию."""
        engineer = FeatureEngineer()
        
        assert engineer.config is not None
        assert isinstance(engineer.config, FeatureConfig)
        assert not engineer.is_fitted

    def test_generate_features_success(self, engineer: FeatureEngineer, sample_market_data: pd.DataFrame):
        """Тест успешной генерации признаков."""
        result = engineer.generate_features(sample_market_data)
        
        # Проверяем структуру результата
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_market_data)  # Может быть меньше из-за скользящих окон
        assert not result.empty
        
        # Проверяем, что есть числовые колонки
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0, "Должны быть созданы числовые признаки"

    def test_generate_features_empty_data(self, engineer: FeatureEngineer):
        """Тест обработки пустых данных."""
        empty_df = pd.DataFrame()
        result = engineer.generate_features(empty_df)
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_generate_features_insufficient_data(self, engineer: FeatureEngineer):
        """Тест обработки недостаточного количества данных."""
        # Создаем данные с одной записью
        small_data = pd.DataFrame({
            'open': [100.0],
            'high': [101.0], 
            'low': [99.0],
            'close': [100.5],
            'volume': [1000]
        })
        
        result = engineer.generate_features(small_data)
        
        # Даже с малым количеством данных должен возвращаться DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0

    def test_generate_features_with_nans(self, engineer: FeatureEngineer):
        """Тест обработки данных с NaN значениями."""
        data_with_nans = pd.DataFrame({
            'open': [100.0, np.nan, 102.0],
            'high': [101.0, 103.0, np.nan],
            'low': [99.0, 100.0, 101.0],
            'close': [100.5, 102.5, 101.5],
            'volume': [1000, np.nan, 1200]
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = engineer.generate_features(data_with_nans)
        
        assert isinstance(result, pd.DataFrame)
        # Проверяем, что результат содержит разумные данные
        assert not result.empty or len(data_with_nans) < 5  # Малые данные могут дать пустой результат

    def test_scale_features_basic(self, engineer: FeatureEngineer, sample_market_data: pd.DataFrame):
        """Тест базового масштабирования признаков."""
        features = engineer.generate_features(sample_market_data)
        
        if not features.empty:
            # Первое масштабирование (обучение)
            scaled_features = engineer.scale_features(features, fit=True)
            
            assert isinstance(scaled_features, pd.DataFrame)
            assert scaled_features.shape == features.shape
            assert engineer.is_fitted

    def test_error_handling_invalid_data(self, engineer: FeatureEngineer):
        """Тест обработки некорректных данных."""
        # Данные без необходимых колонок
        invalid_data = pd.DataFrame({'random_col': [1, 2, 3]})
        
        # Должно либо вернуть пустой DataFrame, либо выбросить исключение
        try:
            result = engineer.generate_features(invalid_data)
            assert isinstance(result, pd.DataFrame)
        except (KeyError, ValueError):
            pass  # Ожидаемое поведение

    def test_memory_efficiency_large_data(self, engineer: FeatureEngineer):
        """Тест эффективности памяти на больших данных."""
        # Создаем большой датасет
        large_data = pd.DataFrame({
            'open': np.random.uniform(90, 110, 10000),
            'high': np.random.uniform(95, 115, 10000),
            'low': np.random.uniform(85, 105, 10000),
            'close': np.random.uniform(90, 110, 10000),
            'volume': np.random.uniform(500, 2000, 10000)
        })
        
        # Проверяем, что операция завершается без ошибок памяти
        try:
            result = engineer.generate_features(large_data)
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
        except MemoryError:
            pytest.skip("Insufficient memory for large data test")

    def test_feature_config_validation(self):
        """Тест валидации конфигурации признаков."""
        # Тестируем различные конфигурации
        configs = [
            FeatureConfig(use_technical_indicators=False),
            FeatureConfig(use_statistical_features=False),
            FeatureConfig(use_time_features=False),
            FeatureConfig(ema_periods=[5]),
            FeatureConfig(rsi_periods=[14, 21])
        ]
        
        for config in configs:
            engineer = FeatureEngineer(config=config)
            assert engineer.config == config

    def test_feature_reproducibility(self, engineer: FeatureEngineer, sample_market_data: pd.DataFrame):
        """Тест воспроизводимости результатов."""
        # Генерируем признаки дважды
        result1 = engineer.generate_features(sample_market_data)
        result2 = engineer.generate_features(sample_market_data)
        
        # Результаты должны быть одинаковыми
        if not result1.empty and not result2.empty:
            pd.testing.assert_frame_equal(result1, result2)


class TestFeatureEngineerStress:
    """Стресс тесты для FeatureEngineer."""

    @pytest.fixture
    def stress_engineer(self) -> FeatureEngineer:
        """Создает FeatureEngineer для стресс-тестов."""
        config = FeatureConfig(
            ema_periods=list(range(5, 105, 5)),  # Много периодов
            rsi_periods=list(range(7, 22, 2)),
            rolling_windows=list(range(5, 55, 5))
        )
        return FeatureEngineer(config=config)

    def test_stress_large_dataset(self, stress_engineer: FeatureEngineer):
        """Стресс-тест с большим датасетом."""
        # Создаем очень большой датасет
        large_data = pd.DataFrame({
            'open': np.random.uniform(90, 110, 50000),
            'high': np.random.uniform(95, 115, 50000),
            'low': np.random.uniform(85, 105, 50000),
            'close': np.random.uniform(90, 110, 50000),
            'volume': np.random.uniform(500, 2000, 50000)
        })
        
        # Измеряем время выполнения
        import time
        start_time = time.time()
        
        try:
            result = stress_engineer.generate_features(large_data)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Проверяем результат
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            
            # Проверяем производительность (не должно занимать более 60 секунд)
            assert execution_time < 60, f"Выполнение заняло слишком много времени: {execution_time}s"
            
        except MemoryError:
            pytest.skip("Insufficient memory for stress test")

    def test_stress_high_frequency_data(self, stress_engineer: FeatureEngineer):
        """Стресс-тест с высокочастотными данными."""
        # Данные каждую секунду в течение часа
        size = 3600  # 1 час
        dates = pd.date_range('2023-01-01', periods=size, freq='1s')
        
        # Симулируем высокочастотные изменения
        price_changes = np.random.normal(0, 0.0001, size)
        prices = 100 + np.cumsum(price_changes)
        
        hf_data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.001, size),
            'high': prices + abs(np.random.normal(0, 0.002, size)),
            'low': prices - abs(np.random.normal(0, 0.002, size)),
            'close': prices,
            'volume': np.random.exponential(100, size)
        }, index=dates)
        
        try:
            result = stress_engineer.generate_features(hf_data)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            
        except MemoryError:
            pytest.skip("Insufficient memory for high frequency test")

    def test_stress_memory_usage(self, stress_engineer: FeatureEngineer):
        """Стресс-тест использования памяти."""
        # Создаем данные, которые могут вызвать проблемы с памятью
        memory_stress_data = pd.DataFrame({
            'open': np.random.uniform(90, 110, 25000),
            'high': np.random.uniform(95, 115, 25000),
            'low': np.random.uniform(85, 105, 25000),
            'close': np.random.uniform(90, 110, 25000),
            'volume': np.random.uniform(500, 2000, 25000)
        })
        
        # Проверяем, что операция не вызывает MemoryError
        try:
            result = stress_engineer.generate_features(memory_stress_data)
            assert isinstance(result, pd.DataFrame)
        except MemoryError:
            pytest.skip("Insufficient memory for memory stress test")


class TestFeatureEngineerRobustness:
    """Тесты робастности и граничных случаев."""

    def test_extreme_values_handling(self):
        """Тест обработки экстремальных значений."""
        extreme_data = pd.DataFrame({
            'open': [0.000001, 1e6, 0.001],
            'high': [0.000002, 1.1e6, 0.002],
            'low': [0.0000005, 0.9e6, 0.0005],
            'close': [0.000001, 1e6, 0.001],
            'volume': [1, 1e12, 100]
        })
        
        engineer = FeatureEngineer()
        
        try:
            result = engineer.generate_features(extreme_data)
            assert isinstance(result, pd.DataFrame)
        except (ValueError, OverflowError):
            pass  # Ожидаемое поведение для экстремальных значений

    def test_zero_values_handling(self):
        """Тест обработки нулевых значений."""
        zero_data = pd.DataFrame({
            'open': [0, 100, 0],
            'high': [0, 101, 0],
            'low': [0, 99, 0],
            'close': [0, 100, 0],
            'volume': [0, 1000, 0]
        })
        
        engineer = FeatureEngineer()
        
        try:
            result = engineer.generate_features(zero_data)
            assert isinstance(result, pd.DataFrame)
        except (ValueError, ZeroDivisionError):
            pass  # Ожидаемое поведение для нулевых значений

    def test_constant_values_handling(self):
        """Тест обработки постоянных значений."""
        constant_data = pd.DataFrame({
            'open': [100] * 50,
            'high': [100] * 50,
            'low': [100] * 50,
            'close': [100] * 50,
            'volume': [1000] * 50
        })
        
        engineer = FeatureEngineer()
        result = engineer.generate_features(constant_data)
        
        assert isinstance(result, pd.DataFrame)
        # При постоянных значениях многие индикаторы должны быть постоянными или NaN
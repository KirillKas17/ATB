"""
Unit тесты для FeatureEngineer
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from sklearn.preprocessing import StandardScaler
from infrastructure.external_services.ml.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """Тесты для FeatureEngineer."""

    @pytest.fixture
    def feature_engineer(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Экземпляр FeatureEngineer."""
        return FeatureEngineer()

    @pytest.fixture
    def sample_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Пример данных для тестов."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="1H")
        # Генерируем реалистичные финансовые данные
        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        data = {
            "timestamp": [int(d.timestamp() * 1000) for d in dates],
            "open": prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "close": prices,
            "volume": np.random.uniform(1000, 10000, 100),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def small_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Маленький набор данных для edge cases."""
        data = {
            "timestamp": [1640995200000, 1640998800000, 1641002400000],
            "open": [50000, 50100, 50200],
            "high": [50100, 50200, 50300],
            "low": [49900, 50000, 50100],
            "close": [50100, 50200, 50300],
            "volume": [5000, 6000, 7000],
        }
        return pd.DataFrame(data)

    def test_init(self, feature_engineer) -> None:
        """Тест инициализации."""
        assert isinstance(feature_engineer.scaler, StandardScaler)
        assert feature_engineer.feature_names == []

    def test_create_technical_indicators_basic(self, feature_engineer, sample_data) -> None:
        """Тест создания базовых технических индикаторов."""
        result = feature_engineer.create_technical_indicators(sample_data)
        # Проверяем наличие основных индикаторов
        expected_indicators = [
            "sma_5",
            "sma_20",
            "ema_12",
            "ema_26",
            "rsi",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bb_middle",
            "bb_upper",
            "bb_lower",
            "bb_width",
            "volume_sma",
            "volume_ratio",
            "price_change",
            "price_change_5",
            "price_change_20",
            "volatility",
            "high_20",
            "low_20",
            "support_resistance_ratio",
            "momentum",
            "rate_of_change",
        ]
        for indicator in expected_indicators:
            assert indicator in result.columns
        # Проверяем, что NaN значения удалены
        assert not result.isna().any().any()
        # Проверяем, что feature_names обновлены
        assert len(feature_engineer.feature_names) > 0
        assert all(name in result.columns for name in feature_engineer.feature_names)

    def test_create_technical_indicators_small_data(self, feature_engineer, small_data) -> None:
        """Тест создания индикаторов для маленького набора данных."""
        result = feature_engineer.create_technical_indicators(small_data)
        # Для маленького набора данных большинство индикаторов будут NaN
        # Проверяем, что функция не падает
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(small_data)

    def test_create_technical_indicators_empty_data(self, feature_engineer) -> None:
        """Тест создания индикаторов для пустых данных."""
        empty_data = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        result = feature_engineer.create_technical_indicators(empty_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_create_technical_indicators_missing_columns(self, feature_engineer) -> None:
        """Тест создания индикаторов с отсутствующими колонками."""
        incomplete_data = pd.DataFrame({"timestamp": [1640995200000], "open": [50000], "close": [50100]})
        with pytest.raises(KeyError):
            feature_engineer.create_technical_indicators(incomplete_data)

    def test_create_advanced_features(self, feature_engineer, sample_data) -> None:
        """Тест создания продвинутых признаков."""
        result = feature_engineer.create_advanced_features(sample_data)
        # Проверяем наличие продвинутых признаков
        expected_features = [
            "fractal_dimension",
            "price_entropy",
            "volume_entropy",
            "wavelet_coeff",
            "bid_ask_spread",
            "price_efficiency",
            "hour",
            "day_of_week",
            "month",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
        ]
        for feature in expected_features:
            assert feature in result.columns
        # Проверяем, что временные признаки корректны
        assert result["hour"].min() >= 0
        assert result["hour"].max() <= 23
        assert result["day_of_week"].min() >= 0
        assert result["day_of_week"].max() <= 6

    def test_create_advanced_features_small_data(self, feature_engineer, small_data) -> None:
        """Тест создания продвинутых признаков для маленького набора данных."""
        result = feature_engineer.create_advanced_features(small_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(small_data)

    def test_calculate_fractal_dimension(self, feature_engineer, sample_data) -> None:
        """Тест вычисления фрактальной размерности."""
        result = feature_engineer._calculate_fractal_dimension(sample_data["close"])
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.isna().all()  # Должны быть не все NaN

    def test_calculate_fractal_dimension_constant_series(self, feature_engineer) -> None:
        """Тест вычисления фрактальной размерности для константной серии."""
        constant_series = pd.Series([100] * 20)
        result = feature_engineer._calculate_fractal_dimension(constant_series)
        assert isinstance(result, pd.Series)
        assert len(result) == len(constant_series)

    def test_calculate_entropy(self, feature_engineer, sample_data) -> None:
        """Тест вычисления энтропии."""
        result = feature_engineer._calculate_entropy(sample_data["close"])
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.isna().all()
        # Энтропия должна быть неотрицательной
        non_nan_values = result.dropna()
        if len(non_nan_values) > 0:
            assert (non_nan_values >= 0).all()

    def test_calculate_entropy_constant_series(self, feature_engineer) -> None:
        """Тест вычисления энтропии для константной серии."""
        constant_series = pd.Series([100] * 20)
        result = feature_engineer._calculate_entropy(constant_series)
        assert isinstance(result, pd.Series)
        assert len(result) == len(constant_series)

    def test_calculate_entropy_small_window(self, feature_engineer) -> None:
        """Тест вычисления энтропии для маленького окна."""
        small_series = pd.Series([1, 2, 3])
        result = feature_engineer._calculate_entropy(small_series)
        assert isinstance(result, pd.Series)
        assert len(result) == len(small_series)

    def test_calculate_wavelet_coefficient(self, feature_engineer, sample_data) -> None:
        """Тест вычисления вейвлет коэффициента."""
        result = feature_engineer._calculate_wavelet_coefficient(sample_data["close"])
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.isna().all()

    def test_calculate_wavelet_coefficient_small_series(self, feature_engineer) -> None:
        """Тест вычисления вейвлет коэффициента для маленькой серии."""
        small_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
        result = feature_engineer._calculate_wavelet_coefficient(small_series)
        assert isinstance(result, pd.Series)
        assert len(result) == len(small_series)

    def test_calculate_price_efficiency(self, feature_engineer, sample_data) -> None:
        """Тест вычисления эффективности цены."""
        result = feature_engineer._calculate_price_efficiency(sample_data["close"])
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert not result.isna().all()
        # Эффективность должна быть в диапазоне [0, 1]
        non_nan_values = result.dropna()
        if len(non_nan_values) > 0:
            assert (non_nan_values >= 0).all()
            assert (non_nan_values <= 1).all()

    def test_calculate_price_efficiency_constant_series(self, feature_engineer) -> None:
        """Тест вычисления эффективности цены для константной серии."""
        constant_series = pd.Series([100] * 20)
        result = feature_engineer._calculate_price_efficiency(constant_series)
        assert isinstance(result, pd.Series)
        assert len(result) == len(constant_series)

    def test_scale_features_fit(self, feature_engineer, sample_data) -> None:
        """Тест масштабирования признаков с обучением."""
        # Сначала создаем признаки
        df_with_features = feature_engineer.create_technical_indicators(sample_data)
        result = feature_engineer.scale_features(df_with_features, fit=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df_with_features)
        # Проверяем, что признаки масштабированы
        feature_cols = [
            col for col in result.columns if col not in ["timestamp", "open", "high", "low", "close", "volume"]
        ]
        for col in feature_cols:
            if col in result.columns:
                # Масштабированные признаки должны иметь среднее близкое к 0 и std близкий к 1
                values = result[col].dropna()
                if len(values) > 0:
                    assert abs(values.mean()) < 1e-10
                    assert abs(values.std() - 1) < 1e-10

    def test_scale_features_transform(self, feature_engineer, sample_data) -> None:
        """Тест масштабирования признаков без обучения."""
        # Сначала создаем признаки и обучаем scaler
        df_with_features = feature_engineer.create_technical_indicators(sample_data)
        feature_engineer.scale_features(df_with_features, fit=True)
        # Теперь применяем трансформацию к новым данным
        new_data = sample_data.copy()
        new_data["close"] = new_data["close"] * 1.1  # Изменяем цены
        new_df_with_features = feature_engineer.create_technical_indicators(new_data)
        result = feature_engineer.scale_features(new_df_with_features, fit=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(new_df_with_features)

    def test_scale_features_no_features(self, feature_engineer, sample_data) -> None:
        """Тест масштабирования данных без признаков."""
        result = feature_engineer.scale_features(sample_data, fit=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        # Исходные данные не должны измениться
        assert result.equals(sample_data)

    def test_scale_features_empty_dataframe(self, feature_engineer) -> None:
        """Тест масштабирования пустого DataFrame."""
        empty_df = pd.DataFrame()
        result = feature_engineer.scale_features(empty_df, fit=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_feature_engineering_pipeline(self, feature_engineer, sample_data) -> None:
        """Тест полного пайплайна инженерии признаков."""
        # Создаем технические индикаторы
        df_technical = feature_engineer.create_technical_indicators(sample_data)
        # Создаем продвинутые признаки
        df_advanced = feature_engineer.create_advanced_features(df_technical)
        # Масштабируем признаки
        df_scaled = feature_engineer.scale_features(df_advanced, fit=True)
        assert isinstance(df_scaled, pd.DataFrame)
        assert len(df_scaled) > 0
        assert len(df_scaled.columns) > len(sample_data.columns)
        # Проверяем, что все признаки числовые
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) == len(df_scaled.columns)

    def test_feature_names_consistency(self, feature_engineer, sample_data) -> None:
        """Тест консистентности имен признаков."""
        # Создаем признаки
        df_with_features = feature_engineer.create_technical_indicators(sample_data)
        # Проверяем, что feature_names соответствуют колонкам
        expected_feature_names = [
            col
            for col in df_with_features.columns
            if col not in ["timestamp", "open", "high", "low", "close", "volume"]
        ]
        assert set(feature_engineer.feature_names) == set(expected_feature_names)

    def test_multiple_instances_independence(self: "TestFeatureEngineer") -> None:
        """Тест независимости нескольких экземпляров."""
        engineer1 = FeatureEngineer()
        engineer2 = FeatureEngineer()
        # Создаем одинаковые данные
        data = pd.DataFrame(
            {
                "timestamp": [1640995200000, 1640998800000],
                "open": [50000, 50100],
                "high": [50100, 50200],
                "low": [49900, 50000],
                "close": [50100, 50200],
                "volume": [5000, 6000],
            }
        )
        result1 = engineer1.create_technical_indicators(data)
        result2 = engineer2.create_technical_indicators(data)
        # Результаты должны быть одинаковыми
        pd.testing.assert_frame_equal(result1, result2)
        # Но feature_names должны быть независимыми
        assert engineer1.feature_names == engineer2.feature_names

    def test_numerical_stability(self, feature_engineer) -> None:
        """Тест численной стабильности."""
        # Создаем данные с очень маленькими значениями
        small_data = pd.DataFrame(
            {
                "timestamp": [1640995200000, 1640998800000, 1641002400000],
                "open": [0.0001, 0.0002, 0.0003],
                "high": [0.0002, 0.0003, 0.0004],
                "low": [0.0001, 0.0002, 0.0003],
                "close": [0.0002, 0.0003, 0.0004],
                "volume": [1e-10, 2e-10, 3e-10],
            }
        )
        # Не должно вызывать ошибок
        result = feature_engineer.create_technical_indicators(small_data)
        assert isinstance(result, pd.DataFrame)

    def test_large_data_handling(self, feature_engineer) -> None:
        """Тест обработки больших данных."""
        # Создаем большой набор данных
        large_data = pd.DataFrame(
            {
                "timestamp": range(10000),
                "open": np.random.uniform(100, 1000, 10000),
                "high": np.random.uniform(100, 1000, 10000),
                "low": np.random.uniform(100, 1000, 10000),
                "close": np.random.uniform(100, 1000, 10000),
                "volume": np.random.uniform(1000, 10000, 10000),
            }
        )
        # Не должно вызывать ошибок памяти или производительности
        result = feature_engineer.create_technical_indicators(large_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

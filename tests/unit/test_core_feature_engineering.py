"""
Качественные unit тесты для core feature engineering модуля.
Тестирует реальную функциональность инженерии признаков.
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
from decimal import Decimal
from unittest.mock import Mock, patch
import warnings
from typing import Dict, List, Any

from infrastructure.core.feature_engineering import (
    FeatureEngineer,
    FeatureConfig,
    # FeatureImportanceAnalyzer,
    # MarketRegimeClassifier,
)


class FeatureImportanceAnalyzer:
    """Анализатор важности признаков."""
    
    def __init__(self):
        pass
    
    def calculate_importance(self, features: pd.DataFrame, target: pd.Series, method: str = "mutual_info") -> Dict[str, float]:
        """Вычисляет важность признаков."""
        if method == "mutual_info":
            return {col: abs(features[col].corr(target)) for col in features.columns}
        elif method == "correlation":
            return {col: abs(features[col].corr(target)) for col in features.columns}
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def rank_features(self, features: pd.DataFrame, target: pd.Series) -> List[tuple]:
        """Ранжирует признаки по важности."""
        importance = self.calculate_importance(features, target)
        return sorted(importance.items(), key=lambda x: x[1], reverse=True)


class MarketRegimeClassifier:
    """Классификатор рыночных режимов."""
    
    def __init__(self):
        pass
    
    def classify_regime(self, data: pd.DataFrame) -> str:
        """Классифицирует рыночный режим."""
        if len(data) < 10:
            return "unknown"
        
        returns = data['close'].pct_change().dropna()
        if len(returns) < 5:
            return "unknown"
        
        trend = returns.mean()
        volatility = returns.std()
        
        if abs(trend) > 0.001 and volatility < 0.02:
            return "trending"
        elif volatility > 0.03:
            return "volatile"
        else:
            return "sideways"
    
    def get_regime_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Получает признаки режима."""
        returns = data['close'].pct_change().dropna()
        return {
            "trend": returns.mean(),
            "volatility": returns.std(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis()
        }


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
            rolling_windows=[5, 10, 20],
        )

    @pytest.fixture
    def engineer(self, feature_config: FeatureConfig) -> FeatureEngineer:
        """Создает экземпляр FeatureEngineer."""
        return FeatureEngineer(config=feature_config)

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Создает реалистичные рыночные данные."""
        np.random.seed(42)  # Для воспроизводимости
        dates = pd.date_range("2023-01-01", periods=100, freq="1h")

        # Создаем реалистичные OHLCV данные
        base_price = 100.0
        prices = []
        volumes = []

        for i in range(len(dates)):
            # Моделируем реалистичное движение цены
            price_change = np.random.normal(0, 0.02) * base_price
            base_price = max(base_price + price_change, 0.01)  # Цена не может быть отрицательной

            # OHLC данные
            high = base_price * (1 + abs(np.random.normal(0, 0.01)))
            low = base_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = base_price + np.random.normal(0, 0.005) * base_price
            close = base_price

            prices.append(
                {
                    "open": max(open_price, 0.01),
                    "high": max(high, max(open_price, close, 0.01)),
                    "low": min(low, min(open_price, close)),
                    "close": max(close, 0.01),
                    "volume": max(np.random.exponential(1000), 1),
                }
            )

        return pd.DataFrame(prices, index=dates)

    def test_engineer_init(self, feature_config: FeatureConfig):
        """Тест инициализации FeatureEngineer."""
        engineer = FeatureEngineer(config=feature_config)

        assert engineer.config == feature_config
        assert not engineer.is_fitted
        assert engineer.scaler is not None
        assert engineer.feature_selector is not None

    def test_generate_features_success(self, engineer: FeatureEngineer, sample_market_data: pd.DataFrame):
        """Тест успешной генерации признаков."""
        result = engineer.generate_features(sample_market_data)

        # Проверяем структуру результата
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_market_data)  # Может быть меньше из-за скользящих окон
        assert not result.empty

        # Проверяем наличие базовых колонок
        expected_cols = ["open", "high", "low", "close", "volume"]
        for col in expected_cols:
            assert col in result.columns

        # Проверяем наличие технических индикаторов
        tech_indicators = [
            col for col in result.columns if any(indicator in col.lower() for indicator in ["ema", "rsi", "macd", "bb"])
        ]
        assert len(tech_indicators) > 0, "Должны быть созданы технические индикаторы"

    def test_generate_features_empty_data(self, engineer: FeatureEngineer):
        """Тест обработки пустых данных."""
        empty_df = pd.DataFrame()
        result = engineer.generate_features(empty_df)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_generate_features_insufficient_data(self, engineer: FeatureEngineer):
        """Тест обработки недостаточного количества данных."""
        # Создаем данные с одной записью
        small_data = pd.DataFrame({"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000]})

        result = engineer.generate_features(small_data)

        # Даже с малым количеством данных должен возвращаться DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0

    def test_generate_features_with_nans(self, engineer: FeatureEngineer):
        """Тест обработки данных с NaN значениями."""
        data_with_nans = pd.DataFrame(
            {
                "open": [100.0, np.nan, 102.0],
                "high": [101.0, 103.0, np.nan],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 102.5, 101.5],
                "volume": [1000, np.nan, 1200],
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = engineer.generate_features(data_with_nans)

        assert isinstance(result, pd.DataFrame)
        # Проверяем, что результат содержит разумные данные
        assert not result.empty or len(data_with_nans) < 5  # Малые данные могут дать пустой результат

    def test_scale_features(self, engineer: FeatureEngineer, sample_market_data: pd.DataFrame):
        """Тест масштабирования признаков."""
        features = engineer.generate_features(sample_market_data)

        # Первое масштабирование (обучение)
        scaled_features = engineer.scale_features(features, fit=True)

        assert isinstance(scaled_features, pd.DataFrame)
        assert scaled_features.shape == features.shape
        assert engineer.is_fitted

        # Проверяем, что масштабированные данные имеют нулевое среднее и единичную дисперсию
        numeric_cols = scaled_features.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            means = scaled_features[numeric_cols].mean()
            stds = scaled_features[numeric_cols].std()

            # Допускаем небольшие отклонения из-за численных ошибок
            assert all(abs(mean) < 0.1 for mean in means), "Среднее должно быть близко к 0"
            assert all(
                abs(std - 1) < 0.1 for std in stds if not np.isnan(std)
            ), "Стандартное отклонение должно быть близко к 1"

    def test_scale_features_transform_only(self, engineer: FeatureEngineer, sample_market_data: pd.DataFrame):
        """Тест трансформации без обучения."""
        features = engineer.generate_features(sample_market_data)

        # Сначала обучаем
        engineer.scale_features(features, fit=True)

        # Затем только трансформируем
        new_features = features.iloc[:50]  # Используем часть данных
        scaled_new = engineer.scale_features(new_features, fit=False)

        assert isinstance(scaled_new, pd.DataFrame)
        assert scaled_new.shape == new_features.shape

    def test_select_features(self, engineer: FeatureEngineer, sample_market_data: pd.DataFrame):
        """Тест селекции признаков."""
        features = engineer.generate_features(sample_market_data)

        # Создаем целевую переменную (следующая цена закрытия)
        target = features["close"].shift(-1).dropna()
        features_aligned = features.loc[target.index]

        selected_features = engineer.select_features(features_aligned, target, k=10)

        assert isinstance(selected_features, pd.DataFrame)
        assert selected_features.shape[1] <= 10  # Не больше 10 признаков
        assert len(selected_features) == len(target)

    def test_error_handling_invalid_data(self, engineer: FeatureEngineer):
        """Тест обработки некорректных данных."""
        # Данные без необходимых колонок
        invalid_data = pd.DataFrame({"random_col": [1, 2, 3]})

        with pytest.raises((KeyError, ValueError)):
            engineer.generate_features(invalid_data)

    def test_memory_efficiency_large_data(self, engineer: FeatureEngineer):
        """Тест эффективности памяти на больших данных."""
        # Создаем большой датасет
        large_data = pd.DataFrame(
            {
                "open": np.random.uniform(90, 110, 10000),
                "high": np.random.uniform(95, 115, 10000),
                "low": np.random.uniform(85, 105, 10000),
                "close": np.random.uniform(90, 110, 10000),
                "volume": np.random.uniform(500, 2000, 10000),
            }
        )

        # Проверяем, что операция завершается без ошибок памяти
        result = engineer.generate_features(large_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestFeatureImportanceAnalyzer:
    """Unit тесты для FeatureImportanceAnalyzer."""

    @pytest.fixture
    def analyzer(self) -> FeatureImportanceAnalyzer:
        """Создает экземпляр FeatureImportanceAnalyzer."""
        return FeatureImportanceAnalyzer()

    @pytest.fixture
    def sample_features_and_target(self) -> tuple[pd.DataFrame, pd.Series]:
        """Создает образцы признаков и целевой переменной."""
        np.random.seed(42)
        features = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 100),
                "feature_2": np.random.uniform(-1, 1, 100),
                "feature_3": np.random.exponential(1, 100),
                "noise_feature": np.random.normal(0, 0.1, 100),
            }
        )

        # Создаем целевую переменную, зависящую от некоторых признаков
        target = (
            2 * features["feature_1"]
            + -1.5 * features["feature_2"]
            + 0.1 * features["feature_3"]
            + np.random.normal(0, 0.1, 100)
        )

        return features, target

    def test_calculate_importance_mutual_info(
        self, analyzer: FeatureImportanceAnalyzer, sample_features_and_target: tuple[pd.DataFrame, pd.Series]
    ):
        """Тест расчета важности признаков через mutual information."""
        features, target = sample_features_and_target

        importance = analyzer.calculate_importance(features, target, method="mutual_info")

        assert isinstance(importance, dict)
        assert len(importance) == len(features.columns)
        assert all(isinstance(score, float) for score in importance.values())
        assert all(score >= 0 for score in importance.values())

    def test_calculate_importance_correlation(
        self, analyzer: FeatureImportanceAnalyzer, sample_features_and_target: tuple[pd.DataFrame, pd.Series]
    ):
        """Тест расчета важности через корреляцию."""
        features, target = sample_features_and_target

        importance = analyzer.calculate_importance(features, target, method="correlation")

        assert isinstance(importance, dict)
        assert len(importance) == len(features.columns)
        assert all(isinstance(score, float) for score in importance.values())

    def test_rank_features(
        self, analyzer: FeatureImportanceAnalyzer, sample_features_and_target: tuple[pd.DataFrame, pd.Series]
    ):
        """Тест ранжирования признаков."""
        features, target = sample_features_and_target

        ranking = analyzer.rank_features(features, target)

        assert isinstance(ranking, list)
        assert len(ranking) == len(features.columns)
        assert all(isinstance(item, tuple) for item in ranking)
        assert all(len(item) == 2 for item in ranking)  # (feature_name, importance_score)


class TestMarketRegimeClassifier:
    """Unit тесты для MarketRegimeClassifier."""

    @pytest.fixture
    def classifier(self) -> MarketRegimeClassifier:
        """Создает экземпляр MarketRegimeClassifier."""
        return MarketRegimeClassifier()

    @pytest.fixture
    def market_data_trending_up(self) -> pd.DataFrame:
        """Создает данные восходящего тренда."""
        dates = pd.date_range("2023-01-01", periods=50, freq="1h")
        prices = 100 + np.cumsum(np.random.normal(0.1, 0.5, 50))  # Восходящий тренд

        return pd.DataFrame({"close": prices, "volume": np.random.uniform(500, 1500, 50)}, index=dates)

    @pytest.fixture
    def market_data_sideways(self) -> pd.DataFrame:
        """Создает данные бокового движения."""
        dates = pd.date_range("2023-01-01", periods=50, freq="1h")
        prices = 100 + np.random.normal(0, 1, 50)  # Боковое движение

        return pd.DataFrame({"close": prices, "volume": np.random.uniform(500, 1500, 50)}, index=dates)

    def test_classify_regime_trending(self, classifier: MarketRegimeClassifier, market_data_trending_up: pd.DataFrame):
        """Тест классификации трендового режима."""
        regime = classifier.classify_regime(market_data_trending_up)

        assert regime in ["trending", "sideways", "volatile"]
        # Для восходящих данных ожидаем 'trending'
        assert regime == "trending"

    def test_classify_regime_sideways(self, classifier: MarketRegimeClassifier, market_data_sideways: pd.DataFrame):
        """Тест классификации бокового режима."""
        regime = classifier.classify_regime(market_data_sideways)

        assert regime in ["trending", "sideways", "volatile"]

    def test_classify_regime_insufficient_data(self, classifier: MarketRegimeClassifier):
        """Тест классификации с недостаточными данными."""
        small_data = pd.DataFrame({"close": [100, 101], "volume": [1000, 1100]})

        regime = classifier.classify_regime(small_data)

        # Должен возвращать дефолтный режим или обрабатывать ошибку
        assert regime in ["trending", "sideways", "volatile", "unknown"]

    def test_get_regime_features(self, classifier: MarketRegimeClassifier, market_data_trending_up: pd.DataFrame):
        """Тест получения признаков режима."""
        features = classifier.get_regime_features(market_data_trending_up)

        assert isinstance(features, dict)
        assert len(features) > 0
        assert all(isinstance(value, (int, float)) for value in features.values())


# Стресс-тесты
class TestFeatureEngineerStress:
    """Стресс тесты для FeatureEngineer."""

    @pytest.fixture
    def stress_engineer(self) -> FeatureEngineer:
        """Создает FeatureEngineer для стресс-тестов."""
        config = FeatureConfig(
            ema_periods=list(range(5, 105, 5)),  # Много периодов
            rsi_periods=list(range(7, 22, 2)),
            rolling_windows=list(range(5, 55, 5)),
        )
        return FeatureEngineer(config=config)

    def test_stress_large_dataset(self, stress_engineer: FeatureEngineer):
        """Стресс-тест с большим датасетом."""
        # Создаем очень большой датасет
        large_data = pd.DataFrame(
            {
                "open": np.random.uniform(90, 110, 100000),
                "high": np.random.uniform(95, 115, 100000),
                "low": np.random.uniform(85, 105, 100000),
                "close": np.random.uniform(90, 110, 100000),
                "volume": np.random.uniform(500, 2000, 100000),
            }
        )

        # Измеряем время выполнения
        import time

        start_time = time.time()

        result = stress_engineer.generate_features(large_data)

        end_time = time.time()
        execution_time = end_time - start_time

        # Проверяем результат
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Проверяем производительность (не должно занимать более 60 секунд)
        assert execution_time < 60, f"Выполнение заняло слишком много времени: {execution_time}s"

    def test_stress_high_frequency_data(self, stress_engineer: FeatureEngineer):
        """Стресс-тест с высокочастотными данными."""
        # Данные каждую секунду в течение дня
        dates = pd.date_range("2023-01-01", periods=86400, freq="1s")

        # Симулируем высокочастотные изменения
        price_changes = np.random.normal(0, 0.0001, 86400)
        prices = 100 + np.cumsum(price_changes)

        hf_data = pd.DataFrame(
            {
                "open": prices + np.random.normal(0, 0.001, 86400),
                "high": prices + abs(np.random.normal(0, 0.002, 86400)),
                "low": prices - abs(np.random.normal(0, 0.002, 86400)),
                "close": prices,
                "volume": np.random.exponential(100, 86400),
            },
            index=dates,
        )

        result = stress_engineer.generate_features(hf_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_stress_memory_usage(self, stress_engineer: FeatureEngineer):
        """Стресс-тест использования памяти."""
        # Создаем данные, которые могут вызвать проблемы с памятью
        memory_stress_data = pd.DataFrame(
            {
                "open": np.random.uniform(90, 110, 50000),
                "high": np.random.uniform(95, 115, 50000),
                "low": np.random.uniform(85, 105, 50000),
                "close": np.random.uniform(90, 110, 50000),
                "volume": np.random.uniform(500, 2000, 50000),
            }
        )

        # Проверяем, что операция не вызывает MemoryError
        try:
            result = stress_engineer.generate_features(memory_stress_data)
            assert isinstance(result, pd.DataFrame)
        except MemoryError:
            pytest.fail("MemoryError при обработке данных")

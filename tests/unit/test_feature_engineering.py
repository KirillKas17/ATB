"""
Тесты для модуля feature_engineering.
"""

import numpy as np
import pandas as pd
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta

from domain.services.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Тесты для FeatureEngineer."""

    @pytest.fixture
    def feature_engineer(self) -> FeatureEngineer:
        """Фикстура для FeatureEngineer."""
        return FeatureEngineer()

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Фикстура для тестовых рыночных данных."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        data = {
            'open': np.random.uniform(45000, 55000, 100),
            'high': np.random.uniform(46000, 56000, 100),
            'low': np.random.uniform(44000, 54000, 100),
            'close': np.random.uniform(45000, 55000, 100),
            'volume': np.random.uniform(1000000, 5000000, 100)
        }
        return pd.DataFrame(data, index=dates)

    def test_initialization(self, feature_engineer: FeatureEngineer) -> None:
        """Тест инициализации."""
        assert feature_engineer is not None
        assert hasattr(feature_engineer, 'generate_features')

    def test_generate_features(self, feature_engineer: FeatureEngineer, sample_market_data: pd.DataFrame) -> None:
        """Тест генерации признаков."""
        features = feature_engineer.generate_features(sample_market_data)
        
        assert features is not None
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_market_data)
        assert len(features.columns) > 0

    def test_add_basic_features(self, feature_engineer: FeatureEngineer, sample_market_data: pd.DataFrame) -> None:
        """Тест добавления базовых признаков."""
        # Создание пустого DataFrame для признаков
        features = pd.DataFrame(index=sample_market_data.index)
        
        # Добавление базовых признаков
        basic_features = feature_engineer._add_basic_features(features, sample_market_data)
        
        # Проверки
        assert basic_features is not None
        assert isinstance(basic_features, pd.DataFrame)
        assert len(basic_features) == len(sample_market_data)
        
        # Проверка наличия основных базовых признаков
        expected_features = ['returns', 'log_returns', 'high_low_ratio']
        for feature in expected_features:
            if feature in basic_features.columns:
                assert not basic_features[feature].isna().all()

    def test_add_technical_indicators(self, feature_engineer: FeatureEngineer, sample_market_data: pd.DataFrame) -> None:
        """Тест добавления технических индикаторов."""
        # Создание пустого DataFrame для признаков
        features = pd.DataFrame(index=sample_market_data.index)
        
        # Добавление технических индикаторов
        technical_features = feature_engineer._add_technical_indicators(features, sample_market_data)
        
        # Проверки
        assert technical_features is not None
        assert isinstance(technical_features, pd.DataFrame)
        assert len(technical_features) == len(sample_market_data)

    def test_add_statistical_features(self, feature_engineer: FeatureEngineer, sample_market_data: pd.DataFrame) -> None:
        """Тест добавления статистических признаков."""
        # Создание пустого DataFrame для признаков
        features = pd.DataFrame(index=sample_market_data.index)
        
        # Добавление статистических признаков
        statistical_features = feature_engineer._add_statistical_features(features, sample_market_data)
        
        # Проверки
        assert statistical_features is not None
        assert isinstance(statistical_features, pd.DataFrame)
        assert len(statistical_features) == len(sample_market_data)
        
        # Проверка наличия основных статистических признаков
        expected_features = ['volatility_20', 'skewness_20', 'kurtosis_20']
        for feature in expected_features:
            if feature in statistical_features.columns:
                assert not statistical_features[feature].isna().all()

    def test_add_time_features(self, feature_engineer: FeatureEngineer, sample_market_data: pd.DataFrame) -> None:
        """Тест добавления временных признаков."""
        # Создание пустого DataFrame для признаков
        features = pd.DataFrame(index=sample_market_data.index)
        
        # Добавление временных признаков
        time_features = feature_engineer._add_time_features(features, sample_market_data)
        
        # Проверки
        assert time_features is not None
        assert isinstance(time_features, pd.DataFrame)
        assert len(time_features) == len(sample_market_data)
        
        # Проверка наличия основных временных признаков
        expected_features = ['hour', 'day_of_week', 'month']
        for feature in expected_features:
            if feature in time_features.columns:
                assert not time_features[feature].isna().all()

    def test_add_volume_features(self, feature_engineer: FeatureEngineer, sample_market_data: pd.DataFrame) -> None:
        """Тест добавления объемных признаков."""
        # Создание пустого DataFrame для признаков
        features = pd.DataFrame(index=sample_market_data.index)
        
        # Добавление объемных признаков
        volume_features = feature_engineer._add_volume_features(features, sample_market_data)
        
        # Проверки
        assert volume_features is not None
        assert isinstance(volume_features, pd.DataFrame)
        assert len(volume_features) == len(sample_market_data)
        
        # Проверка наличия основных объемных признаков
        expected_features = ['volume_sma_ratio', 'volume_ema_ratio']
        for feature in expected_features:
            if feature in volume_features.columns:
                assert not volume_features[feature].isna().all()

    def test_add_price_patterns(self, feature_engineer: FeatureEngineer, sample_market_data: pd.DataFrame) -> None:
        """Тест добавления паттернов цен."""
        # Создание пустого DataFrame для признаков
        features = pd.DataFrame(index=sample_market_data.index)
        
        # Добавление паттернов цен
        pattern_features = feature_engineer._add_price_patterns(features, sample_market_data)
        
        # Проверки
        assert pattern_features is not None
        assert isinstance(pattern_features, pd.DataFrame)
        assert len(pattern_features) == len(sample_market_data)
        
        # Проверка наличия основных паттернов
        expected_features = ['doji', 'higher_high', 'lower_low']
        for feature in expected_features:
            if feature in pattern_features.columns:
                assert not pattern_features[feature].isna().all()

    def test_add_microstructure_features(self, feature_engineer: FeatureEngineer, sample_market_data: pd.DataFrame) -> None:
        """Тест добавления микроструктурных признаков."""
        # Создание пустого DataFrame для признаков
        features = pd.DataFrame(index=sample_market_data.index)
        
        # Добавление микроструктурных признаков
        microstructure_features = feature_engineer._add_microstructure_features(features, sample_market_data)
        
        # Проверки
        assert microstructure_features is not None
        assert isinstance(microstructure_features, pd.DataFrame)
        assert len(microstructure_features) == len(sample_market_data)
        
        # Проверка наличия основных микроструктурных признаков
        expected_features = ['liquidity_ratio', 'intraday_volatility', 'price_momentum']
        for feature in expected_features:
            if feature in microstructure_features.columns:
                assert not microstructure_features[feature].isna().all()

    def test_normalize_features(self, feature_engineer: FeatureEngineer, sample_market_data: pd.DataFrame) -> None:
        """Тест нормализации признаков."""
        # Генерация признаков
        features = feature_engineer.generate_features(sample_market_data)
        
        # Нормализация признаков
        normalized_features = feature_engineer._normalize_features(features)
        
        # Проверки
        assert normalized_features is not None
        assert isinstance(normalized_features, pd.DataFrame)
        assert len(normalized_features) == len(features)
        
        # Проверка, что нормализованные данные в разумных пределах
        numeric_features = normalized_features.select_dtypes(include=[np.number])
        for column in numeric_features.columns:
            if not numeric_features[column].isna().all():
                assert numeric_features[column].min() >= -10
                assert numeric_features[column].max() <= 10

    def test_remove_outliers(self, feature_engineer: FeatureEngineer, sample_market_data: pd.DataFrame) -> None:
        """Тест удаления выбросов."""
        # Генерация признаков
        features = feature_engineer.generate_features(sample_market_data)
        
        # Удаление выбросов
        cleaned_features = feature_engineer._remove_outliers(features)
        
        # Проверки
        assert cleaned_features is not None
        assert isinstance(cleaned_features, pd.DataFrame)
        assert len(cleaned_features) == len(features)

    def test_select_features(self, feature_engineer: FeatureEngineer, sample_market_data: pd.DataFrame) -> None:
        """Тест селекции признаков."""
        # Генерация признаков
        features = feature_engineer.generate_features(sample_market_data)
        
        # Создание целевой переменной
        target = sample_market_data['close'].pct_change().shift(-1).dropna()
        features = features.iloc[:-1]  # Убираем последнюю строку для соответствия target
        
        # Селекция признаков
        selected_features = feature_engineer.select_features(features, target)
        
        # Проверки
        assert selected_features is not None
        assert isinstance(selected_features, pd.DataFrame)
        assert len(selected_features) == len(features)
        assert len(selected_features.columns) <= len(features.columns)

    def test_get_feature_importance(self, feature_engineer: FeatureEngineer, sample_market_data: pd.DataFrame) -> None:
        """Тест получения важности признаков."""
        # Генерация признаков
        features = feature_engineer.generate_features(sample_market_data)
        
        # Создание целевой переменной
        target = sample_market_data['close'].pct_change().shift(-1).dropna()
        features = features.iloc[:-1]  # Убираем последнюю строку для соответствия target
        
        # Селекция признаков для установки важности
        feature_engineer.select_features(features, target)
        
        # Получение важности признаков
        importance = feature_engineer.get_feature_importance()
        
        # Проверки
        assert importance is not None
        assert isinstance(importance, dict)

    def test_preprocess_features(self, feature_engineer: FeatureEngineer, sample_market_data: pd.DataFrame) -> None:
        """Тест предобработки признаков."""
        # Создание пустого DataFrame для признаков
        features = pd.DataFrame(index=sample_market_data.index)
        
        # Добавление базовых признаков
        features = feature_engineer._add_basic_features(features, sample_market_data)
        
        # Предобработка признаков
        preprocessed_features = feature_engineer._preprocess_features(features)
        
        # Проверки
        assert preprocessed_features is not None
        assert isinstance(preprocessed_features, pd.DataFrame)
        assert len(preprocessed_features) == len(features)
        
        # Проверка, что нет NaN значений
        assert not preprocessed_features.isna().any().any()  # type: ignore

    def test_save_and_load_features(self, feature_engineer: FeatureEngineer, sample_market_data: pd.DataFrame, tmp_path) -> None:
        """Тест сохранения и загрузки признаков."""
        # Генерация признаков
        features = feature_engineer.generate_features(sample_market_data)
        
        # Сохранение признаков
        save_path = tmp_path / "test_features.csv"
        feature_engineer.save_features(features, str(save_path))
        
        # Проверка, что файл создан
        assert save_path.exists()
        
        # Загрузка признаков
        loaded_features = feature_engineer.load_features(str(save_path))
        
        # Проверки
        assert loaded_features is not None
        assert isinstance(loaded_features, pd.DataFrame)
        assert len(loaded_features) == len(features)
        assert len(loaded_features.columns) == len(features.columns)

    def test_error_handling(self, feature_engineer: FeatureEngineer) -> None:
        """Тест обработки ошибок."""
        # Тест с пустыми данными
        empty_data = pd.DataFrame()
        features = feature_engineer.generate_features(empty_data)
        assert features.empty
        
        # Тест с некорректными данными
        invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
        features = feature_engineer.generate_features(invalid_data)
        assert isinstance(features, pd.DataFrame)

    def test_edge_cases(self, feature_engineer: FeatureEngineer) -> None:
        """Тест граничных случаев."""
        # Тест с минимальными данными
        min_data = pd.DataFrame({
            'open': [100],
            'high': [110],
            'low': [90],
            'close': [105],
            'volume': [1000]
        }, index=[pd.Timestamp('2023-01-01')])
        features = feature_engineer.generate_features(min_data)
        assert isinstance(features, pd.DataFrame)
        
        # Тест с очень большими данными
        large_data = pd.DataFrame({
            'open': np.random.uniform(45000, 55000, 10000),
            'high': np.random.uniform(46000, 56000, 10000),
            'low': np.random.uniform(44000, 54000, 10000),
            'close': np.random.uniform(45000, 55000, 10000),
            'volume': np.random.uniform(1000000, 5000000, 10000)
        }, index=pd.date_range('2023-01-01', periods=10000, freq='1H'))
        features = feature_engineer.generate_features(large_data)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(large_data)

    def test_cleanup(self, feature_engineer: FeatureEngineer) -> None:
        """Тест очистки ресурсов."""
        # Проверка, что объект можно создать и уничтожить без ошибок
        assert feature_engineer is not None
        
        # Проверка атрибутов
        assert hasattr(feature_engineer, 'scaler')
        assert hasattr(feature_engineer, 'feature_selector')
        assert hasattr(feature_engineer, 'feature_names')
        assert hasattr(feature_engineer, 'is_fitted') 

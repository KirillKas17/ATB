"""
Unit тесты для MarketRegimeDetector.
Тестирует определение режима рынка, включая классификацию режимов,
анализ переходов между режимами и прогнозирование изменений режима.
"""
import pytest
import pandas as pd
from shared.numpy_utils import np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from infrastructure.core.market_regime_detector import MarketRegimeDetector
class TestMarketRegimeDetector:
    """Тесты для MarketRegimeDetector."""
    @pytest.fixture
    def market_regime_detector(self) -> MarketRegimeDetector:
        """Фикстура для MarketRegimeDetector."""
        return MarketRegimeDetector()
    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Фикстура с тестовыми рыночными данными."""
        dates = pd.DatetimeIndex(pd.date_range('2023-01-01', periods=1000, freq='1H'))
        np.random.seed(42)
        # Создание данных с разными режимами
        data = pd.DataFrame({
            'close': np.random.uniform(45000, 55000, 1000),
            'volume': np.random.uniform(1000000, 5000000, 1000),
            'volatility': np.random.uniform(0.01, 0.05, 1000)
        }, index=dates)
        # Добавление трендов для разных режимов
        data.loc[:299, 'close'] = data.loc[:299, 'close'] + np.linspace(0, 5000, 300)  # type: ignore  # Восходящий тренд
        data.loc[300:599, 'close'] = data.loc[300:599, 'close'] + np.linspace(5000, 0, 300)  # type: ignore  # Нисходящий тренд
        data.loc[600:, 'close'] = data.loc[600:, 'close'] + np.random.uniform(-1000, 1000, 400)  # type: ignore  # Боковой тренд
        return data
    def test_initialization(self, market_regime_detector: MarketRegimeDetector) -> None:
        """Тест инициализации детектора режимов рынка."""
        assert market_regime_detector is not None
        assert hasattr(market_regime_detector, 'regime_classifiers')
        assert hasattr(market_regime_detector, 'transition_analyzers')
        assert hasattr(market_regime_detector, 'regime_predictors')
    def test_detect_market_regime(self, market_regime_detector: MarketRegimeDetector, sample_market_data: pd.DataFrame) -> None:
        """Тест определения режима рынка."""
        # Определение режима рынка
        regime_result = market_regime_detector.detect_market_regime(sample_market_data)
        # Проверки
        assert regime_result is not None
        assert "current_regime" in regime_result
        assert "regime_confidence" in regime_result
        assert "regime_metrics" in regime_result
        assert "regime_duration" in regime_result
        # Проверка типов данных
        assert regime_result["current_regime"] in ["trending", "ranging", "volatile", "stable"]
        assert isinstance(regime_result["regime_confidence"], float)
        assert isinstance(regime_result["regime_metrics"], dict)
        assert isinstance(regime_result["regime_duration"], timedelta)
        # Проверка диапазона
        assert 0.0 <= regime_result["regime_confidence"] <= 1.0
    def test_classify_regime_type(self, market_regime_detector: MarketRegimeDetector, sample_market_data: pd.DataFrame) -> None:
        """Тест классификации типа режима."""
        # Классификация режима
        classification_result = market_regime_detector.classify_regime_type(sample_market_data)
        # Проверки
        assert classification_result is not None
        assert "regime_type" in classification_result
        assert "classification_score" in classification_result
        assert "regime_characteristics" in classification_result
        assert "classification_method" in classification_result
        # Проверка типов данных
        assert classification_result["regime_type"] in ["trending", "ranging", "volatile", "stable"]
        assert isinstance(classification_result["classification_score"], float)
        assert isinstance(classification_result["regime_characteristics"], dict)
        assert isinstance(classification_result["classification_method"], str)
        # Проверка диапазона
        assert 0.0 <= classification_result["classification_score"] <= 1.0
    def test_analyze_regime_transitions(self, market_regime_detector: MarketRegimeDetector, sample_market_data: pd.DataFrame) -> None:
        """Тест анализа переходов между режимами."""
        # Анализ переходов
        transitions_result = market_regime_detector.analyze_regime_transitions(sample_market_data)
        # Проверки
        assert transitions_result is not None
        assert "regime_transitions" in transitions_result
        assert "transition_probabilities" in transitions_result
        assert "transition_triggers" in transitions_result
        assert "transition_patterns" in transitions_result
        # Проверка типов данных
        assert isinstance(transitions_result["regime_transitions"], list)
        assert isinstance(transitions_result["transition_probabilities"], dict)
        assert isinstance(transitions_result["transition_triggers"], dict)
        assert isinstance(transitions_result["transition_patterns"], dict)
    def test_predict_regime_changes(self, market_regime_detector: MarketRegimeDetector, sample_market_data: pd.DataFrame) -> None:
        """Тест прогнозирования изменений режима."""
        # Прогнозирование изменений
        prediction_result = market_regime_detector.predict_regime_changes(sample_market_data)
        # Проверки
        assert prediction_result is not None
        assert "predicted_regime" in prediction_result
        assert "prediction_confidence" in prediction_result
        assert "time_horizon" in prediction_result
        assert "regime_probabilities" in prediction_result
        # Проверка типов данных
        assert prediction_result["predicted_regime"] in ["trending", "ranging", "volatile", "stable"]
        assert isinstance(prediction_result["prediction_confidence"], float)
        assert isinstance(prediction_result["time_horizon"], timedelta)
        assert isinstance(prediction_result["regime_probabilities"], dict)
        # Проверка диапазона
        assert 0.0 <= prediction_result["prediction_confidence"] <= 1.0
    def test_calculate_regime_metrics(self, market_regime_detector: MarketRegimeDetector, sample_market_data: pd.DataFrame) -> None:
        """Тест расчета метрик режима."""
        # Расчет метрик
        metrics = market_regime_detector.calculate_regime_metrics(sample_market_data)
        # Проверки
        assert metrics is not None
        assert "volatility" in metrics
        assert "trend_strength" in metrics
        assert "mean_reversion" in metrics
        assert "momentum" in metrics
        assert "regime_score" in metrics
        # Проверка типов данных
        assert isinstance(metrics["volatility"], float)
        assert isinstance(metrics["trend_strength"], float)
        assert isinstance(metrics["mean_reversion"], float)
        assert isinstance(metrics["momentum"], float)
        assert isinstance(metrics["regime_score"], float)
        # Проверка диапазонов
        assert metrics["volatility"] >= 0.0
        assert -1.0 <= metrics["trend_strength"] <= 1.0
        assert 0.0 <= metrics["regime_score"] <= 1.0
    def test_identify_regime_characteristics(self, market_regime_detector: MarketRegimeDetector, sample_market_data: pd.DataFrame) -> None:
        """Тест идентификации характеристик режима."""
        # Идентификация характеристик
        characteristics = market_regime_detector.identify_regime_characteristics(sample_market_data)
        # Проверки
        assert characteristics is not None
        assert "price_behavior" in characteristics
        assert "volume_patterns" in characteristics
        assert "volatility_profile" in characteristics
        assert "correlation_structure" in characteristics
        # Проверка типов данных
        assert isinstance(characteristics["price_behavior"], dict)
        assert isinstance(characteristics["volume_patterns"], dict)
        assert isinstance(characteristics["volatility_profile"], dict)
        assert isinstance(characteristics["correlation_structure"], dict)
    def test_analyze_regime_stability(self, market_regime_detector: MarketRegimeDetector, sample_market_data: pd.DataFrame) -> None:
        """Тест анализа стабильности режима."""
        # Анализ стабильности
        stability_result = market_regime_detector.analyze_regime_stability(sample_market_data)
        # Проверки
        assert stability_result is not None
        assert "stability_score" in stability_result
        assert "stability_factors" in stability_result
        assert "stability_trend" in stability_result
        assert "stability_forecast" in stability_result
        # Проверка типов данных
        assert isinstance(stability_result["stability_score"], float)
        assert isinstance(stability_result["stability_factors"], dict)
        assert isinstance(stability_result["stability_trend"], str)
        assert isinstance(stability_result["stability_forecast"], dict)
        # Проверка диапазона
        assert 0.0 <= stability_result["stability_score"] <= 1.0
    def test_detect_regime_anomalies(self, market_regime_detector: MarketRegimeDetector, sample_market_data: pd.DataFrame) -> None:
        """Тест обнаружения аномалий режима."""
        # Обнаружение аномалий
        anomalies_result = market_regime_detector.detect_regime_anomalies(sample_market_data)
        # Проверки
        assert anomalies_result is not None
        assert "anomalies" in anomalies_result
        assert "anomaly_scores" in anomalies_result
        assert "anomaly_types" in anomalies_result
        assert "anomaly_impact" in anomalies_result
        # Проверка типов данных
        assert isinstance(anomalies_result["anomalies"], list)
        assert isinstance(anomalies_result["anomaly_scores"], dict)
        assert isinstance(anomalies_result["anomaly_types"], dict)
        assert isinstance(anomalies_result["anomaly_impact"], dict)
    def test_generate_regime_signals(self, market_regime_detector: MarketRegimeDetector, sample_market_data: pd.DataFrame) -> None:
        """Тест генерации сигналов режима."""
        # Генерация сигналов
        signals_result = market_regime_detector.generate_regime_signals(sample_market_data)
        # Проверки
        assert signals_result is not None
        assert "regime_signals" in signals_result
        assert "signal_strength" in signals_result
        assert "signal_confidence" in signals_result
        assert "signal_recommendations" in signals_result
        # Проверка типов данных
        assert isinstance(signals_result["regime_signals"], list)
        assert isinstance(signals_result["signal_strength"], dict)
        assert isinstance(signals_result["signal_confidence"], float)
        assert isinstance(signals_result["signal_recommendations"], list)
        # Проверка диапазона
        assert 0.0 <= signals_result["signal_confidence"] <= 1.0
    def test_get_regime_statistics(self, market_regime_detector: MarketRegimeDetector, sample_market_data: pd.DataFrame) -> None:
        """Тест получения статистики режимов."""
        # Получение статистики
        statistics = market_regime_detector.get_regime_statistics(sample_market_data)
        # Проверки
        assert statistics is not None
        assert "regime_distribution" in statistics
        assert "regime_durations" in statistics
        assert "regime_performance" in statistics
        assert "regime_transitions" in statistics
        # Проверка типов данных
        assert isinstance(statistics["regime_distribution"], dict)
        assert isinstance(statistics["regime_durations"], dict)
        assert isinstance(statistics["regime_performance"], dict)
        assert isinstance(statistics["regime_transitions"], dict)
    def test_validate_regime_data(self, market_regime_detector: MarketRegimeDetector, sample_market_data: pd.DataFrame) -> None:
        """Тест валидации данных режима."""
        # Валидация данных
        validation_result = market_regime_detector.validate_regime_data(sample_market_data)
        # Проверки
        assert validation_result is not None
        assert "is_valid" in validation_result
        assert "validation_errors" in validation_result
        assert "data_quality_score" in validation_result
        # Проверка типов данных
        assert isinstance(validation_result["is_valid"], bool)
        assert isinstance(validation_result["validation_errors"], list)
        assert isinstance(validation_result["data_quality_score"], float)
        # Проверка диапазона
        assert 0.0 <= validation_result["data_quality_score"] <= 1.0
    def test_error_handling(self, market_regime_detector: MarketRegimeDetector) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            market_regime_detector.detect_market_regime(None)
        with pytest.raises(ValueError):
            market_regime_detector.validate_regime_data(None)
    def test_edge_cases(self, market_regime_detector: MarketRegimeDetector) -> None:
        """Тест граничных случаев."""
        # Тест с очень короткими данными
        short_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        })
        regime_result = market_regime_detector.detect_market_regime(short_data)
        assert regime_result is not None
        # Тест с очень волатильными данными
        volatile_data = pd.DataFrame({
            'close': np.random.uniform(100, 1000, 100)
        })
        regime_result = market_regime_detector.detect_market_regime(volatile_data)
        assert regime_result is not None
    def test_cleanup(self, market_regime_detector: MarketRegimeDetector) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        market_regime_detector.cleanup()
        # Проверка, что ресурсы освобождены
        assert market_regime_detector.regime_classifiers == {}
        assert market_regime_detector.transition_analyzers == {}
        assert market_regime_detector.regime_predictors == {} 

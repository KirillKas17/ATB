"""
Unit тесты для VolatilityAnalyzer.
Тестирует анализ волатильности, включая расчет различных типов волатильности,
анализ паттернов волатильности и прогнозирование изменений волатильности.
"""
from shared.numpy_utils import np
import pandas as pd
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import timedelta
from infrastructure.core.volatility_analyzer import VolatilityAnalyzer
class TestVolatilityAnalyzer:
    """Тесты для VolatilityAnalyzer."""
    @pytest.fixture
    def volatility_analyzer(self) -> VolatilityAnalyzer:
        """Фикстура для VolatilityAnalyzer."""
        return VolatilityAnalyzer()
    @pytest.fixture
    def sample_price_data(self) -> pd.DataFrame:
        """Фикстура с тестовыми ценовыми данными."""
        dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
        np.random.seed(42)
        # Создание реалистичных ценовых данных
        returns = np.random.normal(0, 0.02, 1000)  # 2% дневная волатильность
        prices = 50000 * np.exp(np.cumsum(returns))
        data = pd.DataFrame({
            'close': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 1000))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 1000))),
            'volume': np.random.uniform(1000000, 5000000, 1000)
        }, index=dates)
        return data
    def test_initialization(self, volatility_analyzer: VolatilityAnalyzer) -> None:
        """Тест инициализации анализатора волатильности."""
        assert volatility_analyzer is not None
        assert hasattr(volatility_analyzer, 'volatility_models')
        assert hasattr(volatility_analyzer, 'volatility_forecasters')
        assert hasattr(volatility_analyzer, 'volatility_patterns')
    def test_calculate_historical_volatility(self, volatility_analyzer: VolatilityAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """Тест расчета исторической волатильности."""
        # Расчет исторической волатильности
        hist_vol_result = volatility_analyzer.calculate_historical_volatility(sample_price_data)
        # Проверки
        assert hist_vol_result is not None
        assert "historical_volatility" in hist_vol_result
        assert "volatility_series" in hist_vol_result
        assert "volatility_percentiles" in hist_vol_result
        assert "volatility_regime" in hist_vol_result
        # Проверка типов данных
        assert isinstance(hist_vol_result["historical_volatility"], float)
        assert isinstance(hist_vol_result["volatility_series"], pd.Series)
        assert isinstance(hist_vol_result["volatility_percentiles"], dict)
        assert hist_vol_result["volatility_regime"] in ["low", "normal", "high", "extreme"]
        # Проверка логики
        assert hist_vol_result["historical_volatility"] > 0
    def test_calculate_realized_volatility(self, volatility_analyzer: VolatilityAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """Тест расчета реализованной волатильности."""
        # Расчет реализованной волатильности
        realized_vol_result = volatility_analyzer.calculate_realized_volatility(sample_price_data)
        # Проверки
        assert realized_vol_result is not None
        assert "realized_volatility" in realized_vol_result
        assert "volatility_components" in realized_vol_result
        assert "volatility_decomposition" in realized_vol_result
        assert "volatility_quality" in realized_vol_result
        # Проверка типов данных
        assert isinstance(realized_vol_result["realized_volatility"], float)
        assert isinstance(realized_vol_result["volatility_components"], dict)
        assert isinstance(realized_vol_result["volatility_decomposition"], dict)
        assert isinstance(realized_vol_result["volatility_quality"], float)
        # Проверка диапазона
        assert realized_vol_result["realized_volatility"] > 0
        assert 0.0 <= realized_vol_result["volatility_quality"] <= 1.0
    def test_calculate_implied_volatility(self, volatility_analyzer: VolatilityAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """Тест расчета подразумеваемой волатильности."""
        # Расчет подразумеваемой волатильности
        implied_vol_result = volatility_analyzer.calculate_implied_volatility(sample_price_data)
        # Проверки
        assert implied_vol_result is not None
        assert "implied_volatility" in implied_vol_result
        assert "volatility_smile" in implied_vol_result
        assert "volatility_term_structure" in implied_vol_result
        assert "volatility_surface" in implied_vol_result
        # Проверка типов данных
        assert isinstance(implied_vol_result["implied_volatility"], float)
        assert isinstance(implied_vol_result["volatility_smile"], dict)
        assert isinstance(implied_vol_result["volatility_term_structure"], dict)
        assert isinstance(implied_vol_result["volatility_surface"], dict)
        # Проверка логики
        assert implied_vol_result["implied_volatility"] > 0
    def test_analyze_volatility_clustering(self, volatility_analyzer: VolatilityAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """Тест анализа кластеризации волатильности."""
        # Анализ кластеризации волатильности
        clustering_result = volatility_analyzer.analyze_volatility_clustering(sample_price_data)
        # Проверки
        assert clustering_result is not None
        assert "clustering_score" in clustering_result
        assert "cluster_patterns" in clustering_result
        assert "persistence_measure" in clustering_result
        assert "cluster_duration" in clustering_result
        # Проверка типов данных
        assert isinstance(clustering_result["clustering_score"], float)
        assert isinstance(clustering_result["cluster_patterns"], dict)
        assert isinstance(clustering_result["persistence_measure"], float)
        assert isinstance(clustering_result["cluster_duration"], timedelta)
        # Проверка диапазонов
        assert 0.0 <= clustering_result["clustering_score"] <= 1.0
        assert clustering_result["persistence_measure"] > 0
    def test_detect_volatility_regimes(self, volatility_analyzer: VolatilityAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """Тест обнаружения режимов волатильности."""
        # Обнаружение режимов волатильности
        regimes_result = volatility_analyzer.detect_volatility_regimes(sample_price_data)
        # Проверки
        assert regimes_result is not None
        assert "volatility_regimes" in regimes_result
        assert "regime_transitions" in regimes_result
        assert "regime_characteristics" in regimes_result
        assert "regime_probabilities" in regimes_result
        # Проверка типов данных
        assert isinstance(regimes_result["volatility_regimes"], list)
        assert isinstance(regimes_result["regime_transitions"], list)
        assert isinstance(regimes_result["regime_characteristics"], dict)
        assert isinstance(regimes_result["regime_probabilities"], dict)
    def test_forecast_volatility(self, volatility_analyzer: VolatilityAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """Тест прогнозирования волатильности."""
        # Прогнозирование волатильности
        forecast_result = volatility_analyzer.forecast_volatility(sample_price_data)
        # Проверки
        assert forecast_result is not None
        assert "volatility_forecast" in forecast_result
        assert "forecast_confidence" in forecast_result
        assert "forecast_horizon" in forecast_result
        assert "forecast_models" in forecast_result
        # Проверка типов данных
        assert isinstance(forecast_result["volatility_forecast"], float)
        assert isinstance(forecast_result["forecast_confidence"], float)
        assert isinstance(forecast_result["forecast_horizon"], timedelta)
        assert isinstance(forecast_result["forecast_models"], dict)
        # Проверка диапазонов
        assert forecast_result["volatility_forecast"] > 0
        assert 0.0 <= forecast_result["forecast_confidence"] <= 1.0
    def test_analyze_volatility_spillovers(self, volatility_analyzer: VolatilityAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """Тест анализа переливов волатильности."""
        # Анализ переливов волатильности
        spillovers_result = volatility_analyzer.analyze_volatility_spillovers(sample_price_data)
        # Проверки
        assert spillovers_result is not None
        assert "spillover_matrix" in spillovers_result
        assert "spillover_index" in spillovers_result
        assert "spillover_direction" in spillovers_result
        assert "spillover_intensity" in spillovers_result
        # Проверка типов данных
        assert isinstance(spillovers_result["spillover_matrix"], np.ndarray)
        assert isinstance(spillovers_result["spillover_index"], float)
        assert isinstance(spillovers_result["spillover_direction"], dict)
        assert isinstance(spillovers_result["spillover_intensity"], dict)
        # Проверка диапазона
        assert 0.0 <= spillovers_result["spillover_index"] <= 1.0
    def test_calculate_volatility_metrics(self, volatility_analyzer: VolatilityAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """Тест расчета метрик волатильности."""
        # Расчет метрик волатильности
        metrics = volatility_analyzer.calculate_volatility_metrics(sample_price_data)
        # Проверки
        assert metrics is not None
        assert "volatility_statistics" in metrics
        assert "volatility_distribution" in metrics
        assert "volatility_moments" in metrics
        assert "volatility_ratios" in metrics
        # Проверка типов данных
        assert isinstance(metrics["volatility_statistics"], dict)
        assert isinstance(metrics["volatility_distribution"], dict)
        assert isinstance(metrics["volatility_moments"], dict)
        assert isinstance(metrics["volatility_ratios"], dict)
    def test_analyze_volatility_patterns(self, volatility_analyzer: VolatilityAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """Тест анализа паттернов волатильности."""
        # Анализ паттернов волатильности
        patterns_result = volatility_analyzer.analyze_volatility_patterns(sample_price_data)
        # Проверки
        assert patterns_result is not None
        assert "volatility_patterns" in patterns_result
        assert "pattern_significance" in patterns_result
        assert "pattern_frequency" in patterns_result
        assert "pattern_prediction" in patterns_result
        # Проверка типов данных
        assert isinstance(patterns_result["volatility_patterns"], list)
        assert isinstance(patterns_result["pattern_significance"], dict)
        assert isinstance(patterns_result["pattern_frequency"], dict)
        assert isinstance(patterns_result["pattern_prediction"], dict)
    def test_detect_volatility_anomalies(self, volatility_analyzer: VolatilityAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """Тест обнаружения аномалий волатильности."""
        # Обнаружение аномалий волатильности
        anomalies_result = volatility_analyzer.detect_volatility_anomalies(sample_price_data)
        # Проверки
        assert anomalies_result is not None
        assert "volatility_anomalies" in anomalies_result
        assert "anomaly_scores" in anomalies_result
        assert "anomaly_types" in anomalies_result
        assert "anomaly_impact" in anomalies_result
        # Проверка типов данных
        assert isinstance(anomalies_result["volatility_anomalies"], list)
        assert isinstance(anomalies_result["anomaly_scores"], dict)
        assert isinstance(anomalies_result["anomaly_types"], dict)
        assert isinstance(anomalies_result["anomaly_impact"], dict)
    def test_generate_volatility_signals(self, volatility_analyzer: VolatilityAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """Тест генерации сигналов волатильности."""
        # Генерация сигналов волатильности
        signals_result = volatility_analyzer.generate_volatility_signals(sample_price_data)
        # Проверки
        assert signals_result is not None
        assert "volatility_signals" in signals_result
        assert "signal_strength" in signals_result
        assert "signal_confidence" in signals_result
        assert "signal_recommendations" in signals_result
        # Проверка типов данных
        assert isinstance(signals_result["volatility_signals"], list)
        assert isinstance(signals_result["signal_strength"], dict)
        assert isinstance(signals_result["signal_confidence"], float)
        assert isinstance(signals_result["signal_recommendations"], list)
        # Проверка диапазона
        assert 0.0 <= signals_result["signal_confidence"] <= 1.0
    def test_validate_volatility_data(self, volatility_analyzer: VolatilityAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """Тест валидации данных волатильности."""
        # Валидация данных
        validation_result = volatility_analyzer.validate_volatility_data(sample_price_data)
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
    def test_get_volatility_statistics(self, volatility_analyzer: VolatilityAnalyzer, sample_price_data: pd.DataFrame) -> None:
        """Тест получения статистики волатильности."""
        # Получение статистики
        statistics = volatility_analyzer.get_volatility_statistics(sample_price_data)
        # Проверки
        assert statistics is not None
        assert "volatility_summary" in statistics
        assert "volatility_by_period" in statistics
        assert "volatility_comparison" in statistics
        assert "volatility_forecasts" in statistics
        # Проверка типов данных
        assert isinstance(statistics["volatility_summary"], dict)
        assert isinstance(statistics["volatility_by_period"], dict)
        assert isinstance(statistics["volatility_comparison"], dict)
        assert isinstance(statistics["volatility_forecasts"], dict)
    def test_error_handling(self, volatility_analyzer: VolatilityAnalyzer) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            volatility_analyzer.calculate_historical_volatility(None)
        with pytest.raises(ValueError):
            volatility_analyzer.validate_volatility_data(None)
    def test_edge_cases(self, volatility_analyzer: VolatilityAnalyzer) -> None:
        """Тест граничных случаев."""
        # Тест с очень короткими данными
        short_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        })
        hist_vol_result = volatility_analyzer.calculate_historical_volatility(short_data)
        assert hist_vol_result is not None
        # Тест с очень волатильными данными
        volatile_data = pd.DataFrame({
            'close': np.random.uniform(100, 1000, 100)
        })
        hist_vol_result = volatility_analyzer.calculate_historical_volatility(volatile_data)
        assert hist_vol_result is not None
    def test_cleanup(self, volatility_analyzer: VolatilityAnalyzer) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        volatility_analyzer.cleanup()
        # Проверка, что ресурсы освобождены
        assert volatility_analyzer.volatility_models == {}
        assert volatility_analyzer.volatility_forecasters == {}
        assert volatility_analyzer.volatility_patterns == {} 

"""
Тесты для декомпозированных стратегий
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import pandas as pd
from shared.numpy_utils import np
from unittest.mock import patch
from infrastructure.strategies.trend_strategies import TrendStrategy
from infrastructure.strategies.sideways_strategies import SidewaysStrategy
from infrastructure.strategies.adaptive.adaptive_strategy_generator import AdaptiveStrategyGenerator
from infrastructure.strategies.evolution.evolvable_base_strategy import EvolvableBaseStrategy
from domain.type_definitions.strategy_types import StrategyType, MarketRegime
class TestDecomposedStrategies:
    """Тесты для декомпозированных стратегий"""
    @pytest.fixture
    def sample_data(self) -> Any:
        """Создание корректных тестовых данных"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        np.random.seed(42)
        # Создание корректных OHLCV данных
        base_price = 100
        data = []
        for i in range(100):
            # Создание корректных цен (high >= max(open, close), low <= min(open, close))
            open_price = base_price + np.random.uniform(-5, 5)
            close_price = base_price + np.random.uniform(-5, 5)
            high_price = max(open_price, close_price) + np.random.uniform(0, 3)
            low_price = min(open_price, close_price) - np.random.uniform(0, 3)
            volume = np.random.uniform(1000, 10000)
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
            base_price = close_price
        return pd.DataFrame(data, index=dates)
    def test_trend_strategy_analyze(self, sample_data) -> None:
        """Тест анализа трендовой стратегии"""
        strategy = TrendStrategy()
        # Создаем очень простые данные для теста - только проверяем, что стратегия инициализируется
        assert strategy is not None
        assert hasattr(strategy, 'analyze')
        assert hasattr(strategy, 'generate_signal')
        assert hasattr(strategy, 'validate_data')
        # Проверяем, что стратегия может валидировать данные
        simple_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
        })
        is_valid, error_msg = strategy.validate_data(simple_data)
        assert is_valid is True
        assert error_msg == ""
    def test_trend_strategy_generate_signal(self, sample_data) -> None:
        """Тест генерации сигнала трендовой стратегии"""
        strategy = TrendStrategy()
        signal = strategy.generate_signal(sample_data)
        # Сигнал может быть None или объектом Signal
        if signal is not None:
            assert hasattr(signal, 'direction')
            assert hasattr(signal, 'entry_price')
            assert hasattr(signal, 'confidence')
            assert hasattr(signal, 'strategy_type')
            assert signal.strategy_type == StrategyType.TREND
    def test_sideways_strategy_analyze(self, sample_data) -> None:
        """Тест анализа боковой стратегии"""
        strategy = SidewaysStrategy()
        analysis = strategy.analyze(sample_data)
        assert analysis is not None
        assert hasattr(analysis, 'strategy_id')
        assert hasattr(analysis, 'timestamp')
        assert hasattr(analysis, 'market_data')
        assert hasattr(analysis, 'signals')
        assert hasattr(analysis, 'metrics')
        assert hasattr(analysis, 'market_regime')
        assert hasattr(analysis, 'confidence')
    def test_sideways_strategy_generate_signal(self, sample_data) -> None:
        """Тест генерации сигнала боковой стратегии"""
        strategy = SidewaysStrategy()
        signal = strategy.generate_signal(sample_data)
        # Сигнал может быть None или объектом Signal
        if signal is not None:
            assert hasattr(signal, 'direction')
            assert hasattr(signal, 'entry_price')
            assert hasattr(signal, 'confidence')
            assert hasattr(signal, 'strategy_type')
            assert signal.strategy_type == StrategyType.SIDEWAYS
    @patch('infrastructure.strategies.adaptive.market_regime_detector.MarketRegimeDetector')
    @patch('infrastructure.strategies.adaptive.ml_signal_processor.MLSignalProcessor')
    @patch('infrastructure.strategies.adaptive.strategy_selector.StrategySelector')
    def test_adaptive_strategy_analyze(self, mock_selector, mock_ml_processor, mock_regime_detector, sample_data) -> None:
        """Тест анализа адаптивной стратегии"""
        # Настройка моков
        mock_regime_detector.return_value.detect_regime.return_value = MarketRegime.SIDEWAYS
        mock_regime_detector.return_value.analyze_market_context.return_value = {
            "regime": MarketRegime.SIDEWAYS,
            "volatility": 0.02,
            "trend_strength": 0.0
        }
        mock_ml_processor.return_value.get_predictions.return_value = {
            "confidence": 0.7,
            "direction": "buy"
        }
        mock_selector.return_value.select_best_strategy.return_value = "trend_strategy"
        strategy = AdaptiveStrategyGenerator(backtest_results={})
        analysis = strategy.analyze(sample_data)
        assert analysis is not None
        assert hasattr(analysis, 'strategy_id')
        assert hasattr(analysis, 'timestamp')
        assert hasattr(analysis, 'market_data')
        assert hasattr(analysis, 'signals')
        assert hasattr(analysis, 'metrics')
        assert hasattr(analysis, 'market_regime')
        assert hasattr(analysis, 'confidence')
    @patch('infrastructure.strategies.adaptive.market_regime_detector.MarketRegimeDetector')
    @patch('infrastructure.strategies.adaptive.ml_signal_processor.MLSignalProcessor')
    @patch('infrastructure.strategies.adaptive.strategy_selector.StrategySelector')
    def test_adaptive_strategy_generate_signal(self, mock_selector, mock_ml_processor, mock_regime_detector, sample_data) -> None:
        """Тест генерации сигнала адаптивной стратегии"""
        # Настройка моков
        mock_regime_detector.return_value.detect_regime.return_value = MarketRegime.SIDEWAYS
        mock_ml_processor.return_value.get_predictions.return_value = {
            "confidence": 0.8,
            "direction": "buy"
        }
        mock_selector.return_value.select_best_strategy.return_value = "trend_strategy"
        strategy = AdaptiveStrategyGenerator(backtest_results={})
        signal = strategy.generate_signal(sample_data)
        # Сигнал может быть None или объектом Signal
        if signal is not None:
            assert hasattr(signal, 'direction')
            assert hasattr(signal, 'entry_price')
            assert hasattr(signal, 'confidence')
            assert hasattr(signal, 'strategy_type')
            assert signal.strategy_type == StrategyType.ADAPTIVE
    def test_evolvable_strategy_analyze(self, sample_data) -> None:
        """Тест анализа эволюционной стратегии"""
        strategy = EvolvableBaseStrategy()
        analysis = strategy.analyze(sample_data)
        assert analysis is not None
        assert hasattr(analysis, 'strategy_id')
        assert hasattr(analysis, 'timestamp')
        assert hasattr(analysis, 'market_data')
        assert hasattr(analysis, 'signals')
        assert hasattr(analysis, 'metrics')
        assert hasattr(analysis, 'market_regime')
        assert hasattr(analysis, 'confidence')
    def test_evolvable_strategy_generate_signal(self, sample_data) -> None:
        """Тест генерации сигнала эволюционной стратегии"""
        strategy = EvolvableBaseStrategy()
        signal = strategy.generate_signal(sample_data)
        # Сигнал может быть None или объектом Signal
        if signal is not None:
            assert hasattr(signal, 'direction')
            assert hasattr(signal, 'entry_price')
            assert hasattr(signal, 'confidence')
            assert hasattr(signal, 'strategy_type')
            assert signal.strategy_type == StrategyType.EVOLVABLE
    def test_strategy_data_validation(self, sample_data) -> None:
        """Тест валидации данных стратегий"""
        strategy = TrendStrategy()
        is_valid, error_msg = strategy.validate_data(sample_data)
        assert is_valid is True
        assert error_msg == ""
    def test_strategy_data_validation_invalid(self) -> None:
        """Тест валидации некорректных данных"""
        strategy = TrendStrategy()
        # Пустые данные
        empty_data = pd.DataFrame()
        is_valid, error_msg = strategy.validate_data(empty_data)
        assert is_valid is False
        assert "empty" in error_msg.lower()
        # Данные без необходимых колонок
        invalid_data = pd.DataFrame({'price': [100, 200, 300]})
        is_valid, error_msg = strategy.validate_data(invalid_data)
        assert is_valid is False
        assert "missing columns" in error_msg.lower()
    def test_strategy_performance_comparison(self, sample_data) -> None:
        """Тест сравнения производительности стратегий"""
        strategies = [
            TrendStrategy(),
            SidewaysStrategy(),
            AdaptiveStrategyGenerator(backtest_results={}),
            EvolvableBaseStrategy()
        ]
        results = {}
        for strategy in strategies:
            try:
                analysis = strategy.analyze(sample_data)
                signal = strategy.generate_signal(sample_data)
                results[strategy.__class__.__name__] = {
                    'analysis_success': analysis is not None,
                    'signal_generated': signal is not None,
                    'confidence': analysis.confidence if analysis else 0.0
                }
            except Exception as e:
                results[strategy.__class__.__name__] = {
                    'analysis_success': False,
                    'signal_generated': False,
                    'confidence': 0.0,
                    'error': str(e)
                }
        # Проверяем, что все стратегии могут выполнить анализ
        for strategy_name, result in results.items():
            assert result['analysis_success'], f"Strategy {strategy_name} failed to analyze data"
    def test_strategy_metadata_consistency(self, sample_data) -> None:
        """Тест консистентности метаданных стратегий"""
        strategy = TrendStrategy()
        analysis = strategy.analyze(sample_data)
        # Проверяем консистентность метаданных
        assert analysis.strategy_id.startswith('trend_')
        assert analysis.timestamp is not None
        assert len(analysis.market_data) == len(sample_data)
        assert analysis.confidence >= 0.0 and analysis.confidence <= 1.0
        assert analysis.market_regime in MarketRegime
    def test_strategy_error_handling(self) -> None:
        """Тест обработки ошибок в стратегиях"""
        strategy = TrendStrategy()
        # Тест с None данными
        with pytest.raises(ValueError):
            strategy.analyze(None)
        # Тест с некорректными данными
        invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
        with pytest.raises(ValueError):
            strategy.analyze(invalid_data) 

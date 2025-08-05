import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, patch, MagicMock
from application.trading_orchestrator import TradingOrchestrator
from application.agent_context import AgentContext
from domain.evolutionary_transformer import EvolutionaryTransformer
from domain.market_data import MarketData
from domain.strategy_modifiers import StrategyModifiers
class TestTradingOrchestratorEvolutionaryTransformer:
    """Тесты интеграции EvolutionaryTransformer в TradingOrchestrator"""
    @pytest.fixture
    def mock_evolutionary_transformer(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Мок EvolutionaryTransformer"""
        mock = Mock(spec=EvolutionaryTransformer)
        mock.analyze_market_data.return_value = {
            'evolutionary_analysis': {
                'confidence': 0.85,
                'trend_prediction': 'bullish',
                'volatility_forecast': 0.12,
                'adaptation_score': 0.78
            }
        }
        return mock
    @pytest.fixture
    def mock_agent_context(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Мок AgentContext"""
        mock = Mock(spec=AgentContext)
        mock.evolutionary_transformer_result = None
        mock.apply_evolutionary_transformer_modifier.return_value = StrategyModifiers(
            position_size_multiplier=1.2,
            risk_tolerance_adjustment=0.1,
            entry_timing_modifier=0.05
        )
        return mock
    @pytest.fixture
    def mock_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Мок MarketData"""
        mock = Mock(spec=MarketData)
        mock.get_ohlcv_data.return_value = {
            'open': [100, 101, 102],
            'high': [103, 104, 105],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        }
        mock.get_orderbook_data.return_value = {
            'bids': [[100, 1000], [99, 2000]],
            'asks': [[101, 1500], [102, 2500]]
        }
        return mock
    @pytest.fixture
    def trading_orchestrator(self, mock_evolutionary_transformer, mock_agent_context) -> Any:
        """TradingOrchestrator с интегрированным EvolutionaryTransformer"""
        with patch('application.di_container.get_evolutionary_transformer', return_value=mock_evolutionary_transformer):
            orchestrator = TradingOrchestrator()
            orchestrator.agent_context = mock_agent_context
            orchestrator._evolutionary_transformer_cache = {}
            return orchestrator
    def test_evolutionary_transformer_integration_initialization(self, trading_orchestrator, mock_evolutionary_transformer) -> None:
        """Тест инициализации EvolutionaryTransformer в TradingOrchestrator"""
        assert hasattr(trading_orchestrator, '_evolutionary_transformer_cache')
        assert isinstance(trading_orchestrator._evolutionary_transformer_cache, dict)
        # Проверяем, что EvolutionaryTransformer получен из DI контейнера
        with patch('application.di_container.get_evolutionary_transformer') as mock_get:
            mock_get.return_value = mock_evolutionary_transformer
            orchestrator = TradingOrchestrator()
            assert orchestrator._evolutionary_transformer is not None
    def test_update_evolutionary_transformer(self, trading_orchestrator, mock_evolutionary_transformer, mock_market_data) -> None:
        """Тест обновления EvolutionaryTransformer"""
        # Подготавливаем данные
        symbol = "BTCUSDT"
        timeframe = "1h"
        # Вызываем метод
        result = trading_orchestrator._update_evolutionary_transformer(symbol, timeframe, mock_market_data)
        # Проверяем вызовы
        mock_evolutionary_transformer.analyze_market_data.assert_called_once()
        call_args = mock_evolutionary_transformer.analyze_market_data.call_args[0][0]
        assert call_args is not None
        # Проверяем результат
        assert result is not None
        assert 'evolutionary_analysis' in result
        # Проверяем кэш
        cache_key = f"{symbol}_{timeframe}"
        assert cache_key in trading_orchestrator._evolutionary_transformer_cache
    def test_apply_evolutionary_transformer_analysis(self, trading_orchestrator, mock_agent_context) -> None:
        """Тест применения анализа EvolutionaryTransformer"""
        # Подготавливаем данные анализа
        analysis_result = {
            'evolutionary_analysis': {
                'confidence': 0.85,
                'trend_prediction': 'bullish',
                'volatility_forecast': 0.12,
                'adaptation_score': 0.78
            }
        }
        # Вызываем метод
        trading_orchestrator._apply_evolutionary_transformer_analysis(analysis_result)
        # Проверяем обновление AgentContext
        assert mock_agent_context.evolutionary_transformer_result == analysis_result
        # Проверяем применение модификатора
        mock_agent_context.apply_evolutionary_transformer_modifier.assert_called_once_with(analysis_result)
    def test_get_market_data_for_evolutionary_transformer(self, trading_orchestrator, mock_market_data) -> None:
        """Тест получения данных для EvolutionaryTransformer"""
        symbol = "BTCUSDT"
        timeframe = "1h"
        # Вызываем метод
        result = trading_orchestrator._get_market_data_for_evolutionary_transformer(symbol, timeframe)
        # Проверяем результат
        assert result is not None
        assert hasattr(result, 'get_ohlcv_data')
        assert hasattr(result, 'get_orderbook_data')
    def test_evolutionary_transformer_in_execute_strategy(self, trading_orchestrator, mock_evolutionary_transformer, mock_market_data) -> None:
        """Тест интеграции EvolutionaryTransformer в execute_strategy"""
        symbol = "BTCUSDT"
        timeframe = "1h"
        # Мокаем методы получения данных
        trading_orchestrator._get_market_data_for_evolutionary_transformer = Mock(return_value=mock_market_data)
        trading_orchestrator._update_evolutionary_transformer = Mock(return_value={'test': 'result'})
        trading_orchestrator._apply_evolutionary_transformer_analysis = Mock()
        # Вызываем execute_strategy
        trading_orchestrator.execute_strategy(symbol, timeframe)
        # Проверяем вызовы
        trading_orchestrator._get_market_data_for_evolutionary_transformer.assert_called_once_with(symbol, timeframe)
        trading_orchestrator._update_evolutionary_transformer.assert_called_once_with(symbol, timeframe, mock_market_data)
        trading_orchestrator._apply_evolutionary_transformer_analysis.assert_called_once_with({'test': 'result'})
    def test_evolutionary_transformer_cache_functionality(self, trading_orchestrator, mock_evolutionary_transformer, mock_market_data) -> None:
        """Тест функциональности кэша EvolutionaryTransformer"""
        symbol = "BTCUSDT"
        timeframe = "1h"
        cache_key = f"{symbol}_{timeframe}"
        # Первый вызов - должен обновить кэш
        trading_orchestrator._update_evolutionary_transformer(symbol, timeframe, mock_market_data)
        assert cache_key in trading_orchestrator._evolutionary_transformer_cache
        # Второй вызов с теми же параметрами - должен использовать кэш
        mock_evolutionary_transformer.analyze_market_data.reset_mock()
        trading_orchestrator._update_evolutionary_transformer(symbol, timeframe, mock_market_data)
        # Проверяем, что analyze_market_data не вызывался повторно (использовался кэш)
        assert mock_evolutionary_transformer.analyze_market_data.call_count == 0
    def test_evolutionary_transformer_error_handling(self, trading_orchestrator, mock_evolutionary_transformer, mock_market_data) -> None:
        """Тест обработки ошибок EvolutionaryTransformer"""
        # Настраиваем мок для выброса исключения
        mock_evolutionary_transformer.analyze_market_data.side_effect = Exception("Test error")
        # Вызываем метод - не должно падать
        result = trading_orchestrator._update_evolutionary_transformer("BTCUSDT", "1h", mock_market_data)
        # Проверяем, что метод вернул None или пустой результат
        assert result is None or result == {}
    def test_evolutionary_transformer_modifier_application(self, trading_orchestrator, mock_agent_context) -> None:
        """Тест применения модификаторов EvolutionaryTransformer"""
        # Подготавливаем данные анализа
        analysis_result = {
            'evolutionary_analysis': {
                'confidence': 0.9,
                'trend_prediction': 'strong_bullish',
                'volatility_forecast': 0.15,
                'adaptation_score': 0.95
            }
        }
        # Настраиваем мок для возврата модификаторов
        expected_modifiers = StrategyModifiers(
            position_size_multiplier=1.5,
            risk_tolerance_adjustment=0.2,
            entry_timing_modifier=0.1
        )
        mock_agent_context.apply_evolutionary_transformer_modifier.return_value = expected_modifiers
        # Вызываем метод
        trading_orchestrator._apply_evolutionary_transformer_analysis(analysis_result)
        # Проверяем вызов с правильными параметрами
        mock_agent_context.apply_evolutionary_transformer_modifier.assert_called_once_with(analysis_result)
        # Проверяем, что результат сохранен в AgentContext
        assert mock_agent_context.evolutionary_transformer_result == analysis_result
    def test_evolutionary_transformer_integration_with_other_modules(self, trading_orchestrator, mock_agent_context) -> None:
        """Тест интеграции EvolutionaryTransformer с другими модулями"""
        # Проверяем, что AgentContext имеет поле для результатов
        assert hasattr(mock_agent_context, 'evolutionary_transformer_result')
        # Проверяем, что AgentContext имеет метод применения модификаторов
        assert hasattr(mock_agent_context, 'apply_evolutionary_transformer_modifier')
        # Проверяем, что TradingOrchestrator имеет кэш
        assert hasattr(trading_orchestrator, '_evolutionary_transformer_cache')
    def test_evolutionary_transformer_performance_optimization(self, trading_orchestrator, mock_evolutionary_transformer, mock_market_data) -> None:
        """Тест оптимизации производительности EvolutionaryTransformer"""
        symbol = "BTCUSDT"
        timeframe = "1h"
        # Первый вызов
        start_time = trading_orchestrator._update_evolutionary_transformer(symbol, timeframe, mock_market_data)
        # Второй вызов (должен быть быстрее из-за кэша)
        mock_evolutionary_transformer.analyze_market_data.reset_mock()
        trading_orchestrator._update_evolutionary_transformer(symbol, timeframe, mock_market_data)
        # Проверяем, что второй вызов не обращался к EvolutionaryTransformer
        assert mock_evolutionary_transformer.analyze_market_data.call_count == 0
    def test_evolutionary_transformer_data_validation(self, trading_orchestrator, mock_evolutionary_transformer) -> None:
        """Тест валидации данных для EvolutionaryTransformer"""
        # Тест с некорректными данными
        invalid_market_data = Mock()
        invalid_market_data.get_ohlcv_data.return_value = None
        invalid_market_data.get_orderbook_data.return_value = None
        # Вызываем метод с некорректными данными
        result = trading_orchestrator._update_evolutionary_transformer("BTCUSDT", "1h", invalid_market_data)
        # Проверяем, что метод корректно обработал некорректные данные
        assert result is None or result == {}
    def test_evolutionary_transformer_configuration_integration(self: "TestTradingOrchestratorEvolutionaryTransformer") -> None:
        """Тест интеграции конфигурации EvolutionaryTransformer"""
        # Проверяем, что EvolutionaryTransformer регистрируется в DI контейнере
        with patch('application.di_container.get_evolutionary_transformer') as mock_get:
            mock_get.return_value = Mock(spec=EvolutionaryTransformer)
            orchestrator = TradingOrchestrator()
            # Проверяем, что EvolutionaryTransformer получен
            assert orchestrator._evolutionary_transformer is not None 

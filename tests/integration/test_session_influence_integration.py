import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, patch, MagicMock
from application.use_cases.trading_orchestrator.core import DefaultTradingOrchestratorUseCase as TradingOrchestrator
from domain.intelligence.session_influence_analyzer import SessionInfluenceAnalyzer
from domain.entities.market import Market
from domain.value_objects.currency import Currency
from infrastructure.agents.agent_context_refactored import AgentContext
class TestSessionInfluenceIntegration:
    """Тесты интеграции SessionInfluenceAnalyzer в TradingOrchestrator"""
    @pytest.fixture
    def mock_session_influence_analyzer(self) -> Any:
        """Мок SessionInfluenceAnalyzer"""
        analyzer = Mock(spec=SessionInfluenceAnalyzer)
        analyzer.analyze_session_influence.return_value = {
            'session_strength': 0.75,
            'influence_factor': 1.2,
            'confidence': 0.85,
            'session_type': 'bullish',
            'volatility_impact': 0.3,
            'liquidity_effect': 0.4
        }
        return analyzer
    @pytest.fixture
    def mock_market_data(self) -> Any:
        """Мок данных рынка"""
        return {
            'orderbook': {
                'bids': [[50000, 1.5], [49999, 2.0], [49998, 1.8]],
                'asks': [[50001, 1.2], [50002, 1.8], [50003, 2.1]]
            },
            'trades': [
                {'price': 50000, 'quantity': 0.5, 'side': 'buy', 'timestamp': 1640995200},
                {'price': 50001, 'quantity': 0.3, 'side': 'sell', 'timestamp': 1640995201},
                {'price': 49999, 'quantity': 0.8, 'side': 'buy', 'timestamp': 1640995202}
            ],
            'session_info': {
                'session_start': 1640995200,
                'session_duration': 3600,
                'volume_profile': {'asian': 0.3, 'european': 0.4, 'american': 0.3}
            }
        }
    @pytest.fixture
    def trading_orchestrator(self, mock_session_influence_analyzer) -> Any:
        """TradingOrchestrator с интегрированным SessionInfluenceAnalyzer"""
        with patch('application.use_cases.trading_orchestrator.SessionInfluenceAnalyzer') as mock_analyzer_class:
            mock_analyzer_class.return_value = mock_session_influence_analyzer
            orchestrator = TradingOrchestrator()
            return orchestrator
    def test_session_influence_analyzer_integration_in_constructor(self, trading_orchestrator) -> None:
        """Тест интеграции SessionInfluenceAnalyzer в конструкторе"""
        assert hasattr(trading_orchestrator, '_session_influence_analyzer')
        assert trading_orchestrator._session_influence_analyzer is not None
    def test_update_session_influence_analysis(self, trading_orchestrator, mock_market_data) -> None:
        """Тест обновления анализа влияния сессий"""
        context = AgentContext(symbol="BTC/USDT")
        # Обновляем анализ
        trading_orchestrator._update_session_influence_analysis(context, mock_market_data)
        # Проверяем, что анализ был вызван
        trading_orchestrator._session_influence_analyzer.analyze_session_influence.assert_called_once()
        # Проверяем, что результат сохранен в контексте
        assert context.session_influence_result is not None
        assert 'session_strength' in context.session_influence_result
        assert 'influence_factor' in context.session_influence_result
    def test_apply_session_influence_to_signal(self, trading_orchestrator) -> None:
        """Тест применения анализа влияния сессий к сигналу"""
        context = AgentContext(symbol="BTC/USDT")
        context.session_influence_result = {
            'session_strength': 0.75,
            'influence_factor': 1.2,
            'confidence': 0.85,
            'session_type': 'bullish',
            'volatility_impact': 0.3,
            'liquidity_effect': 0.4
        }
        original_signal = {
            'action': 'buy',
            'confidence': 0.7,
            'price': 50000,
            'quantity': 1.0
        }
        # Применяем влияние сессий
        modified_signal = trading_orchestrator._apply_session_influence_to_signal(
            context, original_signal
        )
        # Проверяем, что сигнал был модифицирован
        assert modified_signal['action'] == 'buy'
        assert modified_signal['confidence'] > original_signal['confidence']  # Увеличена уверенность
        assert 'session_influence' in modified_signal
        assert modified_signal['session_influence']['factor'] == 1.2
    def test_get_orderbook_data_for_session_analysis(self, trading_orchestrator, mock_market_data) -> None:
        """Тест получения данных ордербука для анализа сессий"""
        orderbook_data = trading_orchestrator._get_orderbook_data_for_session_analysis(mock_market_data)
        assert 'bids' in orderbook_data
        assert 'asks' in orderbook_data
        assert len(orderbook_data['bids']) == 3
        assert len(orderbook_data['asks']) == 3
        assert orderbook_data['bids'][0] == [50000, 1.5]
    def test_get_trade_data_for_session_analysis(self, trading_orchestrator, mock_market_data) -> None:
        """Тест получения данных сделок для анализа сессий"""
        trade_data = trading_orchestrator._get_trade_data_for_session_analysis(mock_market_data)
        assert len(trade_data) == 3
        assert trade_data[0]['price'] == 50000
        assert trade_data[0]['quantity'] == 0.5
        assert trade_data[0]['side'] == 'buy'
    def test_session_influence_in_execute_strategy(self, trading_orchestrator, mock_market_data) -> None:
        """Тест интеграции анализа влияния сессий в execute_strategy"""
        context = AgentContext(symbol="BTC/USD")
        market = Market(
            symbol="BTC/USD",
            base_currency=Currency("BTC"),
            quote_currency=Currency("USD")
        )
        # Выполняем стратегию
        result = trading_orchestrator.execute_strategy(context, market, mock_market_data)
        # Проверяем, что анализ влияния сессий был выполнен
        assert context.session_influence_result is not None
        assert 'session_strength' in context.session_influence_result
    def test_session_influence_in_process_signal(self, trading_orchestrator) -> None:
        """Тест интеграции анализа влияния сессий в process_signal"""
        context = AgentContext(symbol="BTC/USD")
        context.session_influence_result = {
            'session_strength': 0.75,
            'influence_factor': 1.2,
            'confidence': 0.85,
            'session_type': 'bullish',
            'volatility_impact': 0.3,
            'liquidity_effect': 0.4
        }
        signal = {
            'action': 'buy',
            'confidence': 0.7,
            'price': 50000,
            'quantity': 1.0
        }
        # Обрабатываем сигнал
        processed_signal = trading_orchestrator.process_signal(context, signal)
        # Проверяем, что сигнал был обработан с учетом влияния сессий
        assert 'session_influence' in processed_signal
        assert processed_signal['session_influence']['factor'] == 1.2
    def test_session_influence_with_different_session_types(self, trading_orchestrator) -> None:
        """Тест влияния разных типов сессий"""
        context = AgentContext(symbol="BTC/USD")
        # Тест с бычьей сессией
        context.session_influence_result = {
            'session_strength': 0.8,
            'influence_factor': 1.3,
            'confidence': 0.9,
            'session_type': 'bullish',
            'volatility_impact': 0.2,
            'liquidity_effect': 0.5
        }
        signal = {'action': 'buy', 'confidence': 0.6, 'price': 50000, 'quantity': 1.0}
        processed_signal = trading_orchestrator._apply_session_influence_to_signal(context, signal)
        assert processed_signal['confidence'] > signal['confidence']
        assert processed_signal['session_influence']['factor'] == 1.3
        # Тест с медвежьей сессией
        context.session_influence_result['session_type'] = 'bearish'
        context.session_influence_result['influence_factor'] = 0.8
        processed_signal = trading_orchestrator._apply_session_influence_to_signal(context, signal)
        assert processed_signal['session_influence']['factor'] == 0.8
    def test_session_influence_error_handling(self, trading_orchestrator) -> None:
        """Тест обработки ошибок в анализе влияния сессий"""
        context = AgentContext(symbol="BTC/USD")
        # Симулируем ошибку в анализаторе
        trading_orchestrator._session_influence_analyzer.analyze_session_influence.side_effect = Exception("Analysis error")
        mock_market_data = {'orderbook': {'bids': [], 'asks': []}, 'trades': []}
        # Должно обработаться без ошибок
        trading_orchestrator._update_session_influence_analysis(context, mock_market_data)
        # Проверяем, что контекст не был поврежден
        assert context.session_influence_result is None
    def test_session_influence_with_empty_market_data(self, trading_orchestrator) -> None:
        """Тест анализа влияния сессий с пустыми данными рынка"""
        context = AgentContext(symbol="BTC/USD")
        empty_market_data = {
            'orderbook': {'bids': [], 'asks': []},
            'trades': [],
            'session_info': {}
        }
        # Обновляем анализ с пустыми данными
        trading_orchestrator._update_session_influence_analysis(context, empty_market_data)
        # Проверяем, что анализатор был вызван с пустыми данными
        trading_orchestrator._session_influence_analyzer.analyze_session_influence.assert_called_once()
    def test_session_influence_integration_completeness(self, trading_orchestrator) -> None:
        """Тест полноты интеграции SessionInfluenceAnalyzer"""
        # Проверяем наличие всех необходимых методов
        assert hasattr(trading_orchestrator, '_update_session_influence_analysis')
        assert hasattr(trading_orchestrator, '_apply_session_influence_to_signal')
        assert hasattr(trading_orchestrator, '_get_orderbook_data_for_session_analysis')
        assert hasattr(trading_orchestrator, '_get_trade_data_for_session_analysis')
        # Проверяем, что методы являются callable
        assert callable(trading_orchestrator._update_session_influence_analysis)
        assert callable(trading_orchestrator._apply_session_influence_to_signal)
        assert callable(trading_orchestrator._get_orderbook_data_for_session_analysis)
        assert callable(trading_orchestrator._get_trade_data_for_session_analysis) 

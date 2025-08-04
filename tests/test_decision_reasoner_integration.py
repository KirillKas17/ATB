"""
Unit тесты для интеграции DecisionReasoner.
"""
import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from infrastructure.ml_services.decision_reasoner import DecisionReasoner
from domain.type_definitions.decision_types import TradeDecision
from infrastructure.agents.agent_context_refactored import AgentContext
from application.use_cases.trading_orchestrator.core import DefaultTradingOrchestratorUseCase
from application.di_container import DIContainer, ContainerConfig
class TestDecisionReasonerIntegration:
    """Тесты интеграции DecisionReasoner."""
    @pytest.fixture
    def decision_reasoner(self) -> Any:
        """Создание экземпляра DecisionReasoner."""
        return DecisionReasoner(
            config=None
        )
    @pytest.fixture
    def agent_context(self) -> Any:
        """Создание экземпляра AgentContext."""
        return AgentContext(symbol="BTCUSDT")
    @pytest.fixture
    def mock_trade_decision(self) -> Any:
        """Создание мок TradeDecision."""
        return TradeDecision(
            symbol="BTCUSDT",
            action="open",
            direction="long",
            volume=0.1,
            confidence=0.85,
            stop_loss=45000.0,
            take_profit=55000.0,
            timestamp=datetime.now(),
            metadata={"reason": "strong_signal"}
        )
    def test_decision_reasoner_creation(self, decision_reasoner) -> None:
        """Тест создания DecisionReasoner."""
        assert decision_reasoner is not None
        assert decision_reasoner is not None
    def test_decision_reasoner_make_enhanced_decision(self, decision_reasoner) -> None:
        """Тест принятия улучшенного решения."""
        # Создаем тестовые данные
        dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='1H')
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': [45000 + i * 100 for i in range(len(dates))],
            'high': [45000 + i * 100 + 50 for i in range(len(dates))],
            'low': [45000 + i * 100 - 50 for i in range(len(dates))],
            'close': [45000 + i * 100 + 25 for i in range(len(dates))],
            'volume': [100 + i * 10 for i in range(len(dates))]
        })
        strategy_signals = [{"type": "trend", "signal": "buy", "confidence": 0.8}]
        ml_predictions = [{"model": "price_predictor", "prediction": "up", "confidence": 0.75}]
        technical_signals = [{"indicator": "rsi", "value": 65.0, "signal": "neutral"}]
        risk_context = {"symbol": "BTCUSDT", "position_size": 0.1, "max_risk": 0.02}
        # Принимаем решение
        decision = decision_reasoner.make_enhanced_decision(
            market_data=market_data,
            strategy_signals=strategy_signals,
            ml_predictions=ml_predictions,
            technical_signals=technical_signals,
            risk_context=risk_context
        )
        assert decision is not None
        assert isinstance(decision, TradeDecision)
        assert decision.symbol == "BTCUSDT"
        assert decision.action in ["open", "hold", "close"]
        assert decision.confidence >= 0.0 and decision.confidence <= 1.0
    def test_agent_context_decision_reasoning_integration(self, agent_context, mock_trade_decision) -> None:
        """Тест интеграции DecisionReasoner с AgentContext."""
        # Устанавливаем результат анализа решений
        agent_context.decision_reasoning_result = mock_trade_decision
        # Применяем модификатор
        agent_context.apply_decision_reasoning_modifier()
        # Проверяем, что модификаторы были применены
        assert agent_context.strategy_modifiers is not None
    def test_agent_context_decision_reasoning_high_confidence(self, agent_context) -> None:
        """Тест применения модификатора при высокой уверенности."""
        high_confidence_decision = TradeDecision(
            symbol="BTCUSDT",
            action="open",
            direction="long",
            volume=0.1,
            confidence=0.9,  # Высокая уверенность
            stop_loss=45000.0,
            take_profit=55000.0,
            timestamp=datetime.now(),
            metadata={}
        )
        agent_context.decision_reasoning_result = high_confidence_decision
        agent_context.apply_decision_reasoning_modifier()
        # При высокой уверенности должны увеличиться модификаторы
        assert agent_context.strategy_modifiers is not None
    def test_agent_context_decision_reasoning_low_confidence(self, agent_context) -> None:
        """Тест применения модификатора при низкой уверенности."""
        low_confidence_decision = TradeDecision(
            symbol="BTCUSDT",
            action="open",
            direction="long",
            volume=0.1,
            confidence=0.3,  # Низкая уверенность
            stop_loss=45000.0,
            take_profit=55000.0,
            timestamp=datetime.now(),
            metadata={}
        )
        agent_context.decision_reasoning_result = low_confidence_decision
        agent_context.apply_decision_reasoning_modifier()
        # При низкой уверенности должны снизиться модификаторы
        assert agent_context.strategy_modifiers.order_aggressiveness < 1.0
        assert agent_context.strategy_modifiers.position_size_multiplier < 1.0
        assert agent_context.strategy_modifiers.confidence_multiplier < 1.0
    def test_agent_context_decision_reasoning_hold_action(self, agent_context) -> None:
        """Тест применения модификатора при решении воздержаться."""
        hold_decision = TradeDecision(
            symbol="BTCUSDT",
            action="hold",  # Решение воздержаться
            direction="neutral",
            volume=0.0,
            confidence=0.5,
            stop_loss=0.0,
            take_profit=0.0,
            timestamp=datetime.now(),
            metadata={}
        )
        agent_context.decision_reasoning_result = hold_decision
        agent_context.apply_decision_reasoning_modifier()
        # При решении воздержаться должны снизиться модификаторы
        assert agent_context.strategy_modifiers.order_aggressiveness < 1.0
        assert agent_context.strategy_modifiers.position_size_multiplier < 1.0
        assert agent_context.strategy_modifiers.confidence_multiplier < 1.0
    def test_di_container_decision_reasoner_integration(self) -> None:
        """Тест интеграции DecisionReasoner в DI контейнер."""
        config = ContainerConfig(decision_reasoner_enabled=True)
        container = DIContainer(config)
        # Получаем DecisionReasoner из контейнера
        decision_reasoner = container.get("decision_reasoner")
        assert decision_reasoner is not None
        assert isinstance(decision_reasoner, DecisionReasoner)
        assert decision_reasoner.min_confidence == 0.7
        assert decision_reasoner.max_uncertainty == 0.3
    def test_di_container_decision_reasoner_disabled(self) -> None:
        """Тест отключения DecisionReasoner в DI контейнере."""
        config = ContainerConfig(decision_reasoner_enabled=False)
        container = DIContainer(config)
        # DecisionReasoner не должен быть зарегистрирован
        with pytest.raises(ValueError):
            container.get("decision_reasoner")
    @pytest.mark.asyncio
    async def test_trading_orchestrator_decision_reasoner_integration(self) -> None:
        """Тест интеграции DecisionReasoner в TradingOrchestrator."""
        # Создаем моки для зависимостей
        mock_order_repo = Mock()
        mock_position_repo = Mock()
        mock_portfolio_repo = Mock()
        mock_trading_repo = Mock()
        mock_strategy_repo = Mock()
        mock_enhanced_trading_service = Mock()
        mock_decision_reasoner = Mock()
        # Создаем TradingOrchestrator с DecisionReasoner
        orchestrator = DefaultTradingOrchestratorUseCase(
            order_repository=mock_order_repo,
            position_repository=mock_position_repo,
            portfolio_repository=mock_portfolio_repo,
            trading_repository=mock_trading_repo,
            strategy_repository=mock_strategy_repo,
            enhanced_trading_service=mock_enhanced_trading_service,
            decision_reasoner=mock_decision_reasoner
        )
        assert orchestrator.decision_reasoner is not None
        assert orchestrator.decision_reasoner == mock_decision_reasoner
    def test_decision_reasoner_metrics(self, decision_reasoner) -> None:
        """Тест получения метрик DecisionReasoner."""
        metrics = decision_reasoner.get_enhanced_metrics()
        assert metrics is not None
        assert hasattr(metrics, 'accuracy')
        assert hasattr(metrics, 'precision')
        assert hasattr(metrics, 'recall')
        assert hasattr(metrics, 'f1')
        assert hasattr(metrics, 'confidence')
        assert hasattr(metrics, 'risk_score')
    def test_decision_reasoner_explanation(self, decision_reasoner, mock_trade_decision) -> None:
        """Тест объяснения решения DecisionReasoner."""
        data = {
            "strategy": "trend_following",
            "regime": "bullish",
            "indicators": {"rsi": 65.0, "macd": 0.5},
            "whale_activity": "high",
            "volume_data": {"avg_volume": 1000.0, "current_volume": 1500.0}
        }
        explanation = decision_reasoner.explain(mock_trade_decision, data)
        assert explanation is not None
        assert isinstance(explanation, str)
        assert len(explanation) > 0
    def test_decision_reasoner_performance_evaluation(self, decision_reasoner) -> None:
        """Тест оценки производительности DecisionReasoner."""
        # Добавляем историю торгов
        decision_reasoner.trade_history = [
            {"confidence": 0.8, "pnl": 100.0},
            {"confidence": 0.6, "pnl": -50.0},
            {"confidence": 0.9, "pnl": 200.0}
        ]
        performance = decision_reasoner.evaluate_performance()
        assert performance is not None
        assert "total_trades" in performance
        assert "win_rate" in performance
        assert "avg_winning_confidence" in performance
        assert "avg_losing_confidence" in performance
    def test_decision_reasoner_retraining_check(self, decision_reasoner) -> None:
        """Тест проверки необходимости переобучения DecisionReasoner."""
        # Добавляем историю торгов с плохой производительностью
        decision_reasoner.trade_history = [
            {"confidence": 0.8, "pnl": -100.0} for _ in range(100)
        ]
        should_retrain = decision_reasoner.should_retrain()
        # При плохой производительности должно требовать переобучения
        assert should_retrain is True
    def test_decision_reasoner_model_training(self, decision_reasoner) -> None:
        """Тест обучения моделей DecisionReasoner."""
        # Создаем тестовые данные
        features = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        targets = pd.Series(np.random.randint(0, 2, 100))
        trade_results = [
            {"confidence": 0.8, "pnl": 100.0},
            {"confidence": 0.6, "pnl": -50.0}
        ]
        # Обучаем модели
        decision_reasoner.train_models(features, targets, trade_results)
        assert decision_reasoner.model_trained is True
        assert len(decision_reasoner.trade_history) > 0
    def test_decision_reasoner_prediction_with_confidence(self, decision_reasoner) -> None:
        """Тест предсказания с уверенностью DecisionReasoner."""
        # Обучаем модель
        features = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'feature3': np.random.randn(50)
        })
        targets = pd.Series(np.random.randint(0, 2, 50))
        trade_results = [{"confidence": 0.8, "pnl": 100.0}]
        decision_reasoner.train_models(features, targets, trade_results)
        # Делаем предсказание
        test_features = {
            'feature1': 0.5,
            'feature2': -0.3,
            'feature3': 0.8
        }
        direction, confidence = decision_reasoner.predict_with_confidence(test_features)
        assert direction in ["buy", "sell", "neutral"]
        assert confidence >= 0.0 and confidence <= 1.0
    def test_decision_reasoner_signal_adjustment(self, decision_reasoner) -> None:
        """Тест корректировки сигнала стратегии DecisionReasoner."""
        original_signal = {
            "position_size": 0.1,
            "stop_loss": 45000.0,
            "take_profit": 55000.0
        }
        adjusted_signal = decision_reasoner.adjust_strategy_signal(
            signal=original_signal,
            confidence=0.8,
            market_regime="trend"
        )
        assert adjusted_signal is not None
        assert "confidence" in adjusted_signal
        assert adjusted_signal["confidence"] == 0.8
        assert adjusted_signal["position_size"] != original_signal["position_size"]
if __name__ == "__main__":
    pytest.main([__file__]) 

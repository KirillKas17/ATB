"""
Интеграционные тесты для эволюционных агентов.
Проверка согласованности с модульной архитектурой.
"""
import asyncio
import os
import tempfile
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import pandas as pd
from shared.numpy_utils import np
from infrastructure.agents.evolvable_market_maker import EvolvableMarketMakerAgent
from infrastructure.agents.evolvable_risk_agent import EvolvableRiskAgent
from infrastructure.agents.evolvable_portfolio_agent import EvolvablePortfolioAgent
from infrastructure.agents.evolvable_news_agent import EvolvableNewsAgent
from infrastructure.agents.evolvable_market_regime import EvolvableMarketRegimeAgent
from infrastructure.agents.evolvable_strategy_agent import EvolvableStrategyAgent
from infrastructure.agents.evolvable_order_executor import EvolvableOrderExecutor
from infrastructure.agents.evolvable_meta_controller import EvolvableMetaController
from infrastructure.agents.agent_context_refactored import AgentContext
class TestEvolutionAgentsIntegration:
    """Тесты интеграции эволюционных агентов"""
    @pytest.fixture
    def sample_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание тестовых рыночных данных"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        data = {
            'open': np.random.uniform(45000, 55000, 100),
            'high': np.random.uniform(45000, 55000, 100),
            'low': np.random.uniform(45000, 55000, 100),
            'close': np.random.uniform(45000, 55000, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }
        return pd.DataFrame(data, index=dates)
    @pytest.fixture
    def sample_strategy_signals(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание тестовых сигналов стратегий"""
        return {
            'trend_strategy': {
                'direction': 'buy',
                'confidence': 0.8,
                'strength': 0.7,
                'priority': 1
            },
            'momentum_strategy': {
                'direction': 'sell',
                'confidence': 0.6,
                'strength': 0.5,
                'priority': 2
            },
            'mean_reversion_strategy': {
                'direction': 'buy',
                'confidence': 0.4,
                'strength': 0.3,
                'priority': 3
            }
        }
    @pytest.fixture
    def sample_risk_metrics(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание тестовых метрик риска"""
        return {
            'var_95': 0.02,
            'var_99': 0.03,
            'max_drawdown': 0.05,
            'volatility': 0.025,
            'exposure_level': 0.6,
            'confidence_score': 0.7,
            'kelly_criterion': 0.15
        }
    @pytest.fixture
    def temp_dir(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Временная директория для тестов"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    @pytest.mark.asyncio
    async def test_market_maker_agent_integration(self, sample_market_data) -> None:
        """Тест интеграции EvolvableMarketMakerAgent"""
        agent = EvolvableMarketMakerAgent()
        # Тест адаптации
        data = {"market_data": sample_market_data}
        result = await agent.adapt(data)
        assert result is True
        # Тест обучения
        result = await agent.learn(data)
        assert result is True
        # Тест производительности
        performance = agent.get_performance()
        assert 0.0 <= performance <= 1.0
        # Тест уверенности
        confidence = agent.get_confidence()
        assert 0.0 <= confidence <= 1.0
    @pytest.mark.asyncio
    async def test_risk_agent_integration(self, sample_market_data, sample_risk_metrics) -> None:
        """Тест интеграции EvolvableRiskAgent"""
        agent = EvolvableRiskAgent()
        # Тест адаптации
        data = {
            "market_data": sample_market_data,
            "risk_metrics": sample_risk_metrics
        }
        result = await agent.adapt(data)
        assert result is True
        # Тест обучения
        result = await agent.learn(data)
        assert result is True
        # Тест производительности
        performance = agent.get_performance()
        assert 0.0 <= performance <= 1.0
    @pytest.mark.asyncio
    async def test_portfolio_agent_integration(self, sample_market_data) -> None:
        """Тест интеграции EvolvablePortfolioAgent"""
        agent = EvolvablePortfolioAgent()
        current_weights = {
            'BTC': 0.4,
            'ETH': 0.3,
            'ADA': 0.2,
            'DOT': 0.1
        }
        # Тест адаптации
        data = {
            "market_data": sample_market_data,
            "current_weights": current_weights
        }
        result = await agent.adapt(data)
        assert result is True
        # Тест обучения
        result = await agent.learn(data)
        assert result is True
        # Тест расчета весов
        assets = ['BTC', 'ETH', 'ADA', 'DOT']
        market_data_dict = {'BTC': sample_market_data, 'ETH': sample_market_data}
        weights = await agent.calculate_weights(assets, market_data_dict, current_weights)
        assert isinstance(weights, dict)
        assert 'optimal_weights' in weights
    @pytest.mark.asyncio
    async def test_news_agent_integration(self, sample_market_data) -> None:
        """Тест интеграции EvolvableNewsAgent"""
        agent = EvolvableNewsAgent()
        news_data = {
            "sentiment_score": 0.6,
            "news_volume": 100,
            "social_sentiment": 0.7,
            "breaking_news": False
        }
        # Тест адаптации
        data = {
            "market_data": sample_market_data,
            "news_data": news_data
        }
        result = await agent.adapt(data)
        assert result is True
        # Тест обучения
        result = await agent.learn(data)
        assert result is True
        # Тест анализа настроений
        sentiment = await agent.analyze_sentiment(news_data)
        assert isinstance(sentiment, dict)
        assert 'sentiment_score' in sentiment
    @pytest.mark.asyncio
    async def test_market_regime_agent_integration(self, sample_market_data) -> None:
        """Тест интеграции EvolvableMarketRegimeAgent"""
        agent = EvolvableMarketRegimeAgent()
        # Тест адаптации
        data = {"market_data": sample_market_data}
        result = await agent.adapt(data)
        assert result is True
        # Тест обучения
        result = await agent.learn(data)
        assert result is True
        # Тест определения режима
        regime = await agent.detect_regime(sample_market_data)
        assert isinstance(regime, dict)
        assert 'regime_type' in regime
        assert 'confidence' in regime
    @pytest.mark.asyncio
    async def test_strategy_agent_integration(self, sample_market_data, sample_strategy_signals) -> None:
        """Тест интеграции EvolvableStrategyAgent"""
        agent = EvolvableStrategyAgent()
        # Тест адаптации
        data = {
            "market_data": sample_market_data,
            "strategy_signals": sample_strategy_signals
        }
        result = await agent.adapt(data)
        assert result is True
        # Тест обучения
        result = await agent.learn(data)
        assert result is True
        # Тест выбора стратегии
        strategy = await agent.select_strategy(sample_market_data, sample_strategy_signals)
        assert isinstance(strategy, dict)
        assert 'selected_strategy' in strategy
        assert 'confidence' in strategy
    @pytest.mark.asyncio
    async def test_order_executor_integration(self, sample_market_data) -> None:
        """Тест интеграции EvolvableOrderExecutor"""
        agent = EvolvableOrderExecutor()
        order_data = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": 0.1,
            "price": 50000
        }
        # Тест адаптации
        data = {
            "market_data": sample_market_data,
            "order_data": order_data
        }
        result = await agent.adapt(data)
        assert result is True
        # Тест обучения
        result = await agent.learn(data)
        assert result is True
        # Тест оптимизации исполнения
        execution = await agent.optimize_execution(order_data, sample_market_data)
        assert isinstance(execution, dict)
        assert 'price_offset' in execution
        assert 'size_adjustment' in execution
    @pytest.mark.asyncio
    async def test_meta_controller_integration(self, sample_market_data, sample_strategy_signals, sample_risk_metrics) -> None:
        """Тест интеграции EvolvableMetaController"""
        agent = EvolvableMetaController()
        # Тест адаптации
        data = {
            "market_data": sample_market_data,
            "strategy_signals": sample_strategy_signals,
            "risk_metrics": sample_risk_metrics
        }
        result = await agent.adapt(data)
        assert result is True
        # Тест обучения
        result = await agent.learn(data)
        assert result is True
        # Тест координации стратегий
        coordination = await agent.coordinate_strategies(
            "BTCUSDT", sample_market_data, sample_strategy_signals, sample_risk_metrics
        )
        assert isinstance(coordination, dict)
        # Тест оптимизации решений
        decision = await agent.optimize_decision(sample_market_data, sample_strategy_signals, sample_risk_metrics)
        assert isinstance(decision, dict)
        assert 'strategy_weight' in decision
        assert 'risk_level' in decision
    @pytest.mark.asyncio
    async def test_agent_state_persistence(self, temp_dir) -> None:
        """Тест сохранения и загрузки состояния агентов"""
        agents = [
            EvolvableMarketMakerAgent(),
            EvolvableRiskAgent(),
            EvolvablePortfolioAgent(),
            EvolvableNewsAgent(),
            EvolvableMarketRegimeAgent(),
            EvolvableStrategyAgent(),
            EvolvableOrderExecutor(),
            EvolvableMetaController()
        ]
        for i, agent in enumerate(agents):
            # Сохранение состояния
            save_path = os.path.join(temp_dir, f"agent_{i}_state.pkl")
            result = agent.save_state(save_path)
            assert result is True
            assert os.path.exists(save_path)
            # Загрузка состояния
            result = agent.load_state(save_path)
            assert result is True
    @pytest.mark.asyncio
    async def test_agent_evolution_cycle(self, sample_market_data) -> None:
        """Тест полного цикла эволюции агентов"""
        agents = [
            EvolvableMarketMakerAgent(),
            EvolvableRiskAgent(),
            EvolvablePortfolioAgent(),
            EvolvableNewsAgent(),
            EvolvableMarketRegimeAgent(),
            EvolvableStrategyAgent(),
            EvolvableOrderExecutor(),
            EvolvableMetaController()
        ]
        for agent in agents:
            data = {"market_data": sample_market_data}
            # Цикл эволюции
            result = await agent.adapt(data)
            assert result is True
            result = await agent.learn(data)
            assert result is True
            result = await agent.evolve(data)
            assert result is True
            # Проверка метрик
            performance = agent.get_performance()
            confidence = agent.get_confidence()
            assert 0.0 <= performance <= 1.0
            assert 0.0 <= confidence <= 1.0
    @pytest.mark.asyncio
    async def test_agent_integration_with_modular_architecture(self, sample_market_data) -> None:
        """Тест интеграции с модульной архитектурой"""
        # Проверка, что эволюционные агенты используют модульные компоненты
        market_maker = EvolvableMarketMakerAgent()
        assert hasattr(market_maker, 'market_maker_agent')
        assert market_maker.market_maker_agent is not None
        risk_agent = EvolvableRiskAgent()
        assert hasattr(risk_agent, 'risk_agent')
        assert risk_agent.risk_agent is not None
        portfolio_agent = EvolvablePortfolioAgent()
        assert hasattr(portfolio_agent, 'portfolio_agent')
        assert portfolio_agent.portfolio_agent is not None
        meta_controller = EvolvableMetaController()
        assert hasattr(meta_controller, 'meta_controller')
        assert meta_controller.meta_controller is not None
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, sample_market_data) -> None:
        """Тест обработки ошибок в агентах"""
        agent = EvolvableMarketMakerAgent()
        # Тест с некорректными данными
        result = await agent.adapt(None)
        assert result is False
        result = await agent.learn({})
        assert result is False
        # Тест с пустыми данными
        result = await agent.adapt({"market_data": pd.DataFrame()})
        assert result is False
    @pytest.mark.asyncio
    async def test_agent_performance_monitoring(self, sample_market_data) -> None:
        """Тест мониторинга производительности агентов"""
        agent = EvolvableMetaController()
        # Инициализация метрик
        initial_performance = agent.get_performance()
        initial_confidence = agent.get_confidence()
        # Обучение
        data = {"market_data": sample_market_data}
        await agent.learn(data)
        # Проверка изменения метрик
        new_performance = agent.get_performance()
        new_confidence = agent.get_confidence()
        # Метрики должны измениться после обучения
        assert new_performance != initial_performance or new_confidence != initial_confidence
    @pytest.mark.asyncio
    async def test_agent_concurrent_operations(self, sample_market_data) -> None:
        """Тест параллельных операций агентов"""
        agent = EvolvableMarketMakerAgent()
        # Параллельные операции адаптации и обучения
        tasks = [
            agent.adapt({"market_data": sample_market_data}),
            agent.learn({"market_data": sample_market_data}),
            agent.adapt({"market_data": sample_market_data}),
            agent.learn({"market_data": sample_market_data})
        ]
        results = await asyncio.gather(*tasks)
        # Все операции должны завершиться успешно
        assert all(results)
    @pytest.mark.asyncio
    async def test_agent_memory_management(self, sample_market_data) -> None:
        """Тест управления памятью агентов"""
        agent = EvolvableMarketMakerAgent()
        # Добавление большого количества данных
        for i in range(100):
            data = {"market_data": sample_market_data}
            await agent.learn(data)
        # Проверка ограничения размера истории
        assert len(agent.training_data) <= agent.max_training_samples
    @pytest.mark.asyncio
    async def test_agent_ml_model_evolution(self, sample_market_data) -> None:
        """Тест эволюции ML моделей агентов"""
        agent = EvolvableMarketMakerAgent()
        # Инициализация модели
        initial_model_state = agent.ml_model.state_dict()
        # Эволюция
        data = {"market_data": sample_market_data}
        await agent.evolve(data)
        # Проверка изменения модели
        new_model_state = agent.ml_model.state_dict()
        # Модель должна измениться после эволюции
        assert initial_model_state != new_model_state
    @pytest.mark.asyncio
    async def test_agent_configuration_evolution(self, sample_market_data) -> None:
        """Тест эволюции конфигурации агентов"""
        agent = EvolvableMarketMakerAgent()
        # Инициализация конфигурации
        if hasattr(agent, 'market_maker_agent') and hasattr(agent.market_maker_agent, 'config'):
            initial_config = agent.market_maker_agent.config.copy()
            # Эволюция
            data = {"market_data": sample_market_data}
            await agent.evolve(data)
            # Проверка изменения конфигурации
            new_config = agent.market_maker_agent.config
            assert initial_config != new_config
    @pytest.mark.asyncio
    async def test_agent_feature_extraction(self, sample_market_data) -> None:
        """Тест экстракции признаков агентов"""
        agent = EvolvableMarketMakerAgent()
        # Тест экстракции признаков
        features = agent._extract_features(sample_market_data, {})
        # Проверка корректности признаков
        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(f, (int, float)) for f in features)
        assert not any(pd.isna(f) for f in features)
    @pytest.mark.asyncio
    async def test_agent_target_extraction(self, sample_market_data) -> None:
        """Тест экстракции целевых значений агентов"""
        agent = EvolvableMarketMakerAgent()
        # Тест экстракции целевых значений
        targets = agent._extract_targets({}, {})
        # Проверка корректности целевых значений
        assert isinstance(targets, dict)
        assert len(targets) > 0
        assert all(isinstance(v, (int, float)) for v in targets.values())
    @pytest.mark.asyncio
    async def test_agent_metrics_update(self, sample_market_data) -> None:
        """Тест обновления метрик агентов"""
        agent = EvolvableMarketMakerAgent()
        # Инициализация метрик
        initial_performance = agent.performance_metric
        initial_confidence = agent.confidence_metric
        # Обновление метрик
        agent._update_metrics(0.1, np.array([0.8, 0.7, 0.6]))
        # Проверка изменения метрик
        assert agent.performance_metric != initial_performance
        assert agent.confidence_metric != initial_confidence
    @pytest.mark.asyncio
    def test_agent_registration_in_evolution_manager(self: "TestEvolutionAgentsIntegration") -> None:
        """Тест регистрации агентов в эволюционном менеджере"""
        from infrastructure.core.evolution_manager import get_evolution_manager
        # Создание агентов
        agents = [
            EvolvableMarketMakerAgent(),
            EvolvableRiskAgent(),
            EvolvablePortfolioAgent(),
            EvolvableNewsAgent(),
            EvolvableMarketRegimeAgent(),
            EvolvableStrategyAgent(),
            EvolvableOrderExecutor(),
            EvolvableMetaController()
        ]
        # Проверка регистрации
        evolution_manager = get_evolution_manager()
        registered_components = evolution_manager.get_components()
        for agent in agents:
            assert agent.name in [comp.name for comp in registered_components]
    @pytest.mark.asyncio
    async def test_agent_comprehensive_integration(self, sample_market_data, sample_strategy_signals, sample_risk_metrics) -> None:
        """Комплексный тест интеграции всех агентов"""
        # Создание всех агентов
        agents = {
            'market_maker': EvolvableMarketMakerAgent(),
            'risk': EvolvableRiskAgent(),
            'portfolio': EvolvablePortfolioAgent(),
            'news': EvolvableNewsAgent(),
            'market_regime': EvolvableMarketRegimeAgent(),
            'strategy': EvolvableStrategyAgent(),
            'order_executor': EvolvableOrderExecutor(),
            'meta_controller': EvolvableMetaController()
        }
        # Комплексные данные
        data = {
            "market_data": sample_market_data,
            "strategy_signals": sample_strategy_signals,
            "risk_metrics": sample_risk_metrics,
            "current_weights": {'BTC': 0.5, 'ETH': 0.5},
            "news_data": {"sentiment_score": 0.6, "news_volume": 100}
        }
        # Тест всех агентов
        for name, agent in agents.items():
            # Адаптация
            result = await agent.adapt(data)
            assert result is True, f"Adaptation failed for {name}"
            # Обучение
            result = await agent.learn(data)
            assert result is True, f"Learning failed for {name}"
            # Производительность
            performance = agent.get_performance()
            assert 0.0 <= performance <= 1.0, f"Invalid performance for {name}"
            # Уверенность
            confidence = agent.get_confidence()
            assert 0.0 <= confidence <= 1.0, f"Invalid confidence for {name}"
        # Тест взаимодействия между агентами
        meta_controller = agents['meta_controller']
        coordination = await meta_controller.coordinate_strategies(
            "BTCUSDT", sample_market_data, sample_strategy_signals, sample_risk_metrics
        )
        assert isinstance(coordination, dict)
        assert 'evolution_metrics' in coordination
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 

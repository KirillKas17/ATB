"""
Тесты интеграции агентов с BaseAgent
"""
import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Dict, Any
from infrastructure.agents.agent_risk import RiskAgent
from infrastructure.agents.agent_news import NewsAgent
from infrastructure.agents.social_media_agent import SocialMediaAgent
from infrastructure.agents.agent_market_regime import MarketRegimeAgent
from infrastructure.agents.agent_order_executor import OrderExecutorAgent
from infrastructure.agents.agent_portfolio import PortfolioAgent
from infrastructure.agents.agent_whales import WhalesAgent
from infrastructure.agents.agent_meta_controller import MetaControllerAgent
from infrastructure.agents.agent_market_maker_model import MarketMakerModelAgent
from domain.type_definitions.agent_types import AgentStatus
class TestBaseAgentIntegration:
    """Тесты базовой интеграции агентов с BaseAgent"""
    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Образец конфигурации для тестов"""
        return {
            "test_param": 1.0,
            "cache_size": 100,
            "update_interval": 60
        }
    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Образец рыночных данных для тестов"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        data = {
            'open': np.random.uniform(50000, 51000, 100),
            'high': np.random.uniform(51000, 52000, 100),
            'low': np.random.uniform(49000, 50000, 100),
            'close': np.random.uniform(50000, 51000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        }
        return pd.DataFrame(data, index=dates)
    @pytest.mark.asyncio
    async def test_risk_agent_integration(self, sample_config) -> None:
        """Тест интеграции RiskAgent с BaseAgent"""
        agent = RiskAgent(sample_config)
        # Проверка инициализации
        assert agent.name == "RiskAgent"
        assert agent.config == sample_config
        assert agent.state.status == AgentStatus.INITIALIZING
        # Тест инициализации
        success = await agent.initialize()
        assert success is True
        assert agent.state.status == AgentStatus.HEALTHY
        assert agent.state.confidence > 0
        # Тест обработки данных
        test_data = {
            "market_data": {"BTC/USDT": pd.DataFrame()},
            "strategy_confidence": {"BTC/USDT": 0.8},
            "current_positions": {"BTC/USDT": 0.1}
        }
        result = await agent.process(test_data)
        assert "allocations" in result
        assert "portfolio_metrics" in result
        assert "risk_level" in result
        # Тест очистки
        await agent.cleanup()
        assert len(agent.risk_metrics) == 0
    @pytest.mark.asyncio
    async def test_news_agent_integration(self, sample_config) -> None:
        """Тест интеграции NewsAgent с BaseAgent"""
        agent = NewsAgent(sample_config)
        # Проверка инициализации
        assert agent.name == "NewsAgent"
        assert agent.config == sample_config
        # Тест инициализации
        success = await agent.initialize()
        assert success is True
        assert agent.state.status == AgentStatus.HEALTHY
        # Тест обработки данных
        test_data = {"pair": "BTC/USDT"}
        result = await agent.process(test_data)
        assert "news" in result
        assert "sentiment" in result
        # Тест очистки
        await agent.cleanup()
        assert len(agent.news_history) == 0
    @pytest.mark.asyncio
    async def test_social_media_agent_integration(self, sample_config) -> None:
        """Тест интеграции SocialMediaAgent с BaseAgent"""
        agent = SocialMediaAgent(sample_config)
        # Проверка инициализации
        assert agent.name == "SocialMediaAgent"
        assert agent.config == sample_config
        # Тест инициализации
        success = await agent.initialize()
        assert success is True
        assert agent.state.status == AgentStatus.HEALTHY
        # Тест обработки данных
        test_data = {"symbol": "BTC"}
        result = await agent.process(test_data)
        assert "sentiment" in result
        assert "fear_greed_index" in result
        # Тест очистки
        await agent.cleanup()
        assert len(agent.sentiment_cache) == 0
    @pytest.mark.asyncio
    async def test_market_regime_agent_integration(self, sample_config, sample_market_data) -> None:
        """Тест интеграции MarketRegimeAgent с BaseAgent"""
        agent = MarketRegimeAgent(sample_config)
        # Проверка инициализации
        assert agent.name == "MarketRegimeAgent"
        assert agent.config == sample_config
        # Тест инициализации
        success = await agent.initialize()
        assert success is True
        assert agent.state.status == AgentStatus.HEALTHY
        # Тест обработки данных
        test_data = {
            "market_data": sample_market_data,
            "symbol": "BTC/USDT"
        }
        result = await agent.process(test_data)
        assert "regime" in result
        assert "confidence" in result
        assert "indicators" in result
        # Тест очистки
        await agent.cleanup()
        assert len(agent.regime_history) == 0
    @pytest.mark.asyncio
    async def test_order_executor_agent_integration(self, sample_config) -> None:
        """Тест интеграции OrderExecutorAgent с BaseAgent"""
        agent = OrderExecutorAgent(sample_config)
        # Проверка инициализации
        assert agent.name == "OrderExecutorAgent"
        assert agent.config == sample_config
        # Тест инициализации
        success = await agent.initialize()
        assert success is True
        assert agent.state.status == AgentStatus.HEALTHY
        # Тест обработки данных
        test_data = {
            "action": "place_order",
            "order_params": {
                "symbol": "BTC/USDT",
                "direction": "buy",
                "amount": 0.1,
                "entry_price": 50000
            }
        }
        result = await agent.process(test_data)
        assert "success" in result
        # Тест очистки
        await agent.cleanup()
        assert len(agent.active_orders) == 0
    @pytest.mark.asyncio
    async def test_portfolio_agent_integration(self, sample_config, sample_market_data) -> None:
        """Тест интеграции PortfolioAgent с BaseAgent"""
        agent = PortfolioAgent(sample_config)
        # Проверка инициализации
        assert agent.name == "PortfolioAgent"
        assert agent.config == sample_config
        # Тест инициализации
        success = await agent.initialize()
        assert success is True
        assert agent.state.status == AgentStatus.HEALTHY
        # Тест обработки данных
        test_data = {
            "market_data": {"BTC/USDT": sample_market_data},
            "risk_data": {"BTC/USDT": {"var": 0.02}},
            "backtest_results": {"BTC/USDT": {"sharpe": 1.5}}
        }
        result = await agent.process(test_data)
        assert "weights" in result
        assert "trades" in result
        assert "portfolio_state" in result
        # Тест очистки
        await agent.cleanup()
        assert len(agent.cache.metrics) == 0
    @pytest.mark.asyncio
    async def test_whales_agent_integration(self, sample_config) -> None:
        """Тест интеграции WhalesAgent с BaseAgent"""
        agent = WhalesAgent(sample_config)
        # Проверка инициализации
        assert agent.name == "WhalesAgent"
        assert agent.config == sample_config
        # Тест инициализации
        success = await agent.initialize()
        assert success is True
        assert agent.state.status == AgentStatus.HEALTHY
        # Тест обработки данных
        test_data = {
            "pair": "BTC/USDT",
            "market_data": pd.DataFrame(),
            "order_book": {"bids": [], "asks": []}
        }
        result = await agent.process(test_data)
        assert "whale_activities" in result
        assert "whale_patterns" in result
        assert "signals" in result
        # Тест очистки
        await agent.cleanup()
        assert len(agent.activity_cache.activities) == 0
    @pytest.mark.asyncio
    async def test_meta_controller_agent_integration(self, sample_config) -> None:
        """Тест интеграции MetaControllerAgent с BaseAgent"""
        agent = MetaControllerAgent(sample_config)
        # Проверка инициализации
        assert agent.name == "MetaControllerAgent"
        assert agent.config == sample_config
        # Тест инициализации
        success = await agent.initialize()
        assert success is True
        assert agent.state.status == AgentStatus.HEALTHY
        # Тест обработки данных
        test_data = {
            "action": "evaluate_strategies",
            "symbol": "BTC/USDT"
        }
        result = await agent.process(test_data)
        assert result is not None
        # Тест очистки
        await agent.cleanup()
        assert len(agent.strategies) == 0
    @pytest.mark.asyncio
    async def test_market_maker_model_agent_integration(self, sample_config) -> None:
        """Тест интеграции MarketMakerModelAgent с BaseAgent"""
        agent = MarketMakerModelAgent(sample_config)
        # Проверка инициализации
        assert agent.name == "MarketMakerModelAgent"
        assert agent.config == sample_config
        # Тест инициализации
        success = await agent.initialize()
        assert success is True
        assert agent.state.status == AgentStatus.HEALTHY
        # Тест обработки данных
        test_data = {
            "symbol": "BTC/USDT",
            "market_data": pd.DataFrame(),
            "order_book": {"bids": [], "asks": []},
            "trades": []
        }
        result = await agent.process(test_data)
        assert "spread_analysis" in result
        assert "liquidity_analysis" in result
        assert "fakeouts" in result
        # Тест очистки
        await agent.cleanup()
        assert len(agent.cache_service._cache) == 0
class TestAgentErrorHandling:
    """Тесты обработки ошибок в агентах"""
    @pytest.mark.asyncio
    async def test_invalid_config_handling(self) -> None:
        """Тест обработки неверной конфигурации"""
        invalid_config = {"invalid_param": -1}
        agent = RiskAgent(invalid_config)
        success = await agent.initialize()
        assert success is False
        assert agent.state.status == AgentStatus.ERROR
    @pytest.mark.asyncio
    async def test_invalid_data_handling(self) -> None:
        """Тест обработки неверных данных"""
        agent = RiskAgent()
        await agent.initialize()
        # Тест с неверным форматом данных
        result = await agent.process("invalid_data")
        assert "error" in result
        # Тест с пустыми данными
        result = await agent.process({})
        assert "error" in result
    @pytest.mark.asyncio
    async def test_agent_recovery(self) -> None:
        """Тест восстановления агента после ошибки"""
        agent = RiskAgent()
        # Сначала вызываем ошибку
        result = await agent.process("invalid_data")
        assert "error" in result
        # Затем отправляем правильные данные
        test_data = {
            "market_data": {"BTC/USDT": pd.DataFrame()},
            "strategy_confidence": {"BTC/USDT": 0.8},
            "current_positions": {"BTC/USDT": 0.1}
        }
        result = await agent.process(test_data)
        assert "error" not in result
class TestAgentMetrics:
    """Тесты метрик агентов"""
    @pytest.mark.asyncio
    async def test_agent_performance_metrics(self) -> None:
        """Тест метрик производительности агентов"""
        agent = RiskAgent()
        await agent.initialize()
        # Выполняем несколько операций
        for i in range(5):
            test_data = {
                "market_data": {"BTC/USDT": pd.DataFrame()},
                "strategy_confidence": {"BTC/USDT": 0.8},
                "current_positions": {"BTC/USDT": 0.1}
            }
            await agent.process(test_data)
        # Проверяем метрики
        assert agent.state.total_operations > 0
        assert agent.state.successful_operations > 0
        assert agent.state.average_processing_time > 0
    @pytest.mark.asyncio
    async def test_agent_confidence_tracking(self) -> None:
        """Тест отслеживания уверенности агентов"""
        agent = RiskAgent()
        await agent.initialize()
        # Проверяем начальную уверенность
        initial_confidence = agent.state.confidence
        # Выполняем операции
        test_data = {
            "market_data": {"BTC/USDT": pd.DataFrame()},
            "strategy_confidence": {"BTC/USDT": 0.8},
            "current_positions": {"BTC/USDT": 0.1}
        }
        await agent.process(test_data)
        # Уверенность должна измениться
        assert agent.state.confidence != initial_confidence
class TestAgentIntegration:
    """Тесты интеграции между агентами"""
    @pytest.mark.asyncio
    async def test_agent_dependency_injection(self) -> None:
        """Тест внедрения зависимостей между агентами"""
        # Создаем агенты
        risk_agent = RiskAgent()
        market_regime_agent = MarketRegimeAgent()
        order_executor = OrderExecutorAgent()
        # Инициализируем
        await risk_agent.initialize()
        await market_regime_agent.initialize()
        await order_executor.initialize()
        # Устанавливаем зависимости
        order_executor.set_risk_agent(risk_agent)
        order_executor.set_market_regime_agent(market_regime_agent)
        # Проверяем, что зависимости установлены
        assert order_executor.risk_agent is risk_agent
        assert order_executor.market_regime_agent is market_regime_agent
    @pytest.mark.asyncio
    async def test_agent_data_flow(self) -> None:
        """Тест потока данных между агентами"""
        # Создаем цепочку агентов
        market_regime_agent = MarketRegimeAgent()
        portfolio_agent = PortfolioAgent()
        await market_regime_agent.initialize()
        await portfolio_agent.initialize()
        # Устанавливаем зависимости
        portfolio_agent.set_market_regime_agent(market_regime_agent)
        # Тестируем поток данных
        sample_data = pd.DataFrame({
            'open': [50000, 50100, 50200],
            'high': [50100, 50200, 50300],
            'low': [49900, 50000, 50100],
            'close': [50050, 50150, 50250],
            'volume': [100, 150, 200]
        })
        test_data = {
            "market_data": {"BTC/USDT": sample_data},
            "risk_data": {"BTC/USDT": {"var": 0.02}},
            "backtest_results": {"BTC/USDT": {"sharpe": 1.5}}
        }
        result = await portfolio_agent.process(test_data)
        assert "weights" in result
        assert "portfolio_state" in result
if __name__ == "__main__":
    pytest.main([__file__]) 

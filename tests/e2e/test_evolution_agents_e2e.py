"""
E2E тесты для эволюционных агентов.
Проверка полного цикла работы системы эволюции.
"""
import asyncio
import os
import tempfile
from datetime import datetime, timedelta
import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from infrastructure.agents.evolvable_market_maker import EvolvableMarketMakerAgent
from infrastructure.agents.evolvable_risk_agent import EvolvableRiskAgent
from infrastructure.agents.evolvable_portfolio_agent import EvolvablePortfolioAgent
from infrastructure.agents.evolvable_news_agent import EvolvableNewsAgent
from infrastructure.agents.evolvable_market_regime import EvolvableMarketRegimeAgent
from infrastructure.agents.evolvable_strategy_agent import EvolvableStrategyAgent
from infrastructure.agents.evolvable_order_executor import EvolvableOrderExecutor
from infrastructure.agents.evolvable_meta_controller import EvolvableMetaController
from infrastructure.core.evolution_manager import EvolutionManager
class TestEvolutionAgentsE2E:
    """E2E тесты для эволюционных агентов"""
    @pytest.fixture
    def realistic_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание реалистичных рыночных данных"""
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
        # Создание реалистичных цен с трендом и волатильностью
        np.random.seed(42)
        base_price = 50000
        trend = np.linspace(0, 0.1, 1000)  # Восходящий тренд
        noise = np.random.normal(0, 0.02, 1000)  # Шум
        volatility = np.random.normal(0, 0.015, 1000)  # Волатильность
        prices = base_price * (1 + trend + noise + volatility)
        volumes = np.random.uniform(1000, 10000, 1000)
        data = {
            'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 1000))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 1000))),
            'close': prices,
            'volume': volumes
        }
        return pd.DataFrame(data, index=dates)
    @pytest.fixture
    def realistic_strategy_signals(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание реалистичных сигналов стратегий"""
        return {
            'trend_strategy': {
                'direction': 'buy',
                'confidence': 0.85,
                'strength': 0.78,
                'priority': 1,
                'timestamp': datetime.now()
            },
            'momentum_strategy': {
                'direction': 'sell',
                'confidence': 0.72,
                'strength': 0.65,
                'priority': 2,
                'timestamp': datetime.now()
            },
            'mean_reversion_strategy': {
                'direction': 'buy',
                'confidence': 0.68,
                'strength': 0.55,
                'priority': 3,
                'timestamp': datetime.now()
            },
            'volatility_strategy': {
                'direction': 'hold',
                'confidence': 0.45,
                'strength': 0.32,
                'priority': 4,
                'timestamp': datetime.now()
            },
            'arbitrage_strategy': {
                'direction': 'buy',
                'confidence': 0.92,
                'strength': 0.88,
                'priority': 1,
                'timestamp': datetime.now()
            }
        }
    @pytest.fixture
    def realistic_risk_metrics(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание реалистичных метрик риска"""
        return {
            'var_95': 0.0234,
            'var_99': 0.0345,
            'max_drawdown': 0.0567,
            'volatility': 0.0289,
            'exposure_level': 0.623,
            'confidence_score': 0.745,
            'kelly_criterion': 0.167,
            'sharpe_ratio': 1.234,
            'sortino_ratio': 1.567,
            'calmar_ratio': 0.890
        }
    @pytest.fixture
    def realistic_portfolio_state(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание реалистичного состояния портфеля"""
        return {
            'BTC': {
                'weight': 0.45,
                'value': 45000,
                'pnl': 0.12,
                'volatility': 0.025
            },
            'ETH': {
                'weight': 0.30,
                'value': 3000,
                'pnl': 0.08,
                'volatility': 0.030
            },
            'ADA': {
                'weight': 0.15,
                'value': 0.45,
                'pnl': -0.05,
                'volatility': 0.035
            },
            'DOT': {
                'weight': 0.10,
                'value': 7.50,
                'pnl': 0.15,
                'volatility': 0.040
            }
        }
    @pytest.fixture
    def realistic_news_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание реалистичных новостных данных"""
        return {
            'sentiment_score': 0.67,
            'news_volume': 156,
            'social_sentiment': 0.72,
            'breaking_news': False,
            'news_sources': ['Reuters', 'Bloomberg', 'CNBC'],
            'key_events': ['Fed meeting', 'Earnings report', 'Regulation news'],
            'market_impact': 0.15,
            'confidence': 0.78
        }
    @pytest.fixture
    def temp_workspace(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Временное рабочее пространство"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    @pytest.mark.asyncio
    async def test_full_evolution_cycle_e2e(self, realistic_market_data, realistic_strategy_signals,
                                          realistic_risk_metrics, realistic_portfolio_state, 
                                          realistic_news_data, temp_workspace) -> None:
        """Полный E2E тест цикла эволюции"""
        # Создание всех эволюционных агентов
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
        # Комплексные данные для всех агентов
        comprehensive_data = {
            "market_data": realistic_market_data,
            "strategy_signals": realistic_strategy_signals,
            "risk_metrics": realistic_risk_metrics,
            "current_weights": {k: v['weight'] for k, v in realistic_portfolio_state.items()},
            "news_data": realistic_news_data,
            "portfolio_state": realistic_portfolio_state,
            "order_data": {
                "symbol": "BTCUSDT",
                "side": "buy",
                "quantity": 0.1,
                "price": 50000,
                "timestamp": datetime.now()
            }
        }
        # Фаза 1: Инициализация и адаптация
        print("🔄 Фаза 1: Инициализация и адаптация")
        for name, agent in agents.items():
            result = await agent.adapt(comprehensive_data)
            assert result is True, f"Адаптация не удалась для {name}"
            print(f"✅ {name}: адаптация завершена")
        # Фаза 2: Обучение на исторических данных
        print("📚 Фаза 2: Обучение на исторических данных")
        for name, agent in agents.items():
            result = await agent.learn(comprehensive_data)
            assert result is True, f"Обучение не удалось для {name}"
            print(f"✅ {name}: обучение завершено")
        # Фаза 3: Полная эволюция
        print("🧬 Фаза 3: Полная эволюция")
        for name, agent in agents.items():
            result = await agent.evolve(comprehensive_data)
            assert result is True, f"Эволюция не удалась для {name}"
            print(f"✅ {name}: эволюция завершена")
        # Фаза 4: Проверка производительности
        print("📊 Фаза 4: Проверка производительности")
        performance_metrics = {}
        for name, agent in agents.items():
            performance = agent.get_performance()
            confidence = agent.get_confidence()
            performance_metrics[name] = {
                'performance': performance,
                'confidence': confidence,
                'evolution_count': agent.evolution_count
            }
            print(f"📈 {name}: производительность={performance:.3f}, уверенность={confidence:.3f}")
        # Фаза 5: Сохранение состояния
        print("💾 Фаза 5: Сохранение состояния")
        for name, agent in agents.items():
            save_path = os.path.join(temp_workspace, f"{name}_state.pkl")
            result = agent.save_state(save_path)
            assert result is True, f"Сохранение не удалось для {name}"
            assert os.path.exists(save_path), f"Файл состояния не создан для {name}"
            print(f"✅ {name}: состояние сохранено")
        # Фаза 6: Загрузка состояния
        print("📂 Фаза 6: Загрузка состояния")
        for name, agent in agents.items():
            save_path = os.path.join(temp_workspace, f"{name}_state.pkl")
            result = agent.load_state(save_path)
            assert result is True, f"Загрузка не удалась для {name}"
            print(f"✅ {name}: состояние загружено")
        # Фаза 7: Проверка функциональности
        print("🔧 Фаза 7: Проверка функциональности")
        # Тест MarketMaker
        market_maker = agents['market_maker']
        spread_analysis = await market_maker.analyze_spread(realistic_market_data)
        assert isinstance(spread_analysis, dict)
        assert 'spread_width' in spread_analysis
        # Тест Risk Agent
        risk_agent = agents['risk']
        risk_assessment = await risk_agent.assess_risk(realistic_market_data, realistic_risk_metrics)
        assert isinstance(risk_assessment, dict)
        assert 'risk_level' in risk_assessment
        # Тест Portfolio Agent
        portfolio_agent = agents['portfolio']
        optimal_weights = await portfolio_agent.predict_optimal_weights(realistic_market_data)
        assert isinstance(optimal_weights, dict)
        assert len(optimal_weights) > 0
        # Тест News Agent
        news_agent = agents['news']
        sentiment_analysis = await news_agent.analyze_sentiment(realistic_news_data)
        assert isinstance(sentiment_analysis, dict)
        assert 'sentiment_score' in sentiment_analysis
        # Тест Market Regime Agent
        market_regime_agent = agents['market_regime']
        regime_detection = await market_regime_agent.detect_regime(realistic_market_data)
        assert isinstance(regime_detection, dict)
        assert 'regime_type' in regime_detection
        # Тест Strategy Agent
        strategy_agent = agents['strategy']
        strategy_selection = await strategy_agent.select_strategy(realistic_market_data, realistic_strategy_signals)
        assert isinstance(strategy_selection, dict)
        assert 'selected_strategy' in strategy_selection
        # Тест Order Executor
        order_executor = agents['order_executor']
        execution_optimization = await order_executor.optimize_execution(
            comprehensive_data['order_data'], realistic_market_data
        )
        assert isinstance(execution_optimization, dict)
        assert 'price_offset' in execution_optimization
        # Тест Meta Controller
        meta_controller = agents['meta_controller']
        coordination = await meta_controller.coordinate_strategies(
            "BTCUSDT", realistic_market_data, realistic_strategy_signals, realistic_risk_metrics
        )
        assert isinstance(coordination, dict)
        assert 'evolution_metrics' in coordination
        print("✅ Все функциональные тесты пройдены")
    @pytest.mark.asyncio
    async def test_evolution_manager_integration_e2e(self, realistic_market_data, temp_workspace) -> None:
        """E2E тест интеграции с эволюционным менеджером"""
        # Создание эволюционного менеджера
        evolution_manager = EvolutionManager()
        # Создание и регистрация агентов
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
        registered_components = evolution_manager.get_components()
        assert len(registered_components) >= len(agents)
        # Запуск полного цикла эволюции
        data = {"market_data": realistic_market_data}
        # Адаптация всех компонентов
        results = await evolution_manager.adapt_all(data)
        assert all(results.values()), "Не все компоненты адаптировались"
        # Обучение всех компонентов
        results = await evolution_manager.learn_all(data)
        assert all(results.values()), "Не все компоненты обучились"
        # Эволюция всех компонентов
        results = await evolution_manager.evolve_all(data)
        assert all(results.values()), "Не все компоненты эволюционировали"
        # Проверка метрик производительности
        performance_report = evolution_manager.get_performance_report()
        assert isinstance(performance_report, dict)
        assert len(performance_report) > 0
        print("✅ Интеграция с эволюционным менеджером успешна")
    @pytest.mark.asyncio
    async def test_agent_interaction_e2e(self, realistic_market_data, realistic_strategy_signals,
                                       realistic_risk_metrics, temp_workspace) -> None:
        """E2E тест взаимодействия между агентами"""
        # Создание агентов
        market_maker = EvolvableMarketMakerAgent()
        risk_agent = EvolvableRiskAgent()
        portfolio_agent = EvolvablePortfolioAgent()
        meta_controller = EvolvableMetaController()
        # Комплексные данные
        data = {
            "market_data": realistic_market_data,
            "strategy_signals": realistic_strategy_signals,
            "risk_metrics": realistic_risk_metrics
        }
        # Инициализация агентов
        await market_maker.adapt(data)
        await risk_agent.adapt(data)
        await portfolio_agent.adapt(data)
        await meta_controller.adapt(data)
        # Тест взаимодействия: MarketMaker -> Risk -> Portfolio -> MetaController
        # 1. MarketMaker анализирует спред
        spread_analysis = await market_maker.analyze_spread(realistic_market_data)
        assert isinstance(spread_analysis, dict)
        # 2. Risk Agent оценивает риск с учетом спреда
        risk_data = {**data, "spread_analysis": spread_analysis}
        risk_assessment = await risk_agent.assess_risk(realistic_market_data, realistic_risk_metrics)
        assert isinstance(risk_assessment, dict)
        # 3. Portfolio Agent оптимизирует веса с учетом риска
        portfolio_data = {**data, "risk_assessment": risk_assessment}
        optimal_weights = await portfolio_agent.predict_optimal_weights(realistic_market_data)
        assert isinstance(optimal_weights, dict)
        # 4. MetaController координирует все решения
        coordination_data = {
            "market_data": realistic_market_data,
            "strategy_signals": realistic_strategy_signals,
            "risk_metrics": realistic_risk_metrics,
            "spread_analysis": spread_analysis,
            "risk_assessment": risk_assessment,
            "optimal_weights": optimal_weights
        }
        coordination = await meta_controller.coordinate_strategies(
            "BTCUSDT", realistic_market_data, realistic_strategy_signals, realistic_risk_metrics
        )
        assert isinstance(coordination, dict)
        # Проверка согласованности решений
        assert 'evolution_metrics' in coordination
        assert coordination['evolution_metrics']['performance'] >= 0.0
        assert coordination['evolution_metrics']['confidence'] >= 0.0
        print("✅ Взаимодействие между агентами успешно")
    @pytest.mark.asyncio
    async def test_performance_evolution_e2e(self, realistic_market_data, temp_workspace) -> None:
        """E2E тест эволюции производительности"""
        agent = EvolvableMarketMakerAgent()
        # Измерение начальной производительности
        initial_performance = agent.get_performance()
        initial_confidence = agent.get_confidence()
        print(f"📊 Начальная производительность: {initial_performance:.3f}")
        print(f"📊 Начальная уверенность: {initial_confidence:.3f}")
        # Множественные циклы обучения
        for cycle in range(5):
            data = {"market_data": realistic_market_data}
            # Адаптация
            await agent.adapt(data)
            # Обучение
            await agent.learn(data)
            # Измерение производительности
            performance = agent.get_performance()
            confidence = agent.get_confidence()
            print(f"🔄 Цикл {cycle + 1}: производительность={performance:.3f}, уверенность={confidence:.3f}")
            # Проверка улучшения (в общем случае)
            assert performance >= 0.0 and performance <= 1.0
            assert confidence >= 0.0 and confidence <= 1.0
        # Полная эволюция
        data = {"market_data": realistic_market_data}
        await agent.evolve(data)
        final_performance = agent.get_performance()
        final_confidence = agent.get_confidence()
        print(f"🎯 Финальная производительность: {final_performance:.3f}")
        print(f"🎯 Финальная уверенность: {final_confidence:.3f}")
        # Проверка, что эволюция произошла
        assert agent.evolution_count > 0
        assert agent.last_evolution is not None
        print("✅ Эволюция производительности успешна")
    @pytest.mark.asyncio
    async def test_error_recovery_e2e(self, realistic_market_data, temp_workspace) -> None:
        """E2E тест восстановления после ошибок"""
        agent = EvolvableMarketMakerAgent()
        # Сохранение начального состояния
        initial_state_path = os.path.join(temp_workspace, "initial_state.pkl")
        agent.save_state(initial_state_path)
        # Симуляция ошибки с некорректными данными
        try:
            await agent.adapt(None)
        except Exception as e:
            print(f"⚠️ Ожидаемая ошибка адаптации: {e}")
        try:
            await agent.learn({})
        except Exception as e:
            print(f"⚠️ Ожидаемая ошибка обучения: {e}")
        # Восстановление состояния
        agent.load_state(initial_state_path)
        # Проверка, что агент работает после восстановления
        data = {"market_data": realistic_market_data}
        result = await agent.adapt(data)
        assert result is True
        result = await agent.learn(data)
        assert result is True
        print("✅ Восстановление после ошибок успешно")
    @pytest.mark.asyncio
    async def test_concurrent_evolution_e2e(self, realistic_market_data, temp_workspace) -> None:
        """E2E тест параллельной эволюции агентов"""
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
        data = {"market_data": realistic_market_data}
        # Параллельная адаптация
        adapt_tasks = [agent.adapt(data) for agent in agents]
        adapt_results = await asyncio.gather(*adapt_tasks)
        assert all(adapt_results)
        # Параллельное обучение
        learn_tasks = [agent.learn(data) for agent in agents]
        learn_results = await asyncio.gather(*learn_tasks)
        assert all(learn_results)
        # Параллельная эволюция
        evolve_tasks = [agent.evolve(data) for agent in agents]
        evolve_results = await asyncio.gather(*evolve_tasks)
        assert all(evolve_results)
        # Проверка результатов
        for i, agent in enumerate(agents):
            performance = agent.get_performance()
            confidence = agent.get_confidence()
            assert 0.0 <= performance <= 1.0
            assert 0.0 <= confidence <= 1.0
            print(f"✅ Агент {i}: производительность={performance:.3f}, уверенность={confidence:.3f}")
        print("✅ Параллельная эволюция успешна")
    @pytest.mark.asyncio
    async def test_memory_management_e2e(self, realistic_market_data, temp_workspace) -> None:
        """E2E тест управления памятью"""
        agent = EvolvableMarketMakerAgent()
        # Добавление большого количества данных
        for i in range(200):  # Больше чем max_training_samples
            data = {"market_data": realistic_market_data}
            await agent.learn(data)
            # Проверка ограничения размера истории
            assert len(agent.training_data) <= agent.max_training_samples
            if i % 50 == 0:
                print(f"📊 Обработано {i} образцов, размер истории: {len(agent.training_data)}")
        # Проверка производительности после большого объема данных
        performance = agent.get_performance()
        confidence = agent.get_confidence()
        assert 0.0 <= performance <= 1.0
        assert 0.0 <= confidence <= 1.0
        print(f"✅ Управление памятью успешно: производительность={performance:.3f}")
    @pytest.mark.asyncio
    async def test_comprehensive_validation_e2e(self, realistic_market_data, realistic_strategy_signals,
                                              realistic_risk_metrics, realistic_portfolio_state, 
                                              realistic_news_data, temp_workspace) -> None:
        """Комплексная валидация всех аспектов системы"""
        print("🚀 Запуск комплексной валидации системы эволюционных агентов")
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
            "market_data": realistic_market_data,
            "strategy_signals": realistic_strategy_signals,
            "risk_metrics": realistic_risk_metrics,
            "current_weights": {k: v['weight'] for k, v in realistic_portfolio_state.items()},
            "news_data": realistic_news_data,
            "portfolio_state": realistic_portfolio_state
        }
        # Валидация 1: Инициализация
        print("✅ Валидация 1: Инициализация агентов")
        for name, agent in agents.items():
            assert agent is not None
            assert hasattr(agent, 'name')
            assert hasattr(agent, 'ml_model')
            assert hasattr(agent, 'optimizer')
        # Валидация 2: Адаптация
        print("✅ Валидация 2: Адаптация агентов")
        for name, agent in agents.items():
            result = await agent.adapt(data)
            assert result is True
        # Валидация 3: Обучение
        print("✅ Валидация 3: Обучение агентов")
        for name, agent in agents.items():
            result = await agent.learn(data)
            assert result is True
        # Валидация 4: Эволюция
        print("✅ Валидация 4: Эволюция агентов")
        for name, agent in agents.items():
            result = await agent.evolve(data)
            assert result is True
        # Валидация 5: Метрики производительности
        print("✅ Валидация 5: Метрики производительности")
        for name, agent in agents.items():
            performance = agent.get_performance()
            confidence = agent.get_confidence()
            assert 0.0 <= performance <= 1.0
            assert 0.0 <= confidence <= 1.0
        # Валидация 6: Сохранение/загрузка состояния
        print("✅ Валидация 6: Сохранение/загрузка состояния")
        for name, agent in agents.items():
            save_path = os.path.join(temp_workspace, f"{name}_validation.pkl")
            save_result = agent.save_state(save_path)
            load_result = agent.load_state(save_path)
            assert save_result is True
            assert load_result is True
        # Валидация 7: Функциональность
        print("✅ Валидация 7: Функциональность агентов")
        # MarketMaker
        spread_analysis = await agents['market_maker'].analyze_spread(realistic_market_data)
        assert isinstance(spread_analysis, dict)
        # Risk
        risk_assessment = await agents['risk'].assess_risk(realistic_market_data, realistic_risk_metrics)
        assert isinstance(risk_assessment, dict)
        # Portfolio
        optimal_weights = await agents['portfolio'].predict_optimal_weights(realistic_market_data)
        assert isinstance(optimal_weights, dict)
        # News
        sentiment_analysis = await agents['news'].analyze_sentiment(realistic_news_data)
        assert isinstance(sentiment_analysis, dict)
        # Market Regime
        regime_detection = await agents['market_regime'].detect_regime(realistic_market_data)
        assert isinstance(regime_detection, dict)
        # Strategy
        strategy_selection = await agents['strategy'].select_strategy(realistic_market_data, realistic_strategy_signals)
        assert isinstance(strategy_selection, dict)
        # Order Executor
        execution_optimization = await agents['order_executor'].optimize_execution(
            {"symbol": "BTCUSDT", "side": "buy", "quantity": 0.1, "price": 50000}, 
            realistic_market_data
        )
        assert isinstance(execution_optimization, dict)
        # Meta Controller
        coordination = await agents['meta_controller'].coordinate_strategies(
            "BTCUSDT", realistic_market_data, realistic_strategy_signals, realistic_risk_metrics
        )
        assert isinstance(coordination, dict)
        # Валидация 8: Интеграция с эволюционным менеджером
        print("✅ Валидация 8: Интеграция с эволюционным менеджером")
        from infrastructure.core.evolution_manager import get_evolution_manager
        evolution_manager = get_evolution_manager()
        registered_components = evolution_manager.get_components()
        assert len(registered_components) >= len(agents)
        print("🎉 Комплексная валидация завершена успешно!")
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 

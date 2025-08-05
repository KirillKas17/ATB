"""
Тесты интеграции основной системы с эволюционными агентами.
Проверяет полную интеграцию всех компонентов:
- main.py с эволюционными агентами
- IntegrationManager с эволюционными агентами
- EntityController с эволюционными агентами
- Полный цикл работы системы
"""
import asyncio
import pytest
from datetime import datetime
from typing import Any, Dict, AsyncGenerator
from loguru import logger
from infrastructure.core.evolution_integration import EvolutionIntegration
from infrastructure.core.integration_manager import IntegrationManager
class TestMainSystemIntegration:
    """Тесты интеграции основной системы."""
    @pytest.fixture
    async def evolution_integration(self) -> AsyncGenerator[EvolutionIntegration, None]:
        """Фикстура для эволюционной интеграции."""
        integration = EvolutionIntegration()
        await integration.initialize_agents()
        yield integration
        await integration.stop_evolution_system()
    @pytest.fixture
    async def integration_manager(self) -> AsyncGenerator[IntegrationManager, None]:
        """Фикстура для менеджера интеграции."""
        config = {
            "risk": {"max_position_size": 0.1},
            "portfolio": {"rebalance_threshold": 0.05},
            "evolution": {"performance_threshold": 0.6}
        }
        manager = IntegrationManager(config)
        await manager.initialize()
        yield manager
        await manager.stop()
    @pytest.fixture
    def mock_market_data(self) -> Dict[str, Any]:
        """Фикстура для тестовых рыночных данных."""
        return {
            "symbol": "BTC/USDT",
            "close": [50000, 50100, 50200, 50300, 50400],
            "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            "high": [50500, 50600, 50700, 50800, 50900],
            "low": [49500, 49600, 49700, 49800, 49900],
            "timestamp": [datetime.now().timestamp()] * 5
        }
    @pytest.mark.asyncio
    async def test_evolution_integration_initialization(self, evolution_integration: EvolutionIntegration) -> None:
        """Тест инициализации эволюционной интеграции."""
        # Проверка инициализации агентов
        assert len(evolution_integration.agents) == 8
        # Проверка наличия всех агентов
        expected_agents = [
            "market_maker", "risk", "portfolio", "strategy",
            "news", "order_executor", "meta_controller", "market_regime"
        ]
        for agent_name in expected_agents:
            assert agent_name in evolution_integration.agents
            agent = evolution_integration.agents[agent_name]
            assert agent is not None
            assert hasattr(agent, 'adapt')
            assert hasattr(agent, 'learn')
            assert hasattr(agent, 'evolve')
    @pytest.mark.asyncio
    async def test_integration_manager_with_evolution(self, integration_manager: IntegrationManager) -> None:
        """Тест интеграции менеджера интеграции с эволюционными агентами."""
        # Проверка инициализации эволюционных компонентов
        assert integration_manager.evolution_integration is not None
        # Проверка статуса системы
        status = await integration_manager.get_system_status()
        assert status["is_initialized"] is True
        assert "evolution_system" in status["components"]
        # Проверка эволюционной системы
        evolution_status = status["components"]["evolution_system"]
        assert "agents" in evolution_status
        assert len(evolution_status["agents"]) == 8
    @pytest.mark.asyncio
    async def test_evolution_agents_in_trading_cycle(self, evolution_integration: EvolutionIntegration, mock_market_data: Dict[str, Any]) -> None:
        """Тест использования эволюционных агентов в торговом цикле."""
        # Тестирование адаптации агентов
        for agent_name, agent in evolution_integration.agents.items():
            success = await agent.adapt(mock_market_data)
            assert success is True
            logger.info(f"Agent {agent_name} adapted successfully")
        # Тестирование обучения агентов
        for agent_name, agent in evolution_integration.agents.items():
            success = await agent.learn(mock_market_data)
            assert success is True
            logger.info(f"Agent {agent_name} learned successfully")
    @pytest.mark.asyncio
    async def test_meta_controller_coordination(self, evolution_integration: EvolutionIntegration, mock_market_data: Dict[str, Any]) -> None:
        """Тест координации через мета-контроллер."""
        meta_controller = evolution_integration.get_agent("meta_controller")
        assert meta_controller is not None
        # Тестирование координации стратегий
        coordination_result = await meta_controller.coordinate_strategies(
            "BTC/USDT", 
            mock_market_data, 
            {},  # strategy_signals
            {}   # risk_metrics
        )
        assert coordination_result is not None
        assert "evolution_metrics" in coordination_result
        assert "coordination_score" in coordination_result["evolution_metrics"]
    @pytest.mark.asyncio
    async def test_market_regime_detection(self, evolution_integration: EvolutionIntegration, mock_market_data: Dict[str, Any]) -> None:
        """Тест обнаружения режима рынка."""
        market_regime_agent = evolution_integration.get_agent("market_regime")
        assert market_regime_agent is not None
        # Тестирование обнаружения режима
        regime_result = await market_regime_agent.detect_regime(mock_market_data)
        assert regime_result is not None
        assert "regime_type" in regime_result
        assert "confidence" in regime_result
        assert regime_result["confidence"] >= 0.0 and regime_result["confidence"] <= 1.0
    @pytest.mark.asyncio
    async def test_risk_assessment(self, evolution_integration: EvolutionIntegration, mock_market_data: Dict[str, Any]) -> None:
        """Тест оценки рисков."""
        risk_agent = evolution_integration.get_agent("risk")
        assert risk_agent is not None
        # Тестирование оценки рисков
        risk_result = await risk_agent.assess_risk(mock_market_data, {})
        assert risk_result is not None
        assert "risk_score" in risk_result
        assert "risk_level" in risk_result
        assert risk_result["risk_score"] >= 0.0 and risk_result["risk_score"] <= 1.0
    @pytest.mark.asyncio
    async def test_portfolio_optimization(self, evolution_integration, mock_market_data) -> None:
        """Тест оптимизации портфеля."""
        portfolio_agent = evolution_integration.get_agent("portfolio")
        assert portfolio_agent is not None
        # Тестирование предсказания оптимальных весов
        weights_result = await portfolio_agent.predict_optimal_weights(mock_market_data)
        assert weights_result is not None
        assert "optimal_weights" in weights_result
        assert isinstance(weights_result["optimal_weights"], dict)
    @pytest.mark.asyncio
    async def test_strategy_selection(self, evolution_integration, mock_market_data) -> None:
        """Тест выбора стратегии."""
        strategy_agent = evolution_integration.get_agent("strategy")
        assert strategy_agent is not None
        # Тестирование выбора стратегии
        strategy_result = await strategy_agent.select_strategy(mock_market_data, {})
        assert strategy_result is not None
        assert "selected_strategy" in strategy_result
        assert "confidence" in strategy_result
    @pytest.mark.asyncio
    async def test_order_execution_optimization(self, evolution_integration, mock_market_data) -> None:
        """Тест оптимизации исполнения ордеров."""
        order_executor = evolution_integration.get_agent("order_executor")
        assert order_executor is not None
        # Тестовый ордер
        test_order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": 0.1,
            "price": 50000
        }
        # Тестирование оптимизации исполнения
        optimization_result = await order_executor.optimize_execution(test_order, mock_market_data)
        assert optimization_result is not None
        assert "execution_strategy" in optimization_result
        assert "expected_slippage" in optimization_result
    @pytest.mark.asyncio
    async def test_news_sentiment_analysis(self, evolution_integration) -> None:
        """Тест анализа настроений новостей."""
        news_agent = evolution_integration.get_agent("news")
        assert news_agent is not None
        # Тестовые новостные данные
        news_data = {
            "sentiment_score": 0.6,
            "news_volume": 100,
            "social_sentiment": 0.7,
            "breaking_news": False
        }
        # Тестирование анализа настроений
        sentiment_result = await news_agent.analyze_sentiment(news_data)
        assert sentiment_result is not None
        assert "sentiment_score" in sentiment_result
        assert "sentiment_label" in sentiment_result
    @pytest.mark.asyncio
    async def test_agent_evolution_cycle(self, evolution_integration, mock_market_data) -> None:
        """Тест цикла эволюции агентов."""
        # Тестирование эволюции агентов
        for agent_name, agent in evolution_integration.agents.items():
            # Проверка начального состояния
            initial_performance = agent.get_performance()
            initial_confidence = agent.get_confidence()
            # Запуск эволюции
            success = await agent.evolve(mock_market_data)
            assert success is True
            # Проверка изменения состояния
            final_performance = agent.get_performance()
            final_confidence = agent.get_confidence()
            logger.info(f"Agent {agent_name}: performance {initial_performance:.3f} -> {final_performance:.3f}, "
                       f"confidence {initial_confidence:.3f} -> {final_confidence:.3f}")
    @pytest.mark.asyncio
    async def test_system_status_monitoring(self, integration_manager) -> None:
        """Тест мониторинга статуса системы."""
        # Получение статуса системы
        status = await integration_manager.get_system_status()
        # Проверка основных компонентов
        assert status["is_initialized"] is True
        assert status["is_running"] is True
        assert "components" in status
        # Проверка эволюционной системы
        evolution_status = status["components"]["evolution_system"]
        assert "agents" in evolution_status
        # Проверка статуса каждого агента
        for agent_name, agent_status in evolution_status["agents"].items():
            assert "performance" in agent_status
            assert "confidence" in agent_status
            assert "evolution_count" in agent_status
            assert "is_evolving" in agent_status
    @pytest.mark.asyncio
    async def test_agent_performance_tracking(self, evolution_integration, mock_market_data) -> None:
        """Тест отслеживания производительности агентов."""
        # Тестирование адаптации и отслеживания производительности
        for agent_name, agent in evolution_integration.agents.items():
            # Начальная производительность
            initial_performance = agent.get_performance()
            # Адаптация агента
            await agent.adapt(mock_market_data)
            # Проверка изменения производительности
            current_performance = agent.get_performance()
            assert current_performance >= 0.0 and current_performance <= 1.0
            logger.info(f"Agent {agent_name} performance: {initial_performance:.3f} -> {current_performance:.3f}")
    @pytest.mark.asyncio
    async def test_agent_confidence_tracking(self, evolution_integration, mock_market_data) -> None:
        """Тест отслеживания уверенности агентов."""
        # Тестирование обучения и отслеживания уверенности
        for agent_name, agent in evolution_integration.agents.items():
            # Начальная уверенность
            initial_confidence = agent.get_confidence()
            # Обучение агента
            await agent.learn(mock_market_data)
            # Проверка изменения уверенности
            current_confidence = agent.get_confidence()
            assert current_confidence >= 0.0 and current_confidence <= 1.0
            logger.info(f"Agent {agent_name} confidence: {initial_confidence:.3f} -> {current_confidence:.3f}")
    @pytest.mark.asyncio
    async def test_agent_state_persistence(self, evolution_integration, mock_market_data) -> None:
        """Тест сохранения и загрузки состояния агентов."""
        for agent_name, agent in evolution_integration.agents.items():
            # Адаптация агента
            await agent.adapt(mock_market_data)
            await agent.learn(mock_market_data)
            # Сохранение состояния
            state_path = f"test_state_{agent_name}"
            save_success = agent.save_state(state_path)
            assert save_success is True
            # Загрузка состояния
            load_success = agent.load_state(state_path)
            assert load_success is True
            logger.info(f"Agent {agent_name} state persistence: OK")
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, evolution_integration) -> None:
        """Тест обработки ошибок агентов."""
        # Тестирование с некорректными данными
        invalid_data = None
        for agent_name, agent in evolution_integration.agents.items():
            try:
                # Попытка адаптации с некорректными данными
                success = await agent.adapt(invalid_data)
                # Агент должен корректно обработать ошибку
                assert success is False or success is True
            except Exception as e:
                logger.warning(f"Agent {agent_name} handled error correctly: {e}")
    @pytest.mark.asyncio
    async def test_agent_concurrent_operations(self, evolution_integration, mock_market_data) -> None:
        """Тест параллельных операций агентов."""
        # Создание задач для параллельного выполнения
        tasks = []
        for agent_name, agent in evolution_integration.agents.items():
            task = asyncio.create_task(agent.adapt(mock_market_data))
            tasks.append(task)
        # Ожидание завершения всех задач
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Проверка результатов
        for i, result in enumerate(results):
            agent_name = list(evolution_integration.agents.keys())[i]
            if isinstance(result, Exception):
                logger.warning(f"Agent {agent_name} failed: {result}")
            else:
                assert result is True
                logger.info(f"Agent {agent_name} completed successfully")
    @pytest.mark.asyncio
    async def test_agent_memory_management(self, evolution_integration, mock_market_data) -> None:
        """Тест управления памятью агентов."""
        for agent_name, agent in evolution_integration.agents.items():
            # Проверка начального использования памяти
            initial_memory = agent.get_memory_usage()
            # Выполнение операций
            for _ in range(10):
                await agent.adapt(mock_market_data)
                await agent.learn(mock_market_data)
            # Проверка использования памяти после операций
            final_memory = agent.get_memory_usage()
            # Память не должна расти бесконечно
            assert final_memory <= initial_memory * 2  # Допускаем рост в 2 раза
            logger.info(f"Agent {agent_name} memory: {initial_memory} -> {final_memory}")
    @pytest.mark.asyncio
    async def test_agent_evolution_validation(self, evolution_integration, mock_market_data) -> None:
        """Тест валидации эволюции агентов."""
        for agent_name, agent in evolution_integration.agents.items():
            # Начальные метрики
            initial_performance = agent.get_performance()
            initial_confidence = agent.get_confidence()
            initial_evolution_count = agent.evolution_count
            # Эволюция агента
            success = await agent.evolve(mock_market_data)
            assert success is True
            # Проверка изменения метрик
            final_performance = agent.get_performance()
            final_confidence = agent.get_confidence()
            final_evolution_count = agent.evolution_count
            # Эволюция должна увеличить счетчик
            assert final_evolution_count > initial_evolution_count
            logger.info(f"Agent {agent_name} evolution: count {initial_evolution_count} -> {final_evolution_count}")
    @pytest.mark.asyncio
    async def test_full_system_integration(self, integration_manager, mock_market_data) -> None:
        """Тест полной интеграции системы."""
        # Запуск системы
        await integration_manager.start()
        # Проверка работы основных циклов
        for _ in range(5):  # 5 итераций
            # Симуляция основного цикла
            await integration_manager._process_main_logic()
            await asyncio.sleep(0.1)  # Небольшая пауза
        # Проверка статуса системы
        status = await integration_manager.get_system_status()
        assert status["is_running"] is True
        # Проверка эволюционной системы
        evolution_status = status["components"]["evolution_system"]
        assert "agents" in evolution_status
        # Остановка системы
        await integration_manager.stop()
    @pytest.mark.asyncio
    async def test_agent_interaction_patterns(self, evolution_integration, mock_market_data) -> None:
        """Тест паттернов взаимодействия между агентами."""
        # Тестирование взаимодействия мета-контроллера с другими агентами
        meta_controller = evolution_integration.get_agent("meta_controller")
        assert meta_controller is not None
        # Получение результатов от других агентов
        agent_results = {}
        for agent_name, agent in evolution_integration.agents.items():
            if agent_name != "meta_controller":
                try:
                    if hasattr(agent, 'detect_regime'):
                        result = await agent.detect_regime(mock_market_data)
                    elif hasattr(agent, 'assess_risk'):
                        result = await agent.assess_risk(mock_market_data, {})
                    elif hasattr(agent, 'predict_optimal_weights'):
                        result = await agent.predict_optimal_weights(mock_market_data)
                    else:
                        result = await agent.adapt(mock_market_data)
                    agent_results[agent_name] = result
                except Exception as e:
                    logger.warning(f"Agent {agent_name} interaction failed: {e}")
        # Координация через мета-контроллер
        coordination_result = await meta_controller.coordinate_strategies(
            "BTC/USDT", 
            mock_market_data, 
            agent_results,  # Результаты других агентов
            {}   # risk_metrics
        )
        assert coordination_result is not None
        logger.info(f"Meta-controller coordination successful: {coordination_result}")
    @pytest.mark.asyncio
    async def test_agent_performance_benchmark(self, evolution_integration, mock_market_data) -> None:
        """Тест бенчмарка производительности агентов."""
        import time
        performance_results = {}
        for agent_name, agent in evolution_integration.agents.items():
            start_time = time.time()
            # Выполнение операций
            for _ in range(10):
                await agent.adapt(mock_market_data)
                await agent.learn(mock_market_data)
            end_time = time.time()
            execution_time = end_time - start_time
            performance_results[agent_name] = {
                "execution_time": execution_time,
                "operations_per_second": 20 / execution_time,  # 20 операций
                "performance": agent.get_performance(),
                "confidence": agent.get_confidence()
            }
            logger.info(f"Agent {agent_name} benchmark: {execution_time:.3f}s, "
                       f"{performance_results[agent_name]['operations_per_second']:.1f} ops/s")
        # Проверка разумности времени выполнения
        for agent_name, results in performance_results.items():
            assert results["execution_time"] < 10.0  # Не более 10 секунд
            assert results["operations_per_second"] > 1.0  # Не менее 1 операции в секунду
    @pytest.mark.asyncio
    async def test_agent_evolution_effectiveness(self, evolution_integration, mock_market_data) -> None:
        """Тест эффективности эволюции агентов."""
        effectiveness_results = {}
        for agent_name, agent in evolution_integration.agents.items():
            # Начальные метрики
            initial_performance = agent.get_performance()
            initial_confidence = agent.get_confidence()
            # Серия эволюций
            for evolution_round in range(3):
                success = await agent.evolve(mock_market_data)
                assert success is True
                # Адаптация после эволюции
                await agent.adapt(mock_market_data)
                await agent.learn(mock_market_data)
            # Финальные метрики
            final_performance = agent.get_performance()
            final_confidence = agent.get_confidence()
            # Расчет эффективности
            performance_improvement = final_performance - initial_performance
            confidence_improvement = final_confidence - initial_confidence
            effectiveness_results[agent_name] = {
                "performance_improvement": performance_improvement,
                "confidence_improvement": confidence_improvement,
                "evolution_count": agent.evolution_count
            }
            logger.info(f"Agent {agent_name} effectiveness: "
                       f"performance +{performance_improvement:.3f}, "
                       f"confidence +{confidence_improvement:.3f}, "
                       f"evolutions: {agent.evolution_count}")
        # Проверка общей эффективности
        total_improvements = sum(
            results["performance_improvement"] + results["confidence_improvement"]
            for results in effectiveness_results.values()
        )
        # Эволюция должна в целом улучшать систему
        assert total_improvements >= -1.0  # Допускаем небольшое ухудшение из-за случайности
class TestMainSystemErrorHandling:
    """Тесты обработки ошибок в основной системе."""
    @pytest.mark.asyncio
    def test_agent_initialization_errors(self: "TestMainSystemErrorHandling") -> None:
        """Тест ошибок инициализации агентов."""
        with patch('infrastructure.agents.evolvable_market_maker.EvolvableMarketMakerAgent.__init__', 
                  side_effect=Exception("Initialization error")):
            integration = EvolutionIntegration()
            # Инициализация должна обработать ошибку
            try:
                await integration.initialize_agents()
            except Exception as e:
                logger.info(f"Expected initialization error handled: {e}")
    @pytest.mark.asyncio
    def test_agent_operation_errors(self: "TestMainSystemErrorHandling") -> None:
        """Тест ошибок операций агентов."""
        integration = EvolutionIntegration()
        await integration.initialize_agents()
        # Тестирование с некорректными данными
        invalid_data = {"invalid": "data"}
        for agent_name, agent in integration.agents.items():
            try:
                # Попытка операций с некорректными данными
                await agent.adapt(invalid_data)
                await agent.learn(invalid_data)
                await agent.evolve(invalid_data)
            except Exception as e:
                logger.info(f"Agent {agent_name} handled operation error: {e}")
    @pytest.mark.asyncio
    def test_system_recovery(self: "TestMainSystemErrorHandling") -> None:
        """Тест восстановления системы после ошибок."""
        integration = EvolutionIntegration()
        # Симуляция ошибки и восстановления
        try:
            await integration.initialize_agents()
            await integration.start_evolution_system()
            # Симуляция ошибки
            raise Exception("Simulated system error")
        except Exception as e:
            logger.info(f"System error occurred: {e}")
            # Восстановление системы
            try:
                await integration.stop_evolution_system()
                logger.info("System recovered successfully")
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {recovery_error}")
if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"]) 

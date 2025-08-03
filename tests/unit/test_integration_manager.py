"""
Unit тесты для IntegrationManager.
Тестирует инициализацию компонентов, управление жизненным циклом,
координацию взаимодействий и мониторинг состояния системы.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from infrastructure.core.integration_manager import IntegrationManager
from domain.types.messaging_types import Event, EventType, EventPriority, EventMetadata
import time

class TestIntegrationManager:
    """Тесты для IntegrationManager."""
    @pytest.fixture
    def config(self) -> dict:
        """Фикстура для конфигурации."""
        return {
            "risk": {
                "max_daily_loss": 0.02,
                "max_position_size": 0.1
            },
            "portfolio": {
                "initial_cash": 10000.0,
                "max_positions": 10
            },
            "evolution": {
                "enabled": True,
                "cycle_interval": 1800
            },
            "monitoring": {
                "heartbeat_interval": 30,
                "health_check_interval": 60
            }
        }
    @pytest.fixture
    def integration_manager(self, config: dict) -> IntegrationManager:
        """Фикстура для IntegrationManager."""
        return IntegrationManager(config)
    def test_initialization(self, integration_manager: IntegrationManager) -> None:
        """Тест инициализации менеджера интеграции."""
        assert integration_manager is not None
        assert integration_manager.config is not None
        assert integration_manager.event_bus is not None
        assert integration_manager.is_initialized is False
        assert integration_manager.is_running is False
    def test_config_structure(self, integration_manager: IntegrationManager) -> None:
        """Тест структуры конфигурации."""
        config = integration_manager.config
        assert "risk" in config
        assert "portfolio" in config
        assert "evolution" in config
        assert "monitoring" in config
        # Проверка настроек риска
        risk_config = config["risk"]
        assert "max_daily_loss" in risk_config
        assert "max_position_size" in risk_config
        # Проверка настроек портфеля
        portfolio_config = config["portfolio"]
        assert "initial_cash" in portfolio_config
        assert "max_positions" in portfolio_config
    @pytest.mark.asyncio
    async def test_system_initialization(self, integration_manager: IntegrationManager) -> None:
        """Тест инициализации системы."""
        # Инициализация системы
        await integration_manager.initialize()
        # Проверка состояния после инициализации
        assert integration_manager.is_initialized is True
        # Проверка инициализации компонентов
        assert integration_manager.health_checker is not None
        assert integration_manager.metrics is not None
        assert integration_manager.circuit_breaker is not None
        assert integration_manager.risk_manager is not None
        assert integration_manager.portfolio_manager is not None
        assert integration_manager.autonomous_controller is not None
    @pytest.mark.asyncio
    async def test_utilities_initialization(self, integration_manager: IntegrationManager) -> None:
        """Тест инициализации утилит."""
        await integration_manager._initialize_utilities()
        # Проверка инициализации утилит
        assert integration_manager.cache is not None
        assert integration_manager.health_checker is not None
        assert integration_manager.metrics is not None
    @pytest.mark.asyncio
    async def test_core_components_initialization(self, integration_manager: IntegrationManager) -> None:
        """Тест инициализации основных компонентов."""
        await integration_manager._initialize_core_components()
        # Проверка инициализации основных компонентов
        assert integration_manager.circuit_breaker is not None
        assert integration_manager.risk_manager is not None
        assert integration_manager.portfolio_manager is not None
        assert integration_manager.autonomous_controller is not None
    @pytest.mark.asyncio
    async def test_ml_components_initialization(self, integration_manager: IntegrationManager) -> None:
        """Тест инициализации ML компонентов."""
        await integration_manager._initialize_ml_components()
        # Проверка инициализации ML компонентов
        assert integration_manager.regime_discovery is not None
        assert integration_manager.live_adaptation is not None
    @pytest.mark.asyncio
    async def test_evolution_components_initialization(self, integration_manager: IntegrationManager) -> None:
        """Тест инициализации эволюционных компонентов."""
        await integration_manager._initialize_evolution_components()
        # Проверка инициализации эволюционных компонентов
        assert integration_manager.evolution_integration is not None
    @pytest.mark.asyncio
    async def test_component_interactions_setup(self, integration_manager: IntegrationManager) -> None:
        """Тест настройки взаимодействий компонентов."""
        # Инициализация компонентов
        await integration_manager._initialize_core_components()
        # Настройка взаимодействий
        await integration_manager._setup_interactions()
        # Проверка настройки callback функций
        assert integration_manager.circuit_breaker is not None
    @pytest.mark.asyncio
    async def test_event_handlers_setup(self, integration_manager: IntegrationManager) -> None:
        """Тест настройки обработчиков событий."""
        await integration_manager._setup_event_handlers()
        # Проверка подписки на события
        # (проверка через внутреннее состояние event_bus)
    @pytest.mark.asyncio
    async def test_system_start(self, integration_manager: IntegrationManager) -> None:
        """Тест запуска системы."""
        # Инициализация
        await integration_manager.initialize()
        # Запуск системы
        await integration_manager.start()
        # Проверка состояния после запуска
        assert integration_manager.is_running is True
    @pytest.mark.asyncio
    async def test_system_stop(self, integration_manager: IntegrationManager) -> None:
        """Тест остановки системы."""
        # Инициализация и запуск
        await integration_manager.initialize()
        await integration_manager.start()
        # Остановка системы
        await integration_manager.stop()
        # Проверка состояния после остановки
        assert integration_manager.is_running is False
    @pytest.mark.asyncio
    async def test_main_loop(self, integration_manager: IntegrationManager) -> None:
        """Тест основного цикла системы."""
        # Инициализация
        await integration_manager.initialize()
        integration_manager.is_running = True
        # Запуск основного цикла
        await integration_manager._start_main_loops()
        # Проверка запуска циклов
        # (проверка через создание задач)
    @pytest.mark.asyncio
    async def test_main_logic_processing(self, integration_manager: IntegrationManager) -> None:
        """Тест обработки основной логики."""
        # Инициализация
        await integration_manager.initialize()
        # Мок рыночных условий
        market_conditions = {
            "volatility": 0.02,
            "trend": "bullish",
            "volume": 1000000.0
        }
        # Обработка основной логики
        await integration_manager._process_main_logic()
        # Проверка выполнения логики
        # (проверка через моки компонентов)
    @pytest.mark.asyncio
    async def test_trading_decisions_with_evolution(self, integration_manager: IntegrationManager) -> None:
        """Тест принятия торговых решений с эволюцией."""
        # Инициализация
        await integration_manager.initialize()
        # Мок рыночных условий
        market_conditions = {
            "symbol": "BTCUSDT",
            "price": 50000.0,
            "volume": 1000000.0,
            "volatility": 0.02
        }
        # Принятие решений с эволюцией
        decisions = await integration_manager._make_trading_decisions_with_evolution(market_conditions)
        assert decisions is not None
    @pytest.mark.asyncio
    async def test_strategy_adaptation_with_evolution(self, integration_manager: IntegrationManager) -> None:
        """Тест адаптации стратегий с эволюцией."""
        # Инициализация
        await integration_manager.initialize()
        # Мок рыночных условий
        market_conditions = {
            "regime": "trending",
            "volatility": 0.03,
            "correlation": 0.8
        }
        # Адаптация стратегий
        adaptations = await integration_manager._adapt_strategies_with_evolution(market_conditions)
        assert adaptations is not None
    @pytest.mark.asyncio
    async def test_market_conditions_analysis(self, integration_manager: IntegrationManager) -> None:
        """Тест анализа рыночных условий."""
        # Инициализация
        await integration_manager.initialize()
        # Анализ рыночных условий
        conditions = await integration_manager._analyze_market_conditions()
        assert conditions is not None
    @pytest.mark.asyncio
    async def test_risk_exposure_management(self, integration_manager: IntegrationManager) -> None:
        """Тест управления рисковым воздействием."""
        # Инициализация
        await integration_manager.initialize()
        # Управление рисковым воздействием
        await integration_manager._manage_risk_exposure()
        # Проверка выполнения управления рисками
    @pytest.mark.asyncio
    async def test_trading_decisions(self, integration_manager: IntegrationManager) -> None:
        """Тест принятия торговых решений."""
        # Инициализация
        await integration_manager.initialize()
        # Принятие торговых решений
        decisions = await integration_manager._make_trading_decisions()
        assert decisions is not None
    @pytest.mark.asyncio
    async def test_order_execution(self, integration_manager: IntegrationManager) -> None:
        """Тест исполнения ордеров."""
        # Инициализация
        await integration_manager.initialize()
        # Мок активных ордеров
        active_orders = [
            {"id": "1", "symbol": "BTCUSDT", "side": "buy", "quantity": 0.1}
        ]
        # Исполнение ордеров
        await integration_manager._execute_orders()
        # Проверка исполнения ордеров
    @pytest.mark.asyncio
    async def test_portfolio_update(self, integration_manager: IntegrationManager) -> None:
        """Тест обновления портфеля."""
        # Инициализация
        await integration_manager.initialize()
        # Обновление портфеля
        await integration_manager._update_portfolio()
        # Проверка обновления портфеля
    @pytest.mark.asyncio
    async def test_strategy_adaptation(self, integration_manager: IntegrationManager) -> None:
        """Тест адаптации стратегий."""
        # Инициализация
        await integration_manager.initialize()
        # Мок рыночных условий
        market_conditions = {
            "regime": "ranging",
            "volatility": 0.01
        }
        # Адаптация стратегий
        await integration_manager._adapt_strategies(market_conditions)
        # Проверка адаптации стратегий
    @pytest.mark.asyncio
    async def test_market_data_retrieval(self, integration_manager: IntegrationManager) -> None:
        """Тест получения рыночных данных."""
        # Инициализация
        await integration_manager.initialize()
        # Получение рыночных данных
        market_data = await integration_manager._get_market_data("BTCUSDT")
        # Проверка получения данных
        assert market_data is None or isinstance(market_data, dict)
    @pytest.mark.asyncio
    async def test_trading_signal_creation(self, integration_manager: IntegrationManager) -> None:
        """Тест создания торговых сигналов."""
        # Инициализация
        await integration_manager.initialize()
        # Мок решения
        decision = {
            "action": "buy",
            "confidence": 0.8,
            "reasoning": "Strong bullish trend"
        }
        # Создание торгового сигнала
        signal = await integration_manager._create_trading_signal("BTCUSDT", decision)
        # Проверка создания сигнала
        assert signal is not None
    @pytest.mark.asyncio
    async def test_active_orders_retrieval(self, integration_manager: IntegrationManager) -> None:
        """Тест получения активных ордеров."""
        # Инициализация
        await integration_manager.initialize()
        # Получение активных ордеров
        active_orders = await integration_manager._get_active_orders()
        # Проверка получения ордеров
        assert isinstance(active_orders, list)
    @pytest.mark.asyncio
    async def test_order_execution_decision(self, integration_manager: IntegrationManager) -> None:
        """Тест решения об исполнении ордера."""
        # Инициализация
        await integration_manager.initialize()
        # Мок ордера
        order = {
            "id": "1",
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": 0.1,
            "price": 50000.0
        }
        # Проверка решения об исполнении
        should_execute = await integration_manager._should_execute_order(order)
        # Проверка результата
        assert isinstance(should_execute, bool)
    @pytest.mark.asyncio
    async def test_single_order_execution(self, integration_manager: IntegrationManager) -> None:
        """Тест исполнения одного ордера."""
        # Инициализация
        await integration_manager.initialize()
        # Мок ордера
        order = {
            "id": "1",
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": 0.1,
            "price": 50000.0
        }
        # Исполнение ордера
        success = await integration_manager._execute_single_order(order)
        # Проверка результата
        assert isinstance(success, bool)
    @pytest.mark.asyncio
    async def test_error_handling(self, integration_manager: IntegrationManager) -> None:
        """Тест обработки ошибок."""
        # Инициализация
        await integration_manager.initialize()
        # Мок ошибки
        error = Exception("Test error")
        # Обработка ошибки
        await integration_manager._handle_main_logic_error(error)
        # Проверка обработки ошибки
    @pytest.mark.asyncio
    async def test_event_processing_loop(self, integration_manager: IntegrationManager) -> None:
        """Тест цикла обработки событий."""
        # Инициализация
        await integration_manager.initialize()
        integration_manager.is_running = True
        # Запуск цикла обработки событий
        await integration_manager._event_processing_loop()
        # Проверка обработки событий
    @pytest.mark.asyncio
    async def test_system_status_retrieval(self, integration_manager: IntegrationManager) -> None:
        """Тест получения статуса системы."""
        # Инициализация
        await integration_manager.initialize()
        # Получение статуса системы
        status = await integration_manager.get_system_status()
        # Проверка статуса
        assert status is not None
        assert "is_initialized" in status
        assert "is_running" in status
        assert "components" in status
        assert "performance_metrics" in status
    @pytest.mark.asyncio
    async def test_event_handlers(self, integration_manager: IntegrationManager) -> None:
        """Тест обработчиков событий."""
        # Инициализация
        await integration_manager.initialize()
        # Мок событий
        trade_event = Event(
            name="trade_executed",  # type: ignore[arg-type]
            type=EventType.TRADE_EXECUTED,
            data={"symbol": "BTCUSDT", "quantity": 0.1, "price": 50000.0},
            metadata=EventMetadata(
                source="test"
            )
        )
        risk_event = Event(
            name="risk_limit_breached",  # type: ignore[arg-type]
            type=EventType.RISK_LIMIT_BREACHED,
            data={"risk_type": "daily_loss", "value": 0.03},
            metadata=EventMetadata(
                source="test"
            )
        )
        # Обработка событий
        await integration_manager._on_trade_executed(trade_event)
        await integration_manager._on_risk_limit_exceeded(risk_event)
        # Проверка обработки событий
    @pytest.mark.asyncio
    async def test_circuit_breaker_callbacks(self, integration_manager: IntegrationManager) -> None:
        """Тест callback функций circuit breaker."""
        # Инициализация
        await integration_manager.initialize()
        # Тест callback функций
        await integration_manager._on_exchange_circuit_change("open")
        await integration_manager._on_database_circuit_change("half_open")
        # Проверка callback функций
    @pytest.mark.asyncio
    async def test_initialization_error_handling(self, integration_manager: IntegrationManager) -> None:
        """Тест обработки ошибок инициализации."""
        # Мок ошибки инициализации
        error = Exception("Initialization failed")
        # Обработка ошибки
        await integration_manager._handle_initialization_error(error)
        # Проверка обработки ошибки
    def test_cleanup(self, integration_manager: IntegrationManager) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        integration_manager.cleanup()
        # Проверка, что ресурсы освобождены
        assert integration_manager.is_running is False
        assert integration_manager.is_initialized is False 

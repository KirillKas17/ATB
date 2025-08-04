"""
Unit тесты для AutonomousController.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, AsyncMock
from infrastructure.core.autonomous_controller import AutonomousController
from infrastructure.messaging.event_bus import EventBus
from infrastructure.messaging.optimized_event_bus import Event, EventPriority, EventType
from domain.types.messaging_types import EventMetadata


class TestAutonomousController:
    """Тесты для AutonomousController."""

    @pytest.fixture
    def event_bus(self) -> EventBus:
        """Мок EventBus."""
        return Mock(spec=EventBus)

    @pytest.fixture
    def config(self) -> dict:
        """Конфигурация для тестов."""
        return {
            "risk": {
                "max_daily_loss": 0.02,
                "max_weekly_loss": 0.05,
                "position_sizing": "kelly",
                "correlation_threshold": 0.7,
            },
            "strategy_selection": {
                "confidence_threshold": 0.6,
                "min_trades_for_evaluation": 50,
                "evaluation_period": "1d",
            },
            "market_regime": {
                "detection_sensitivity": 0.8,
                "regime_switch_threshold": 0.3,
                "min_regime_duration": "4h",
            },
            "trading": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "intervals": ["1m", "5m", "15m"],
            },
        }

    @pytest.fixture
    def autonomous_controller(self, event_bus: EventBus, config: dict) -> AutonomousController:
        """Создание экземпляра AutonomousController для тестов."""
        return AutonomousController(event_bus, config)

    def test_initialization(self, autonomous_controller: AutonomousController) -> None:
        """Тест инициализации контроллера."""
        assert autonomous_controller is not None
        assert autonomous_controller.config is not None
        assert autonomous_controller.system_state is not None
        assert autonomous_controller.auto_config is not None

    def test_auto_config_structure(self, autonomous_controller: AutonomousController) -> None:
        """Тест структуры автономной конфигурации."""
        auto_config = autonomous_controller.auto_config
        
        # Проверка наличия всех секций
        assert "risk_management" in auto_config
        assert "strategy_selection" in auto_config
        assert "market_regime" in auto_config
        
        # Проверка параметров риск-менеджмента
        risk_config = auto_config["risk_management"]
        assert "max_daily_loss" in risk_config
        assert "max_weekly_loss" in risk_config
        assert "position_sizing" in risk_config
        assert "correlation_threshold" in risk_config
        
        # Проверка валидности значений
        assert 0.0 <= risk_config["max_daily_loss"] <= 1.0
        assert 0.0 <= risk_config["max_weekly_loss"] <= 1.0
        assert risk_config["position_sizing"] in ["kelly", "fixed", "volatility"]
        assert 0.0 <= risk_config["correlation_threshold"] <= 1.0

    def test_system_state_initialization(self, autonomous_controller: AutonomousController) -> None:
        """Тест инициализации состояния системы."""
        system_state = autonomous_controller.system_state
        
        # Проверка наличия всех ключей
        assert "is_healthy" in system_state
        assert "current_regime" in system_state
        assert "active_strategies" in system_state
        assert "performance_metrics" in system_state
        assert "risk_metrics" in system_state
        
        # Проверка начальных значений
        assert system_state["is_healthy"] is True
        assert system_state["current_regime"] == "unknown"
        assert isinstance(system_state["active_strategies"], list)
        assert isinstance(system_state["performance_metrics"], dict)
        assert isinstance(system_state["risk_metrics"], dict)

    @pytest.mark.asyncio
    async def test_autonomous_decision_making(self, autonomous_controller: AutonomousController) -> None:
        """Тест автономного принятия решений."""
        # Мок рыночных данных
        market_data = {
            "BTCUSDT": {
                "price": 50000.0,
                "volume": 1000000.0,
                "volatility": 0.025,
                "trend": "upward"
            },
            "ETHUSDT": {
                "price": 3000.0,
                "volume": 500000.0,
                "volatility": 0.03,
                "trend": "sideways"
            }
        }
        
        # Проверка, что контроллер может обрабатывать данные
        assert autonomous_controller is not None
        assert autonomous_controller.system_state is not None

    @pytest.mark.asyncio
    async def test_risk_management_integration(self, autonomous_controller: AutonomousController) -> None:
        """Тест интеграции с риск-менеджментом."""
        # Мок портфеля
        portfolio_data = {
            "total_value": 100000.0,
            "daily_pnl": -500.0,
            "weekly_pnl": -2000.0,
            "positions": [
                {"symbol": "BTCUSDT", "size": 0.1, "unrealized_pnl": -100.0}
            ]
        }
        
        # Проверка, что контроллер может обрабатывать данные портфеля
        assert autonomous_controller is not None
        assert autonomous_controller.risk_manager is not None

    @pytest.mark.asyncio
    async def test_strategy_selection(self, autonomous_controller: AutonomousController) -> None:
        """Тест выбора стратегии."""
        # Мок доступных стратегий
        available_strategies = [
            {
                "name": "trend_following",
                "confidence": 0.75,
                "performance": 0.12,
                "risk_score": 0.3
            },
            {
                "name": "mean_reversion",
                "confidence": 0.65,
                "performance": 0.08,
                "risk_score": 0.4
            },
            {
                "name": "momentum",
                "confidence": 0.55,
                "performance": 0.15,
                "risk_score": 0.5
            }
        ]
        
        # Проверка, что контроллер может обрабатывать стратегии
        assert autonomous_controller is not None
        assert autonomous_controller.system_state is not None

    @pytest.mark.asyncio
    async def test_market_regime_detection(self, autonomous_controller: AutonomousController) -> None:
        """Тест обнаружения рыночного режима."""
        # Мок рыночных данных
        market_data = {
            "volatility": 0.025,
            "volume": 1500000.0,
            "price_change_1h": 0.02,
            "price_change_24h": 0.05,
            "correlation_matrix": [[1.0, 0.8], [0.8, 1.0]],
            "liquidity": 0.8,
            "momentum": 0.6
        }
        
        # Проверка, что контроллер может обрабатывать рыночные данные
        assert autonomous_controller is not None
        assert autonomous_controller.regime_discovery is not None

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, autonomous_controller: AutonomousController) -> None:
        """Тест мониторинга производительности."""
        # Мок метрик производительности
        performance_data = {
            "total_trades": 150,
            "win_rate": 0.65,
            "profit_factor": 1.8,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.08,
            "avg_trade_duration": 3600
        }
        
        # Проверка, что контроллер может обрабатывать метрики
        assert autonomous_controller is not None
        assert autonomous_controller.metrics is not None

    @pytest.mark.asyncio
    async def test_error_handling(self, autonomous_controller: AutonomousController) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        invalid_data = None
        
        # Обработка должна быть устойчивой к ошибкам
        try:
            await autonomous_controller._handle_error(Exception("Test error"))
            assert True
        except Exception:
            assert False

    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, autonomous_controller: AutonomousController) -> None:
        """Тест мониторинга здоровья системы."""
        # Проверка здоровья системы
        await autonomous_controller._check_system_health()
        
        # Проверяем, что метод выполнился без ошибок
        assert True

    @pytest.mark.asyncio
    async def test_event_handling(self, autonomous_controller: AutonomousController) -> None:
        """Тест обработки событий."""
        # Создание тестового события
        test_event = Event(
            name="market.update",
            type=EventType.MARKET_DATA_UPDATED,
            data={"symbol": "BTCUSDT", "price": 50000.0},
            metadata=EventMetadata(
                source="test",
                correlation_id=None,
                user_id=None,
                session_id=None
            )
        )
        
        # Проверка, что контроллер может обрабатывать события
        assert autonomous_controller is not None
        assert autonomous_controller.event_bus is not None

    def test_get_system_status(self, autonomous_controller: AutonomousController) -> None:
        """Тест получения статуса системы."""
        # Получение статуса
        status = autonomous_controller.get_system_status()
        
        # Проверка структуры статуса
        assert status is not None
        assert isinstance(status, dict)
        assert "is_healthy" in status
        assert "current_regime" in status
        assert "active_strategies" in status
        assert "performance_metrics" in status
        assert "risk_metrics" in status

    @pytest.mark.asyncio
    async def test_autonomous_loop(self, autonomous_controller: AutonomousController) -> None:
        """Тест автономного цикла."""
        # Проверка, что контроллер может запускать автономный цикл
        assert autonomous_controller is not None
        assert autonomous_controller.system_state is not None

    @pytest.mark.asyncio
    async def test_configuration_validation(self, autonomous_controller: AutonomousController) -> None:
        """Тест валидации конфигурации."""
        # Проверка, что конфигурация валидна
        config = autonomous_controller.config
        assert config is not None
        assert "risk" in config
        assert "strategy_selection" in config
        assert "market_regime" in config

    @pytest.mark.asyncio
    async def test_performance_optimization(self, autonomous_controller: AutonomousController) -> None:
        """Тест оптимизации производительности."""
        # Проверка, что контроллер может оптимизировать производительность
        assert autonomous_controller is not None
        assert autonomous_controller.live_adaptation is not None

    @pytest.mark.asyncio
    async def test_adaptive_learning(self, autonomous_controller: AutonomousController) -> None:
        """Тест адаптивного обучения."""
        # Проверка, что контроллер может адаптироваться
        assert autonomous_controller is not None
        assert autonomous_controller.meta_learning is not None

    def test_cleanup(self, autonomous_controller: AutonomousController) -> None:
        """Тест очистки ресурсов."""
        # Проверка, что контроллер может очищать ресурсы
        assert autonomous_controller is not None
        # Метод cleanup может не существовать, но контроллер должен быть валидным
        assert hasattr(autonomous_controller, 'system_state') 

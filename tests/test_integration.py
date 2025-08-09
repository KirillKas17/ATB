#!/usr/bin/env python3
"""
Интеграционные тесты для Syntra
"""

import asyncio

# Добавление корневой директории в путь
import sys
from pathlib import Path

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

sys.path.append(str(Path(__file__).parent.parent))

from infrastructure.agents.local_ai.controller import LocalAIController
from infrastructure.core.circuit_breaker import CircuitBreaker
from infrastructure.core.integration_manager import IntegrationManager
from infrastructure.core.portfolio_manager import PortfolioManager
from infrastructure.core.risk_manager import RiskManager
from infrastructure.ml_services.regime_discovery import RegimeDiscovery

from shared.event_bus import Event, EventBus, EventPriority
# from shared.health_checker import HealthChecker  # Временно отключен
# from shared.metrics import MetricsCollector  # Временно отключен
from shared.unified_cache import get_cache_manager


class TestIntegration:
    """Тесты интеграции всех компонентов"""

    @pytest.fixture
    def config(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Тестовая конфигурация"""
        return {
            "exchange": {
                "name": "testnet",
                "api_key": "test_key",
                "secret": "test_secret",
                "testnet": True,
            },
            "risk": {
                "max_risk_per_trade": 0.02,
                "max_daily_loss": 0.05,
                "max_weekly_loss": 0.15,
            },
            "portfolio": {"max_position_size": 0.20, "min_position_size": 0.01},
            "database": {
                "host": "localhost",
                "port": 5432,
                "user": "test",
                "password": "test",
                "database": "test_db",
            },
        }

    @pytest.fixture
    def event_bus(self: "TestEvolvableMarketMakerAgent") -> Any:
        """EventBus для тестов"""
        return EventBus()

    @pytest.mark.asyncio
    async def test_integration_manager_initialization(self, config) -> None:
        """Тест инициализации IntegrationManager"""
        manager = IntegrationManager(config)

        # Проверка создания компонентов
        assert manager.config == config
        assert manager.event_bus is not None
        assert manager.is_initialized == False
        assert manager.is_running == False

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, event_bus) -> None:
        """Тест интеграции CircuitBreaker"""
        circuit_breaker = CircuitBreaker(event_bus)

        # Проверка инициализации
        assert circuit_breaker.event_bus == event_bus
        assert "exchange" in circuit_breaker.circuits
        assert "database" in circuit_breaker.circuits

        # Тест триггера
        await circuit_breaker.trigger("exchange")
        status = await circuit_breaker.get_status()
        assert status["exchange"]["failure_count"] == 1

        # Тест проверки состояния
        is_open = await circuit_breaker.is_open("exchange")
        assert is_open == False  # Еще не открыт

        # Тест успешной операции
        await circuit_breaker.success("exchange")
        status = await circuit_breaker.get_status()
        assert status["exchange"]["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_risk_manager_integration(self, event_bus, config) -> None:
        """Тест интеграции RiskManager"""
        risk_config = config.get("risk", {})
        risk_manager = RiskManager(event_bus, risk_config)

        # Проверка инициализации
        assert risk_manager.event_bus == event_bus
        assert risk_manager.config == risk_config

        # Тест проверки риска сделки
        trade_params = {
            "symbol": "BTC/USDT",
            "size": 0.01,
            "price": 50000,
            "volatility": 0.02,
            "strategy_confidence": 0.8,
        }

        risk_assessment = await risk_manager.check_trade_risk(trade_params)
        assert risk_assessment["is_allowed"] == True
        assert "risk_score" in risk_assessment
        assert "position_size" in risk_assessment

        # Тест получения метрик
        metrics = await risk_manager.get_risk_metrics()
        assert isinstance(metrics, dict)
        assert "daily_loss" in metrics
        assert "weekly_loss" in metrics

    @pytest.mark.asyncio
    async def test_portfolio_manager_integration(self, event_bus, config) -> None:
        """Тест интеграции PortfolioManager"""
        portfolio_config = config.get("portfolio", {})
        portfolio_manager = PortfolioManager(event_bus, portfolio_config)

        # Проверка инициализации
        assert portfolio_manager.event_bus == event_bus
        assert portfolio_manager.portfolio_config == portfolio_config

        # Тест добавления позиции
        await portfolio_manager.add_position("BTC/USDT", 0.01, 50000)
        metrics = await portfolio_manager.get_portfolio_metrics()
        assert metrics["position_count"] == 1
        assert "BTC/USDT" in metrics["weights"]

        # Тест удаления позиции
        await portfolio_manager.remove_position("BTC/USDT", 0.01, 51000)
        metrics = await portfolio_manager.get_portfolio_metrics()
        assert metrics["position_count"] == 0

        # Тест получения корреляций
        correlations = await portfolio_manager.get_correlations()
        assert isinstance(correlations, dict)

    @pytest.mark.asyncio
    async def test_health_checker_integration(self, event_bus) -> None:
        """Тест интеграции HealthChecker"""
        health_checker = HealthChecker(event_bus)

        # Проверка инициализации
        assert health_checker.event_bus == event_bus

        # Тест проверки здоровья
        health_status = await health_checker.check_all_services()
        assert isinstance(health_status, dict)
        assert "overall_healthy" in health_status
        assert "checks" in health_status

        # Тест получения статуса
        status = await health_checker.get_health_status()
        assert isinstance(status, dict)

    @pytest.mark.asyncio
    async def test_metrics_collector_integration(self, event_bus) -> None:
        """Тест интеграции MetricsCollector"""
        metrics = MetricsCollector(event_bus)

        # Проверка инициализации
        assert metrics.event_bus == event_bus

        # Тест записи торговой операции
        trade_data = {
            "trade_id": "test_123",
            "symbol": "BTC/USDT",
            "strategy": "test_strategy",
            "pnl": 100.0,
            "side": "buy",
        }

        await metrics.record_trade(trade_data)

        # Тест получения метрик производительности
        performance = await metrics.get_performance_metrics()
        assert isinstance(performance, dict)
        assert "total_trades" in performance

        # Тест получения всех метрик
        all_metrics = await metrics.get_all_metrics()
        assert isinstance(all_metrics, dict)
        assert "performance" in all_metrics
        assert "trading" in all_metrics
        assert "system" in all_metrics

    @pytest.mark.asyncio
    async def test_cache_manager_integration(self: "TestIntegration") -> None:
        """Тест интеграции CacheManager"""
        cache_manager = get_cache_manager()
        cache = cache_manager.get_async_cache("test")

        # Проверка инициализации
        assert cache is not None

        # Тест установки и получения значения
        await cache.set("test_key", "test_value", 60)
        value = await cache.get("test_key")
        assert value == "test_value"

        # Тест проверки существования
        exists = await cache.exists("test_key")
        assert exists == True

        # Тест удаления
        await cache.delete("test_key")
        value = await cache.get("test_key")
        assert value is None

        # Тест получения статистики
        stats = await cache.get_stats()
        assert isinstance(stats, dict)
        assert "size" in stats

    @pytest.mark.asyncio
    async def test_regime_discovery_integration(self, event_bus) -> None:
        """Тест интеграции RegimeDiscovery"""
        regime_discovery = RegimeDiscovery(event_bus)

        # Проверка инициализации
        assert regime_discovery.event_bus == event_bus

        # Тест получения текущего режима
        current_regime = regime_discovery.get_current_regime()
        assert current_regime == "unknown"

        # Тест получения истории режимов
        history = regime_discovery.get_regime_history()
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_ai_controller_integration(self, event_bus) -> None:
        """Тест интеграции LocalAIController"""
        ai_controller = LocalAIController(event_bus)

        # Проверка инициализации
        assert ai_controller.event_bus == event_bus

        # Тест получения статуса ИИ
        status = ai_controller.get_ai_status()
        assert isinstance(status, dict)
        assert "knowledge_base_size" in status
        assert "decision_thresholds" in status
        assert "is_running" in status

    @pytest.mark.asyncio
    async def test_event_bus_integration(self, event_bus) -> None:
        """Тест интеграции EventBus"""
        events_received = []

        # Подписка на события
        def event_handler(event: Event) -> Any:
            events_received.append(event)

        event_bus.subscribe("test.event", event_handler)

        # Отправка события
        test_event = Event(
            event_type="test.event",
            data={"message": "test"},
            priority=EventPriority.NORMAL,
        )

        await event_bus.publish(test_event)

        # Проверка получения события
        assert len(events_received) == 1
        assert events_received[0].event_type == "test.event"
        assert events_received[0].data["message"] == "test"

    @pytest.mark.asyncio
    async def test_full_system_integration(self, config) -> None:
        """Тест полной интеграции системы"""
        manager = IntegrationManager(config)

        # Инициализация системы
        await manager.initialize()

        # Проверка инициализации
        assert manager.is_initialized == True

        # Получение статуса системы
        status = await manager.get_system_status()
        assert isinstance(status, dict)
        assert "is_initialized" in status
        assert "is_running" in status
        assert "components" in status

        # Остановка системы
        await manager.stop()
        assert manager.is_running == False

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, event_bus) -> None:
        """Тест обработки ошибок"""
        # Тест CircuitBreaker с ошибками
        circuit_breaker = CircuitBreaker(event_bus)

        # Множественные ошибки для активации circuit breaker
        for _ in range(10):
            await circuit_breaker.trigger("api")

        # Проверка открытия circuit breaker
        is_open = await circuit_breaker.is_open("api")
        assert is_open == True

        # Тест сброса circuit breaker
        await circuit_breaker.reset_circuit("api")
        is_open = await circuit_breaker.is_open("api")
        assert is_open == False

    @pytest.mark.asyncio
    async def test_performance_integration(self, event_bus) -> None:
        """Тест производительности"""
        metrics = MetricsCollector(event_bus)

        # Тест таймеров
        await metrics.start_timer("test_timer")
        await asyncio.sleep(0.1)  # Небольшая задержка
        duration = await metrics.stop_timer("test_timer")

        assert duration is not None
        assert duration > 0

        # Тест счетчиков
        await metrics.increment_counter("test_counter", 5)
        all_metrics = await metrics.get_all_metrics()
        assert all_metrics["counters"]["test_counter"] == 5

    @pytest.mark.asyncio
    async def test_data_persistence_integration(self, event_bus) -> None:
        """Тест персистентности данных"""
        cache_manager = get_cache_manager()
        cache = cache_manager.get_async_cache("persistence_test")

        # Тест множественных операций
        for i in range(10):
            await cache.set(f"key_{i}", f"value_{i}", 60)

        # Проверка всех значений
        for i in range(10):
            value = await cache.get(f"key_{i}")
            assert value == f"value_{i}"

        # Тест получения множественных значений
        keys = [f"key_{i}" for i in range(10)]
        values = await cache.get_multiple(keys)
        assert len(values) == 10

        # Очистка
        await cache.clear()
        stats = await cache.get_stats()
        assert stats["size"] == 0


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"])

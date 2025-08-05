"""
Интеграционные тесты для мониторинга стратегий.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal
from domain.strategies import (
    get_strategy_factory, get_strategy_registry, get_strategy_validator
)
from domain.strategies.exceptions import (
    StrategyCreationError, StrategyValidationError, StrategyRegistryError
)
from domain.entities.strategy import StrategyType, StrategyStatus
from domain.entities.market import MarketData, OrderBook, Trade
class TestStrategyMonitoringIntegration:
    """Интеграционные тесты для мониторинга стратегий."""
    @pytest.fixture
    def factory(self: "TestEvolvableMarketMakerAgent") -> Any:
        return get_strategy_factory()
    @pytest.fixture
    def registry(self: "TestEvolvableMarketMakerAgent") -> Any:
        return get_strategy_registry()
    @pytest.fixture
    def validator(self: "TestEvolvableMarketMakerAgent") -> Any:
        return get_strategy_validator()
    @pytest.fixture
    def sample_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создает тестовые рыночные данные."""
        return MarketData(
            symbol="BTC/USDT",
            timestamp=datetime.now(),
            price=Decimal("50000"),
            volume=Decimal("1000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            open_price=Decimal("49500"),
            close_price=Decimal("50000"),
            order_book=OrderBook(
                symbol="BTC/USDT",
                timestamp=datetime.now(),
                bids=[{"price": Decimal("49999"), "size": Decimal("1.0")}],
                asks=[{"price": Decimal("50001"), "size": Decimal("1.0")}]
            ),
            trades=[
                Trade(
                    id="trade_1",
                    symbol="BTC/USDT",
                    price=Decimal("50000"),
                    size=Decimal("0.1"),
                    side="buy",
                    timestamp=datetime.now()
                )
            ]
        )
    def test_strategy_performance_monitoring(self, factory, registry, sample_market_data) -> None:
        """Тест мониторинга производительности стратегий."""
        # Создаем несколько стратегий
        strategies = []
        for i in range(3):
            strategy = factory.create_strategy(
                name=f"monitoring_test_{i}",
                trading_pairs=["BTC/USDT"],
                parameters={"sma_period": 20 + i * 5},
                risk_level="medium",
                confidence_threshold=Decimal("0.6")
            )
            strategy_id = registry.register_strategy(
                strategy=strategy,
                name=f"Monitoring Test {i}",
                tags=["monitoring", "test"],
                priority=i + 1
            )
            strategies.append((strategy_id, strategy))
        # Симулируем выполнение стратегий
        for strategy_id, strategy in strategies:
            # Обновляем метрики производительности
            registry.update_strategy_metrics(
                strategy_id=strategy_id,
                execution_count=10 + strategy_id.int % 10,
                success_count=7 + strategy_id.int % 7,
                total_pnl=Decimal("1000") + Decimal(str(strategy_id.int % 1000)),
                max_drawdown=Decimal("0.05") + Decimal(str(strategy_id.int % 5)) / 100
            )
        # Получаем статистику производительности
        performance_stats = registry.get_performance_statistics()
        assert performance_stats is not None
        assert performance_stats.total_strategies == 3
        assert performance_stats.active_strategies == 3
        assert performance_stats.average_success_rate > 0
        assert performance_stats.total_pnl > 0
    def test_strategy_alert_system(self, factory, registry) -> None:
        """Тест системы алертов для стратегий."""
        # Создаем стратегию с плохой производительностью
        poor_strategy = factory.create_strategy(
            name="poor_performance_strategy",
            trading_pairs=["BTC/USDT"],
            parameters={"sma_period": 20},
            risk_level="high",
            confidence_threshold=Decimal("0.8")
        )
        strategy_id = registry.register_strategy(
            strategy=poor_strategy,
            name="Poor Performance Strategy",
            tags=["alert", "test"],
            priority=1
        )
        # Симулируем плохую производительность
        registry.update_strategy_metrics(
            strategy_id=strategy_id,
            execution_count=100,
            success_count=20,  # 20% успешность
            total_pnl=Decimal("-5000"),  # Отрицательный PnL
            max_drawdown=Decimal("0.25")  # 25% просадка
        )
        # Проверяем, что система генерирует алерты
        alerts = registry.get_strategy_alerts()
        assert len(alerts) > 0
        assert any(alert.strategy_id == strategy_id for alert in alerts)
        assert any(alert.alert_type in ["low_success_rate", "negative_pnl", "high_drawdown"] 
                  for alert in alerts)
    def test_mass_strategy_registration(self, factory, registry) -> None:
        """Тест массовой регистрации стратегий."""
        # Создаем 100 стратегий
        strategy_ids = []
        for i in range(100):
            strategy = factory.create_strategy(
                name=f"mass_test_{i}",
                trading_pairs=["BTC/USDT"],
                parameters={"sma_period": 20 + i % 10},
                risk_level="medium",
                confidence_threshold=Decimal("0.6")
            )
            strategy_id = registry.register_strategy(
                strategy=strategy,
                name=f"Mass Test {i}",
                tags=["mass", "test"],
                priority=i % 5 + 1
            )
            strategy_ids.append(strategy_id)
        # Проверяем, что все стратегии зарегистрированы
        stats = registry.get_registry_statistics()
        assert stats.total_strategies >= 100
        # Проверяем поиск по тегам
        mass_strategies = registry.search_strategies(tags=["mass"])
        assert len(mass_strategies) >= 100
        # Проверяем поиск по приоритету
        high_priority = registry.search_strategies(priority=5)
        assert len(high_priority) > 0
    def test_strategy_bulk_operations(self, factory, registry) -> None:
        """Тест массовых операций со стратегиями."""
        # Создаем стратегии для массовых операций
        strategies = []
        for i in range(50):
            strategy = factory.create_strategy(
                name=f"bulk_test_{i}",
                trading_pairs=["BTC/USDT"],
                parameters={"sma_period": 20},
                risk_level="medium",
                confidence_threshold=Decimal("0.6")
            )
            strategy_id = registry.register_strategy(
                strategy=strategy,
                name=f"Bulk Test {i}",
                tags=["bulk", "test"],
                priority=1
            )
            strategies.append((strategy_id, strategy))
        # Массовое обновление статуса
        strategy_ids = [s[0] for s in strategies]
        registry.bulk_update_status(strategy_ids, StrategyStatus.ACTIVE)
        # Проверяем, что все стратегии активны
        for strategy_id in strategy_ids:
            metadata = registry.get_strategy_metadata(strategy_id)
            assert metadata.status == StrategyStatus.ACTIVE
        # Массовое обновление метрик
        for strategy_id in strategy_ids:
            registry.update_strategy_metrics(
                strategy_id=strategy_id,
                execution_count=50,
                success_count=35,
                total_pnl=Decimal("1000"),
                max_drawdown=Decimal("0.1")
            )
        # Проверяем статистику
        stats = registry.get_registry_statistics()
        assert stats.active_strategies >= 50
    def test_strategy_metrics_aggregation(self, factory, registry) -> None:
        """Тест агрегации метрик стратегий."""
        # Создаем стратегии разных типов
        strategy_types = [
            StrategyType.TREND_FOLLOWING,
            StrategyType.MEAN_REVERSION,
            StrategyType.BREAKOUT
        ]
        for strategy_type in strategy_types:
            for i in range(5):
                strategy = factory.create_strategy(
                    name=f"metrics_test_{strategy_type.value}_{i}",
                    trading_pairs=["BTC/USDT"],
                    parameters={"sma_period": 20},
                    risk_level="medium",
                    confidence_threshold=Decimal("0.6")
                )
                strategy_id = registry.register_strategy(
                    strategy=strategy,
                    name=f"Metrics Test {strategy_type.value} {i}",
                    tags=["metrics", strategy_type.value],
                    priority=1
                )
                # Обновляем метрики с разными значениями
                registry.update_strategy_metrics(
                    strategy_id=strategy_id,
                    execution_count=100 + i * 10,
                    success_count=70 + i * 5,
                    total_pnl=Decimal("1000") + Decimal(str(i * 100)),
                    max_drawdown=Decimal("0.05") + Decimal(str(i)) / 100
                )
        # Получаем агрегированные метрики по типам
        type_metrics = registry.get_metrics_by_type()
        assert len(type_metrics) >= 3
        for strategy_type in strategy_types:
            assert strategy_type in type_metrics
            metrics = type_metrics[strategy_type]
            assert metrics.total_strategies >= 5
            assert metrics.average_success_rate > 0
    def test_strategy_performance_comparison(self, factory, registry) -> None:
        """Тест сравнения производительности стратегий."""
        # Создаем стратегии с разной производительностью
        performance_levels = [
            {"success_rate": 0.9, "pnl": 5000, "drawdown": 0.05},  # Высокая
            {"success_rate": 0.7, "pnl": 2000, "drawdown": 0.10},  # Средняя
            {"success_rate": 0.5, "pnl": -1000, "drawdown": 0.20}  # Низкая
        ]
        strategy_ids = []
        for i, perf in enumerate(performance_levels):
            strategy = factory.create_strategy(
                name=f"comparison_test_{i}",
                trading_pairs=["BTC/USDT"],
                parameters={"sma_period": 20},
                risk_level="medium",
                confidence_threshold=Decimal("0.6")
            )
            strategy_id = registry.register_strategy(
                strategy=strategy,
                name=f"Comparison Test {i}",
                tags=["comparison", "test"],
                priority=1
            )
            # Обновляем метрики
            execution_count = 100
            success_count = int(execution_count * perf["success_rate"])
            registry.update_strategy_metrics(
                strategy_id=strategy_id,
                execution_count=execution_count,
                success_count=success_count,
                total_pnl=Decimal(str(perf["pnl"])),
                max_drawdown=Decimal(str(perf["drawdown"]))
            )
            strategy_ids.append(strategy_id)
        # Получаем рейтинг стратегий
        ranking = registry.get_strategy_ranking(limit=10)
        assert len(ranking) >= 3
        # Проверяем, что стратегии отсортированы по производительности
        assert ranking[0].success_rate >= ranking[1].success_rate
    def test_strategy_health_monitoring(self, factory, registry) -> None:
        """Тест мониторинга здоровья стратегий."""
        # Создаем стратегии с разными проблемами
        health_issues = [
            {"execution_count": 0, "last_execution": None},  # Неактивная
            {"execution_count": 1000, "success_rate": 0.1},  # Низкая успешность
            {"execution_count": 100, "error_rate": 0.5},     # Высокая ошибка
        ]
        for i, issue in enumerate(health_issues):
            strategy = factory.create_strategy(
                name=f"health_test_{i}",
                trading_pairs=["BTC/USDT"],
                parameters={"sma_period": 20},
                risk_level="medium",
                confidence_threshold=Decimal("0.6")
            )
            strategy_id = registry.register_strategy(
                strategy=strategy,
                name=f"Health Test {i}",
                tags=["health", "test"],
                priority=1
            )
            # Обновляем метрики с проблемами
            if "execution_count" in issue:
                registry.update_strategy_metrics(
                    strategy_id=strategy_id,
                    execution_count=issue["execution_count"],
                    success_count=issue.get("success_rate", 0.7) * issue["execution_count"]
                )
        # Получаем отчет о здоровье
        health_report = registry.get_health_report()
        assert health_report is not None
        assert health_report.total_strategies >= 3
        assert health_report.unhealthy_strategies >= 1
        assert len(health_report.health_issues) > 0
    def test_strategy_monitoring_real_time(self, factory, registry, sample_market_data) -> None:
        """Тест мониторинга стратегий в реальном времени."""
        # Создаем стратегию для мониторинга
        strategy = factory.create_strategy(
            name="realtime_monitoring_test",
            trading_pairs=["BTC/USDT"],
            parameters={"sma_period": 20},
            risk_level="medium",
            confidence_threshold=Decimal("0.6")
        )
        strategy_id = registry.register_strategy(
            strategy=strategy,
            name="Real-time Monitoring Test",
            tags=["realtime", "monitoring"],
            priority=1
        )
        # Симулируем выполнение в реальном времени
        for i in range(10):
            # Обновляем метрики
            registry.update_strategy_metrics(
                strategy_id=strategy_id,
                execution_count=i + 1,
                success_count=i,
                total_pnl=Decimal(str(i * 100)),
                max_drawdown=Decimal("0.05")
            )
            # Проверяем, что метрики обновляются
            metrics = registry.get_strategy_metrics(strategy_id)
            assert metrics.execution_count == i + 1
            assert metrics.success_count == i
        # Проверяем финальные метрики
        final_metrics = registry.get_strategy_metrics(strategy_id)
        assert final_metrics.execution_count == 10
        assert final_metrics.success_count == 9
        assert final_metrics.success_rate == 0.9 

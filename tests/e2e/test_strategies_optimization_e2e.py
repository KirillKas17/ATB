"""
End-to-End тесты для автоматической оптимизации стратегий.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal
from domain.strategies import get_strategy_factory, get_strategy_registry, get_strategy_validator
from domain.strategies.exceptions import StrategyCreationError, StrategyValidationError, StrategyRegistryError
from domain.entities.market import MarketData, OrderBook, Trade


class TestStrategyOptimizationE2E:
    """End-to-End тесты для оптимизации стратегий."""

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
    def realistic_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создает реалистичные рыночные данные для оптимизации."""
        data_series = []
        base_price = Decimal("50000")
        current_price = base_price
        for i in range(500):  # 500 точек данных для оптимизации
            timestamp = datetime.now() + timedelta(minutes=i)
            # Симулируем различные рыночные условия
            if i < 100:
                # Трендовый рынок
                price_change = Decimal(str(0.003 * (i + 1)))
                current_price = base_price + price_change
            elif i < 200:
                # Боковой рынок
                price_change = Decimal(str(0.001 * (i - 99)))
                current_price = base_price + Decimal("300") + price_change
            elif i < 300:
                # Волатильный рынок
                volatility = Decimal(str(0.01 * (i - 199)))
                current_price = base_price + Decimal("200") + volatility
            elif i < 400:
                # Нисходящий тренд
                price_change = Decimal(str(-0.002 * (i - 299)))
                current_price = base_price + Decimal("500") + price_change
            else:
                # Восстановление
                price_change = Decimal(str(0.001 * (i - 399)))
                current_price = base_price + Decimal("100") + price_change
            # Добавляем случайный шум
            noise = Decimal(str(0.002 * (i % 20 - 10)))
            current_price += noise
            # Создаем OHLCV данные
            open_price = current_price - Decimal("10")
            high_price = current_price + Decimal("20")
            low_price = current_price - Decimal("20")
            close_price = current_price
            # Объем с трендом
            base_volume = Decimal("1000")
            volume_trend = Decimal(str(1 + 0.2 * (i % 30)))
            volume = base_volume * volume_trend
            data = MarketData(
                symbol="BTC/USDT",
                timestamp=timestamp,
                price=current_price,
                volume=volume,
                high=high_price,
                low=low_price,
                open_price=open_price,
                close_price=close_price,
                order_book=OrderBook(
                    symbol="BTC/USDT",
                    timestamp=timestamp,
                    bids=[{"price": current_price - Decimal("1"), "size": Decimal("1.0")}],
                    asks=[{"price": current_price + Decimal("1"), "size": Decimal("1.0")}],
                ),
                trades=[
                    Trade(
                        id=f"trade_{i}",
                        symbol="BTC/USDT",
                        price=current_price,
                        size=Decimal("0.1"),
                        side="buy",
                        timestamp=timestamp,
                    )
                ],
            )
            data_series.append(data)
        return data_series

    def test_automatic_parameter_optimization_e2e(self, factory, registry, realistic_market_data) -> None:
        """E2E тест автоматической оптимизации параметров."""
        # 1. Создаем базовую стратегию
        base_strategy = factory.create_strategy(
            name="optimization_base",
            trading_pairs=["BTC/USDT"],
            parameters={"sma_period": 20, "ema_period": 12, "rsi_period": 14},
            risk_level="medium",
            confidence_threshold=Decimal("0.6"),
        )
        base_strategy_id = registry.register_strategy(
            strategy=base_strategy, name="Optimization Base Strategy", tags=["optimization", "base"], priority=1
        )
        # 2. Определяем диапазоны параметров для оптимизации
        parameter_ranges = {"sma_period": (10, 50), "ema_period": (5, 25), "rsi_period": (10, 20)}
        # 3. Создаем варианты стратегий с разными параметрами
        optimization_candidates = []
        # Генерируем комбинации параметров
        sma_periods = [15, 20, 25, 30, 35]
        ema_periods = [8, 12, 16, 20]
        rsi_periods = [12, 14, 16, 18]
        for sma in sma_periods:
            for ema in ema_periods:
                for rsi in rsi_periods:
                    if ema < sma:  # Логическое ограничение
                        strategy = factory.create_strategy(
                            name=f"optimization_candidate_{len(optimization_candidates)}",
                            trading_pairs=["BTC/USDT"],
                            parameters={"sma_period": sma, "ema_period": ema, "rsi_period": rsi},
                            risk_level="medium",
                            confidence_threshold=Decimal("0.6"),
                        )
                        strategy_id = registry.register_strategy(
                            strategy=strategy,
                            name=f"Optimization Candidate {len(optimization_candidates)}",
                            tags=["optimization", "candidate"],
                            priority=2,
                        )
                        optimization_candidates.append(
                            {
                                "id": strategy_id,
                                "strategy": strategy,
                                "parameters": {"sma_period": sma, "ema_period": ema, "rsi_period": rsi},
                            }
                        )
        # 4. Симулируем бэктестинг для всех кандидатов
        results = []
        for candidate in optimization_candidates:
            # Симулируем выполнение стратегии на исторических данных
            total_pnl = Decimal("0")
            execution_count = 0
            success_count = 0
            max_drawdown = Decimal("0")
            for i, market_data in enumerate(realistic_market_data[100:]):  # Пропускаем первые 100 точек
                # Простая симуляция торговли
                if i % 20 == 0:  # Каждые 20 точек
                    execution_count += 1
                    # Симулируем успешность на основе параметров
                    param_score = (
                        candidate["parameters"]["sma_period"] / 50
                        + candidate["parameters"]["ema_period"] / 25
                        + candidate["parameters"]["rsi_period"] / 20
                    ) / 3
                    if param_score > 0.5:
                        success_count += 1
                        pnl = Decimal("100") * param_score
                    else:
                        pnl = Decimal("-50") * (1 - param_score)
                    total_pnl += pnl
                    # Обновляем максимальную просадку
                    if pnl < 0:
                        drawdown = abs(pnl) / Decimal("1000")
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown
            # Обновляем метрики кандидата
            registry.update_strategy_metrics(
                strategy_id=candidate["id"],
                execution_count=execution_count,
                success_count=success_count,
                total_pnl=total_pnl,
                max_drawdown=max_drawdown,
            )
            results.append(
                {
                    "candidate": candidate,
                    "metrics": {
                        "execution_count": execution_count,
                        "success_count": success_count,
                        "success_rate": success_count / execution_count if execution_count > 0 else 0,
                        "total_pnl": total_pnl,
                        "max_drawdown": max_drawdown,
                    },
                }
            )
        # 5. Анализируем результаты и выбираем лучшую стратегию
        best_result = max(results, key=lambda x: x["metrics"]["total_pnl"])
        best_candidate = best_result["candidate"]
        # 6. Проверяем, что оптимизация дала результат
        assert best_result["metrics"]["total_pnl"] > 0
        assert best_result["metrics"]["success_rate"] > 0.5
        # 7. Обновляем теги лучшей стратегии
        registry.update_strategy_metadata(
            strategy_id=best_candidate["id"], tags=["optimization", "best", "optimized"], priority=1
        )
        # 8. Проверяем статистику оптимизации
        optimization_stats = registry.get_optimization_statistics()
        assert optimization_stats.total_candidates >= len(optimization_candidates)
        assert optimization_stats.best_strategy_id == best_candidate["id"]

    def test_ab_testing_e2e(self, factory, registry, realistic_market_data) -> None:
        """E2E тест A/B тестирования стратегий."""
        # 1. Создаем две версии стратегии для A/B тестирования
        strategy_a = factory.create_strategy(
            name="ab_test_strategy_a",
            trading_pairs=["BTC/USDT"],
            parameters={"sma_period": 20, "ema_period": 12, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70},
            risk_level="medium",
            confidence_threshold=Decimal("0.6"),
        )
        strategy_b = factory.create_strategy(
            name="ab_test_strategy_b",
            trading_pairs=["BTC/USDT"],
            parameters={
                "sma_period": 25,  # Другой период
                "ema_period": 15,  # Другой период
                "rsi_period": 16,  # Другой период
                "rsi_oversold": 25,  # Другие уровни
                "rsi_overbought": 75,
            },
            risk_level="medium",
            confidence_threshold=Decimal("0.7"),  # Другой порог
        )
        strategy_a_id = registry.register_strategy(
            strategy=strategy_a, name="A/B Test Strategy A", tags=["ab_test", "version_a"], priority=1
        )
        strategy_b_id = registry.register_strategy(
            strategy=strategy_b, name="A/B Test Strategy B", tags=["ab_test", "version_b"], priority=1
        )
        # 2. Симулируем A/B тестирование
        ab_results = {"A": [], "B": []}
        # Разделяем данные на две части для A/B тестирования
        data_a = realistic_market_data[:250]
        data_b = realistic_market_data[250:]
        # Тестируем версию A
        for i, market_data in enumerate(data_a):
            if i % 15 == 0:  # Каждые 15 точек
                # Симулируем результат
                result = {
                    "timestamp": market_data.timestamp,
                    "pnl": Decimal("50") if i % 3 == 0 else Decimal("-20"),
                    "success": i % 3 == 0,
                }
                ab_results["A"].append(result)
        # Тестируем версию B
        for i, market_data in enumerate(data_b):
            if i % 15 == 0:  # Каждые 15 точек
                # Симулируем результат (немного лучше для B)
                result = {
                    "timestamp": market_data.timestamp,
                    "pnl": Decimal("60") if i % 3 == 0 else Decimal("-15"),
                    "success": i % 3 == 0,
                }
                ab_results["B"].append(result)
        # 3. Обновляем метрики для обеих версий
        for version, results in ab_results.items():
            strategy_id = strategy_a_id if version == "A" else strategy_b_id
            total_pnl = sum(r["pnl"] for r in results)
            success_count = sum(1 for r in results if r["success"])
            execution_count = len(results)
            registry.update_strategy_metrics(
                strategy_id=strategy_id,
                execution_count=execution_count,
                success_count=success_count,
                total_pnl=total_pnl,
                max_drawdown=Decimal("0.1"),
            )
        # 4. Анализируем результаты A/B тестирования
        metrics_a = registry.get_strategy_metrics(strategy_a_id)
        metrics_b = registry.get_strategy_metrics(strategy_b_id)
        # 5. Определяем победителя
        if metrics_b.total_pnl > metrics_a.total_pnl:
            winner_id = strategy_b_id
            winner_version = "B"
        else:
            winner_id = strategy_a_id
            winner_version = "A"
        # 6. Обновляем теги победителя
        registry.update_strategy_metadata(
            strategy_id=winner_id, tags=["ab_test", "winner", f"version_{winner_version.lower()}"], priority=1
        )
        # 7. Проверяем результаты A/B тестирования
        assert metrics_a.execution_count > 0
        assert metrics_b.execution_count > 0
        assert metrics_a.total_pnl != metrics_b.total_pnl  # Должны быть разные результаты
        # 8. Получаем отчет A/B тестирования
        ab_report = registry.get_ab_test_report([strategy_a_id, strategy_b_id])
        assert ab_report is not None
        assert ab_report.winner_id == winner_id
        assert ab_report.confidence_level > 0.5  # Статистическая значимость

    def test_risk_management_automation_e2e(self, factory, registry, realistic_market_data) -> None:
        """E2E тест автоматического управления рисками."""
        # 1. Создаем стратегию с автоматическим риск-менеджментом
        strategy = factory.create_strategy(
            name="risk_managed_strategy",
            trading_pairs=["BTC/USDT"],
            parameters={
                "sma_period": 20,
                "ema_period": 12,
                "rsi_period": 14,
                "max_position_size": 0.1,  # 10% от капитала
                "stop_loss": 0.05,  # 5% стоп-лосс
                "take_profit": 0.15,  # 15% тейк-профит
            },
            risk_level="high",
            confidence_threshold=Decimal("0.8"),
        )
        strategy_id = registry.register_strategy(
            strategy=strategy, name="Risk Managed Strategy", tags=["risk_management", "automated"], priority=1
        )
        # 2. Симулируем торговлю с автоматическим риск-менеджментом
        initial_capital = Decimal("10000")
        current_capital = initial_capital
        position_size = Decimal("0")
        max_drawdown = Decimal("0")
        risk_events = []
        for i, market_data in enumerate(realistic_market_data):
            # Симулируем торговые сигналы
            if i % 20 == 0:  # Каждые 20 точек
                # Проверяем риск-лимиты
                current_drawdown = (initial_capital - current_capital) / initial_capital
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
                # Автоматическое управление позицией
                if current_drawdown > Decimal("0.1"):  # 10% просадка
                    # Уменьшаем размер позиции
                    position_size = min(position_size, Decimal("0.05"))
                    risk_events.append(
                        {"type": "position_reduction", "reason": "high_drawdown", "timestamp": market_data.timestamp}
                    )
                # Симулируем торговлю
                if i % 3 == 0:  # Покупка
                    trade_size = position_size * current_capital
                    if trade_size > 0:
                        # Симулируем результат
                        if i % 5 == 0:  # Успешная сделка
                            profit = trade_size * Decimal("0.1")
                            current_capital += profit
                        else:  # Убыточная сделка
                            loss = trade_size * Decimal("0.05")
                            current_capital -= loss
                # Обновляем метрики риск-менеджмента
                registry.update_risk_metrics(
                    strategy_id=strategy_id,
                    current_capital=current_capital,
                    max_drawdown=max_drawdown,
                    position_size=position_size,
                    risk_events=risk_events,
                )
        # 3. Проверяем эффективность риск-менеджмента
        risk_metrics = registry.get_risk_metrics(strategy_id)
        assert risk_metrics is not None
        assert risk_metrics.max_drawdown <= Decimal("0.2")  # Максимальная просадка не более 20%
        assert risk_metrics.final_capital > initial_capital * Decimal("0.8")  # Сохранили минимум 80% капитала
        assert len(risk_metrics.risk_events) > 0  # Были события риск-менеджмента
        # 4. Анализируем события риск-менеджмента
        position_reductions = [e for e in risk_metrics.risk_events if e["type"] == "position_reduction"]
        assert len(position_reductions) > 0
        # 5. Проверяем, что риск-менеджмент сработал
        final_metrics = registry.get_strategy_metrics(strategy_id)
        assert final_metrics.max_drawdown <= Decimal("0.2")
        # 6. Получаем отчет по риск-менеджменту
        risk_report = registry.get_risk_management_report(strategy_id)
        assert risk_report is not None
        assert risk_report.risk_score <= 0.3  # Низкий риск
        assert risk_report.capital_preservation_rate > 0.8  # Высокое сохранение капитала

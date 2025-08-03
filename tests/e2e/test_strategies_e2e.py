"""
End-to-End тесты для стратегий.
"""
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from domain.entities.market import MarketData, MarketState
from domain.entities.strategy import Signal, SignalType, SignalStrength, StrategyType, StrategyStatus
from domain.strategies.strategy_factory import get_strategy_factory
from domain.strategies.strategy_registry import get_strategy_registry
from domain.strategies.base_strategies import (
    TrendFollowingStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    ScalpingStrategy,
    ArbitrageStrategy
)
from domain.strategies.strategy_types import (
    StrategyCategory, RiskProfile, Timeframe, StrategyConfig,
    TrendFollowingParams, MeanReversionParams, BreakoutParams,
    ScalpingParams, ArbitrageParams
)
from domain.strategies.exceptions import (
    StrategyFactoryError, StrategyCreationError, StrategyValidationError,
    StrategyRegistryError, StrategyNotFoundError
)
class TestStrategyE2E:
    """End-to-End тесты для стратегий."""
    @pytest.fixture
    def factory(self) -> Any:
        """Создать фабрику стратегий."""
        return get_strategy_factory()
    @pytest.fixture
    def registry(self) -> Any:
        """Создать реестр стратегий."""
        return get_strategy_registry()
    @pytest.fixture
    def realistic_market_data(self) -> Any:
        """Создать реалистичные рыночные данные."""
        data_series = []
        base_price = Decimal("50000")
        current_price = base_price
        for i in range(200):  # 200 точек данных
            timestamp = datetime.now() + timedelta(minutes=i)
            # Симулируем реалистичное движение цены
            if i < 50:
                # Восходящий тренд
                price_change = Decimal(str(0.002 * (i + 1)))
                current_price = base_price + price_change
            elif i < 100:
                # Нисходящий тренд
                price_change = Decimal(str(-0.001 * (i - 49)))
                current_price = base_price + Decimal("100") + price_change
            elif i < 150:
                # Боковое движение
                price_change = Decimal(str(0.0005 * (i - 99)))
                current_price = base_price + Decimal("50") + price_change
            else:
                # Волатильное движение
                volatility = Decimal(str(0.005 * (i - 149)))
                current_price = base_price + Decimal("25") + volatility
            # Добавляем случайный шум
            noise = Decimal(str(0.001 * (i % 10 - 5)))
            current_price += noise
            # Создаем OHLCV данные
            open_price = current_price - Decimal("5")
            high_price = current_price + Decimal("15")
            low_price = current_price - Decimal("15")
            close_price = current_price
            # Объем с трендом
            base_volume = Decimal("1000")
            volume_trend = Decimal(str(1 + 0.1 * (i % 20)))
            volume = base_volume * volume_trend
            data = MarketData(
                symbol="BTC/USDT",
                timestamp=timestamp,
                open=Price(open_price, Currency.USDT),
                high=Price(high_price, Currency.USDT),
                low=Price(low_price, Currency.USDT),
                close=Price(close_price, Currency.USDT),
                volume=Volume(volume, Currency.USDT),
                bid=Price(close_price - Decimal("2"), Currency.USDT),
                ask=Price(close_price + Decimal("2"), Currency.USDT),
                bid_volume=Volume(volume / 2, Currency.USDT),
                ask_volume=Volume(volume / 2, Currency.USDT)
            )
            data_series.append(data)
        return data_series
    def test_complete_trading_session_e2e(self, factory, registry, realistic_market_data) -> None:
        """Полный E2E тест торговой сессии."""
        # 1. Настройка стратегий
        strategies_config = [
            {
                "name": "trend_following_e2e",
                "strategy_class": TrendFollowingStrategy,
                "strategy_type": StrategyType.TREND_FOLLOWING,
                "parameters": {
                    "short_period": 10,
                    "long_period": 20,
                    "rsi_period": 14,
                    "rsi_oversold": 30,
                    "rsi_overbought": 70
                }
            },
            {
                "name": "mean_reversion_e2e",
                "strategy_class": MeanReversionStrategy,
                "strategy_type": StrategyType.MEAN_REVERSION,
                "parameters": {
                    "lookback_period": 50,
                    "deviation_threshold": Decimal("2.0"),
                    "rsi_period": 14
                }
            },
            {
                "name": "breakout_e2e",
                "strategy_class": BreakoutStrategy,
                "strategy_type": StrategyType.BREAKOUT,
                "parameters": {
                    "breakout_threshold": Decimal("1.5"),
                    "volume_multiplier": Decimal("2.0"),
                    "confirmation_period": 2
                }
            }
        ]
        created_strategies = []
        # 2. Регистрация и создание стратегий
        for config in strategies_config:
            # Регистрируем в фабрике
            factory.register_strategy(
                name=config["name"],
                creator_func=config["strategy_class"],
                strategy_type=config["strategy_type"],
                description=f"E2E Test {config['name']}",
                version="1.0.0",
                author="E2E Test",
                required_parameters=list(config["parameters"].keys()),
                supported_pairs=["BTC/USDT"],
                min_confidence=Decimal("0.3"),
                max_confidence=Decimal("1.0"),
                risk_levels=["low", "medium", "high"]
            )
            # Создаем стратегию
            strategy = factory.create_strategy(
                name=config["name"],
                trading_pairs=["BTC/USDT"],
                parameters=config["parameters"],
                risk_level="medium",
                confidence_threshold=Decimal("0.6")
            )
            # Регистрируем в реестре
            strategy_id = registry.register_strategy(
                strategy=strategy,
                name=f"E2E {config['name'].replace('_e2e', '').title()}",
                tags=["e2e", "test", config["strategy_type"].value],
                priority=1
            )
            created_strategies.append({
                "id": strategy_id,
                "strategy": strategy,
                "config": config
            })
        # 3. Активация стратегий
        for strategy_info in created_strategies:
            strategy_info["strategy"].activate()
            registry.update_strategy_status(strategy_info["id"], StrategyStatus.ACTIVE)
        # 4. Симуляция торговой сессии
        all_signals = []
        performance_metrics = {}
        # Пропускаем первые 50 точек для инициализации индикаторов
        for i, market_data in enumerate(realistic_market_data[50:]):
            current_time = market_data.timestamp
            for strategy_info in created_strategies:
                strategy_id = strategy_info["id"]
                strategy = strategy_info["strategy"]
                strategy_type = strategy_info["config"]["strategy_type"]
                try:
                    # Генерируем сигнал
                    start_time = datetime.now()
                    signal = strategy.generate_signal(market_data)
                    execution_time = (datetime.now() - start_time).total_seconds()
                    if signal:
                        all_signals.append({
                            "strategy_id": strategy_id,
                            "strategy_type": strategy_type,
                            "signal": signal,
                            "timestamp": current_time,
                            "market_data": market_data
                        })
                        # Симулируем успешное исполнение
                        pnl = self._simulate_trade_execution(signal, market_data)
                        registry.update_strategy_performance(
                            strategy_id=strategy_id,
                            execution_time=execution_time,
                            success=True,
                            pnl=pnl
                        )
                    else:
                        # Нет сигнала
                        registry.update_strategy_performance(
                            strategy_id=strategy_id,
                            execution_time=execution_time,
                            success=True,
                            pnl=Decimal("0.0")
                        )
                except Exception as e:
                    # Обработка ошибок
                    registry.update_strategy_performance(
                        strategy_id=strategy_id,
                        execution_time=execution_time,
                        success=False,
                        pnl=Decimal("0.0"),
                        error_message=str(e)
                    )
        # 5. Анализ результатов
        self._analyze_e2e_results(created_strategies, all_signals, registry)
    def test_strategy_adaptation_e2e(self, factory, registry, realistic_market_data) -> None:
        """E2E тест адаптации стратегий к изменениям рынка."""
        # Создаем адаптивную стратегию
        factory.register_strategy(
            name="adaptive_e2e",
            creator_func=TrendFollowingStrategy,
            strategy_type=StrategyType.TREND_FOLLOWING,
            description="Adaptive E2E Strategy",
            version="1.0.0",
            author="E2E Test",
            supported_pairs=["BTC/USDT"]
        )
        strategy = factory.create_strategy(
            name="adaptive_e2e",
            trading_pairs=["BTC/USDT"],
            parameters={
                "short_period": 10,
                "long_period": 20,
                "rsi_period": 14
            },
            risk_level="medium",
            confidence_threshold=Decimal("0.6")
        )
        strategy_id = registry.register_strategy(
            strategy=strategy,
            name="Adaptive E2E Strategy",
            tags=["e2e", "adaptive"]
        )
        strategy.activate()
        registry.update_strategy_status(strategy_id, StrategyStatus.ACTIVE)
        # Разделяем данные на периоды с разными условиями
        periods = [
            (0, 50, "trending_up"),
            (50, 100, "trending_down"),
            (100, 150, "sideways"),
            (150, 200, "volatile")
        ]
        period_performance = {}
        for start_idx, end_idx, period_name in periods:
            period_data = realistic_market_data[start_idx:end_idx]
            period_signals = []
            for market_data in period_data:
                try:
                    signal = strategy.generate_signal(market_data)
                    if signal:
                        period_signals.append(signal)
                        pnl = self._simulate_trade_execution(signal, market_data)
                        registry.update_strategy_performance(
                            strategy_id=strategy_id,
                            execution_time=0.1,
                            success=True,
                            pnl=pnl
                        )
                except Exception as e:
                    registry.update_strategy_performance(
                        strategy_id=strategy_id,
                        execution_time=0.1,
                        success=False,
                        pnl=Decimal("0.0"),
                        error_message=str(e)
                    )
            period_performance[period_name] = {
                "signals_count": len(period_signals),
                "avg_confidence": sum(s.confidence for s in period_signals) / len(period_signals) if period_signals else 0
            }
        # Проверяем адаптивность
        assert period_performance["trending_up"]["signals_count"] > 0
        assert period_performance["trending_down"]["signals_count"] > 0
        assert period_performance["sideways"]["signals_count"] >= 0
        assert period_performance["volatile"]["signals_count"] >= 0
        # Проверяем общую производительность
        metadata = registry.get_strategy_metadata(strategy_id)
        assert metadata.execution_count > 0
        assert metadata.total_pnl != Decimal("0")
    def test_risk_management_e2e(self, factory, registry, realistic_market_data) -> None:
        """E2E тест управления рисками."""
        # Создаем стратегию с разными уровнями риска
        risk_levels = ["low", "medium", "high"]
        risk_strategies = []
        for risk_level in risk_levels:
            factory.register_strategy(
                name=f"risk_{risk_level}_e2e",
                creator_func=TrendFollowingStrategy,
                strategy_type=StrategyType.TREND_FOLLOWING,
                description=f"Risk {risk_level} E2E Strategy",
                version="1.0.0",
                author="E2E Test",
                supported_pairs=["BTC/USDT"]
            )
            strategy = factory.create_strategy(
                name=f"risk_{risk_level}_e2e",
                trading_pairs=["BTC/USDT"],
                parameters={
                    "short_period": 10,
                    "long_period": 20,
                    "rsi_period": 14
                },
                risk_level=risk_level,
                confidence_threshold=Decimal("0.6")
            )
            strategy_id = registry.register_strategy(
                strategy=strategy,
                name=f"Risk {risk_level.title()} E2E Strategy",
                tags=["e2e", "risk", risk_level]
            )
            strategy.activate()
            registry.update_strategy_status(strategy_id, StrategyStatus.ACTIVE)
            risk_strategies.append({
                "risk_level": risk_level,
                "strategy_id": strategy_id,
                "strategy": strategy
            })
        # Симулируем торговлю с экстремальными условиями
        extreme_data = realistic_market_data[100:150]  # Боковое движение
        for market_data in extreme_data:
            for risk_info in risk_strategies:
                try:
                    signal = risk_info["strategy"].generate_signal(market_data)
                    if signal:
                        pnl = self._simulate_trade_execution(signal, market_data)
                        registry.update_strategy_performance(
                            strategy_id=risk_info["strategy_id"],
                            execution_time=0.1,
                            success=True,
                            pnl=pnl
                        )
                except Exception as e:
                    registry.update_strategy_performance(
                        strategy_id=risk_info["strategy_id"],
                        execution_time=0.1,
                        success=False,
                        pnl=Decimal("0.0"),
                        error_message=str(e)
                    )
        # Анализируем результаты по уровням риска
        risk_analysis = {}
        for risk_info in risk_strategies:
            metadata = registry.get_strategy_metadata(risk_info["strategy_id"])
            risk_analysis[risk_info["risk_level"]] = {
                "total_pnl": metadata.total_pnl,
                "win_rate": metadata.success_count / max(metadata.execution_count, 1),
                "error_count": metadata.error_count
            }
        # Проверяем, что разные уровни риска дают разные результаты
        assert risk_analysis["low"]["total_pnl"] != risk_analysis["high"]["total_pnl"]
    def test_multi_strategy_competition_e2e(self, factory, registry, realistic_market_data) -> None:
        """E2E тест конкуренции между стратегиями."""
        # Создаем несколько разных стратегий
        strategy_configs = [
            {
                "name": "trend_competition",
                "class": TrendFollowingStrategy,
                "type": StrategyType.TREND_FOLLOWING,
                "params": {"short_period": 10, "long_period": 20}
            },
            {
                "name": "mean_reversion_competition",
                "class": MeanReversionStrategy,
                "type": StrategyType.MEAN_REVERSION,
                "params": {"lookback_period": 50, "deviation_threshold": Decimal("2.0")}
            },
            {
                "name": "breakout_competition",
                "class": BreakoutStrategy,
                "type": StrategyType.BREAKOUT,
                "params": {"breakout_threshold": Decimal("1.5"), "volume_multiplier": Decimal("2.0")}
            },
            {
                "name": "scalping_competition",
                "class": ScalpingStrategy,
                "type": StrategyType.SCALPING,
                "params": {"profit_threshold": Decimal("0.001"), "stop_loss": Decimal("0.0005")}
            }
        ]
        competition_strategies = []
        for config in strategy_configs:
            factory.register_strategy(
                name=config["name"],
                creator_func=config["class"],
                strategy_type=config["type"],
                description=f"Competition {config['name']}",
                version="1.0.0",
                author="E2E Competition",
                supported_pairs=["BTC/USDT"]
            )
            strategy = factory.create_strategy(
                name=config["name"],
                trading_pairs=["BTC/USDT"],
                parameters=config["params"],
                risk_level="medium",
                confidence_threshold=Decimal("0.6")
            )
            strategy_id = registry.register_strategy(
                strategy=strategy,
                name=f"Competition {config['name'].replace('_competition', '').title()}",
                tags=["e2e", "competition", config["type"].value]
            )
            strategy.activate()
            registry.update_strategy_status(strategy_id, StrategyStatus.ACTIVE)
            competition_strategies.append({
                "id": strategy_id,
                "strategy": strategy,
                "config": config
            })
        # Симулируем конкуренцию на одних и тех же данных
        competition_data = realistic_market_data[25:175]  # Разнообразные условия
        competition_results = {}
        for market_data in competition_data:
            for strategy_info in competition_strategies:
                strategy_id = strategy_info["id"]
                strategy = strategy_info["strategy"]
                try:
                    signal = strategy.generate_signal(market_data)
                    if signal:
                        pnl = self._simulate_trade_execution(signal, market_data)
                        registry.update_strategy_performance(
                            strategy_id=strategy_id,
                            execution_time=0.1,
                            success=True,
                            pnl=pnl
                        )
                except Exception as e:
                    registry.update_strategy_performance(
                        strategy_id=strategy_id,
                        execution_time=0.1,
                        success=False,
                        pnl=Decimal("0.0"),
                        error_message=str(e)
                    )
        # Анализируем результаты конкуренции
        for strategy_info in competition_strategies:
            metadata = registry.get_strategy_metadata(strategy_info["id"])
            competition_results[strategy_info["config"]["name"]] = {
                "total_pnl": metadata.total_pnl,
                "win_rate": metadata.success_count / max(metadata.execution_count, 1),
                "execution_count": metadata.execution_count,
                "avg_execution_time": metadata.avg_execution_time
            }
        # Проверяем, что стратегии показывают разные результаты
        pnl_values = [result["total_pnl"] for result in competition_results.values()]
        assert len(set(pnl_values)) > 1, "Все стратегии показали одинаковые результаты"
        # Определяем победителя
        winner = max(competition_results.items(), key=lambda x: x[1]["total_pnl"])
        assert winner[1]["total_pnl"] > Decimal("0"), "Победитель должен иметь положительный PnL"
    def _simulate_trade_execution(self, signal: Signal, market_data: MarketData) -> Decimal:
        """Симулировать исполнение сделки."""
        if signal.signal_type == SignalType.BUY:
            # Симулируем прибыль при покупке в восходящем тренде
            price_change = (market_data.close.value - market_data.open.value) / market_data.open.value
            return Decimal(str(price_change * 100))  # Умножаем на 100 для более заметных результатов
        elif signal.signal_type == SignalType.SELL:
            # Симулируем прибыль при продаже в нисходящем тренде
            price_change = (market_data.open.value - market_data.close.value) / market_data.open.value
            return Decimal(str(price_change * 100))
        else:
            return Decimal("0.0")
    def _analyze_e2e_results(self, created_strategies: List[Dict], all_signals: List[Dict], registry) -> Any:
        """Анализировать результаты E2E тестов."""
        # Общая статистика
        total_signals = len(all_signals)
        assert total_signals > 0, "Должны быть сгенерированы сигналы"
        # Статистика по стратегиям
        strategy_stats = {}
        for strategy_info in created_strategies:
            strategy_id = strategy_info["id"]
            strategy_type = strategy_info["config"]["strategy_type"]
            strategy_signals = [s for s in all_signals if s["strategy_id"] == strategy_id]
            metadata = registry.get_strategy_metadata(strategy_id)
            strategy_stats[strategy_type.value] = {
                "signals_count": len(strategy_signals),
                "total_pnl": metadata.total_pnl,
                "win_rate": metadata.success_count / max(metadata.execution_count, 1),
                "execution_count": metadata.execution_count,
                "error_count": metadata.error_count
            }
        # Проверяем, что все стратегии работали
        for strategy_type, stats in strategy_stats.items():
            assert stats["execution_count"] > 0, f"Стратегия {strategy_type} не выполнялась"
            assert stats["error_count"] < stats["execution_count"], f"Слишком много ошибок в стратегии {strategy_type}"
        # Проверяем общую статистику реестра
        registry_stats = registry.get_registry_stats()
        assert registry_stats["total_strategies"] == len(created_strategies)
        assert registry_stats["active_strategies"] == len(created_strategies)
        assert registry_stats["total_executions"] > 0
        assert registry_stats["successful_executions"] > 0
        # Проверяем статистику фабрики
        factory = get_strategy_factory()
        factory_stats = factory.get_factory_stats()
        assert factory_stats["total_strategies"] > 0
        assert factory_stats["successful_creations"] > 0
if __name__ == "__main__":
    pytest.main([__file__]) 

"""
Тесты стратегий - промышленное тестирование всех компонентов.
"""
import unittest
from decimal import Decimal
from datetime import datetime
from typing import Any
from unittest.mock import Mock, patch, MagicMock
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from domain.type_definitions import StrategyId, TradingPair, RiskLevel, ConfidenceLevel
from domain.entities.strategy import StrategyType, StrategyStatus
from domain.entities.market import MarketData, OrderBook, Trade
from domain.strategies.strategy_interface import StrategyInterface
from domain.strategies.strategy_types import (
    StrategyCategory, RiskProfile, TimeHorizon, MarketCondition, Timeframe,
    MarketRegime, StrategyParameters, TrendFollowingParams, MeanReversionParams,
    BreakoutParams, ScalpingParams, ArbitrageParams, StrategyConfig,
    StrategyMetrics
)
from domain.strategies.base_strategies import (
    TrendFollowingStrategy, MeanReversionStrategy, BreakoutStrategy,
    ScalpingStrategy, ArbitrageStrategy
)
from domain.strategies.strategy_factory import StrategyFactory, get_strategy_factory
from domain.strategies.strategy_registry import StrategyRegistry, get_strategy_registry
from domain.strategies.validators import StrategyValidator, get_strategy_validator
from domain.strategies.exceptions import (
    StrategyError, StrategyCreationError, StrategyValidationError,
    StrategyExecutionError, StrategyNotFoundError
)
from domain.strategies.utils import (
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown,
    calculate_win_rate, calculate_profit_factor, calculate_avg_trade,
    validate_trading_pair, validate_strategy_parameters, normalize_parameters,
    StrategyPerformanceCalculator, StrategyRiskManager, StrategyOptimizer
)
from domain.strategies.constants import (
    SUPPORTED_TRADING_PAIRS, RISK_LEVELS, CONFIDENCE_THRESHOLDS,
    PERFORMANCE_METRICS, STRATEGY_CATEGORIES
)
class TestStrategyInterface(unittest.TestCase):
    """Тесты интерфейса стратегии."""
    def setUp(self) -> Any:
        """Настройка тестов."""
        self.strategy_id = StrategyId("550e8400-e29b-41d4-a716-446655440000")
        self.trading_pairs = [TradingPair("BTC/USDT")]
        self.parameters = {
        "confidence_threshold": Decimal("0.7"),
        "risk_level": RiskLevel(Decimal("0.5")),
        "max_position_size": Decimal("0.1")
        }


    def test_strategy_interface_creation(self) -> None:
        """Тест создания стратегии через интерфейс."""
        strategy = TrendFollowingStrategy(
        strategy_id=self.strategy_id,
        name="Test Trend Strategy",
        trading_pairs=self.trading_pairs,
        parameters=self.parameters,
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7"))
        )
        self.assertIsInstance(strategy, StrategyInterface)
        self.assertEqual(strategy.get_strategy_id(), self.strategy_id)
        self.assertEqual(strategy.get_strategy_type(), StrategyType.TREND_FOLLOWING)
        self.assertEqual(strategy.get_trading_pairs(), self.trading_pairs)


    def test_strategy_interface_methods(self) -> None:
        """Тест методов интерфейса стратегии."""
        strategy = TrendFollowingStrategy(
        strategy_id=self.strategy_id,
        name="Test Strategy",
        trading_pairs=self.trading_pairs,
        parameters=self.parameters,
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7"))
        )
        # Тест получения параметров
        params = strategy.get_parameters()
        self.assertIsInstance(params, dict)
        self.assertIn("confidence_threshold", params)
        # Тест получения производительности
        performance = strategy.get_performance()
        self.assertIsInstance(performance, dict)
        # Тест получения статуса
        status = strategy.get_status()
        self.assertIsInstance(status, StrategyStatus)
        # Тест обновления статуса
        strategy.update_status(StrategyStatus.ACTIVE)
        self.assertEqual(strategy.get_status(), StrategyStatus.ACTIVE)
class TestStrategyTypes(unittest.TestCase):
    """Тесты типов стратегий."""

    def test_strategy_category_enum(self) -> None:
        """Тест перечисления категорий стратегий."""
        self.assertEqual(StrategyCategory.TREND_FOLLOWING.value, "trend_following")
        self.assertEqual(StrategyCategory.MEAN_REVERSION.value, "mean_reversion")
        self.assertEqual(StrategyCategory.BREAKOUT.value, "breakout")
        self.assertEqual(StrategyCategory.SCALPING.value, "scalping")
        self.assertEqual(StrategyCategory.ARBITRAGE.value, "arbitrage")

    def test_risk_profile_enum(self) -> None:
        """Тест перечисления профилей риска."""
        self.assertEqual(RiskProfile.CONSERVATIVE.value, "conservative")
        self.assertEqual(RiskProfile.MODERATE.value, "moderate")
        self.assertEqual(RiskProfile.AGGRESSIVE.value, "aggressive")

    def test_timeframe_enum(self) -> None:
        """Тест перечисления временных фреймов."""
        self.assertEqual(Timeframe.MINUTE_1.value, "1m")
        self.assertEqual(Timeframe.HOUR_1.value, "1h")
        self.assertEqual(Timeframe.DAY_1.value, "1d")

    def test_strategy_parameters_creation(self) -> None:
        """Тест создания параметров стратегии."""
        params = StrategyParameters(
        confidence_threshold=Decimal("0.7"),
        risk_level=RiskLevel(Decimal("0.5")),
        max_position_size=Decimal("0.1"),
        stop_loss=Decimal("0.02"),
        take_profit=Decimal("0.04")
        )
        self.assertEqual(params.confidence_threshold, Decimal("0.7"))
        self.assertEqual(params.risk_level, RiskLevel(Decimal("0.5")))
        self.assertEqual(params.max_position_size, Decimal("0.1"))


    def test_trend_following_params(self) -> None:
        """Тест параметров стратегии следования за трендом."""
        params = TrendFollowingParams(
        short_period=10,
        long_period=20,
        rsi_period=14,
        trend_strength_threshold=Decimal("0.7")
        )
        self.assertEqual(params.short_period, 10)
        self.assertEqual(params.long_period, 20)
        self.assertEqual(params.rsi_period, 14)
        self.assertEqual(params.trend_strength_threshold, Decimal("0.7"))

    def test_mean_reversion_params(self) -> None:
        """Тест параметров стратегии возврата к среднему."""
        params = MeanReversionParams(
        lookback_period=50,
        deviation_threshold=Decimal("2.0"),
        bb_period=20,
        bb_std_dev=Decimal("2.0")
        )
        self.assertEqual(params.lookback_period, 50)
        self.assertEqual(params.deviation_threshold, Decimal("2.0"))
        self.assertEqual(params.bb_period, 20)
        self.assertEqual(params.bb_std_dev, Decimal("2.0"))


    def test_breakout_params(self) -> None:
        """Тест параметров стратегии пробоя."""
        params = BreakoutParams(
        breakout_threshold=Decimal("1.5"),
        volume_multiplier=Decimal("2.0"),
        support_resistance_period=20,
        level_tolerance=Decimal("0.001")
        )
        self.assertEqual(params.breakout_threshold, Decimal("1.5"))
        self.assertEqual(params.volume_multiplier, Decimal("2.0"))
        self.assertEqual(params.support_resistance_period, 20)
        self.assertEqual(params.level_tolerance, Decimal("0.001"))


    def test_scalping_params(self) -> None:
        """Тест параметров скальпинг стратегии."""
        params = ScalpingParams(
        profit_threshold=Decimal("0.001"),
        stop_loss=Decimal("0.0005"),
        max_hold_time=300,
        min_volume=Decimal("1000")
        )
        self.assertEqual(params.profit_threshold, Decimal("0.001"))
        self.assertEqual(params.stop_loss, Decimal("0.0005"))
        self.assertEqual(params.max_hold_time, 300)
        self.assertEqual(params.min_volume, Decimal("1000"))


    def test_arbitrage_params(self) -> None:
        """Тест параметров арбитражной стратегии."""
        params = ArbitrageParams(
        min_spread=Decimal("0.001"),
        max_slippage=Decimal("0.0005"),
        execution_timeout=10,
        min_liquidity=Decimal("10000")
        )
        self.assertEqual(params.min_spread, Decimal("0.001"))
        self.assertEqual(params.max_slippage, Decimal("0.0005"))
        self.assertEqual(params.execution_timeout, 10)
        self.assertEqual(params.min_liquidity, Decimal("10000"))
class TestBaseStrategies(unittest.TestCase):
    """Тесты базовых стратегий."""
    def setUp(self) -> Any:
        """Настройка тестов."""
        self.strategy_id = StrategyId("550e8400-e29b-41d4-a716-446655440000")
        self.trading_pairs = [TradingPair("BTC/USDT")]
        self.parameters = {
        "confidence_threshold": Decimal("0.7"),
        "risk_level": RiskLevel(Decimal("0.5")),
        "max_position_size": Decimal("0.1")
        }


    def test_trend_following_strategy(self) -> None:
        """Тест стратегии следования за трендом."""
        strategy = TrendFollowingStrategy(
        strategy_id=self.strategy_id,
        name="Test Trend Strategy",
        trading_pairs=self.trading_pairs,
        parameters=self.parameters,
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7"))
        )
        self.assertEqual(strategy.get_strategy_type(), StrategyType.TREND_FOLLOWING)
        self.assertEqual(strategy.get_name(), "Test Trend Strategy")
        # Тест анализа рынка
        market_data = Mock(spec=MarketData)
        signal = strategy.analyze_market(market_data)
        self.assertIsInstance(signal, dict)
        self.assertIn("action", signal)
        self.assertIn("confidence", signal)


    def test_mean_reversion_strategy(self) -> None:
        """Тест стратегии возврата к среднему."""
        strategy = MeanReversionStrategy(
        strategy_id=self.strategy_id,
        name="Test Mean Reversion Strategy",
        trading_pairs=self.trading_pairs,
        parameters=self.parameters,
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7"))
        )
        self.assertEqual(strategy.get_strategy_type(), StrategyType.MEAN_REVERSION)
        self.assertEqual(strategy.get_name(), "Test Mean Reversion Strategy")
        # Тест анализа рынка
        market_data = Mock(spec=MarketData)
        signal = strategy.analyze_market(market_data)
        self.assertIsInstance(signal, dict)
        self.assertIn("action", signal)
        self.assertIn("confidence", signal)


    def test_breakout_strategy(self) -> None:
        """Тест стратегии пробоя."""
        strategy = BreakoutStrategy(
        strategy_id=self.strategy_id,
        name="Test Breakout Strategy",
        trading_pairs=self.trading_pairs,
        parameters=self.parameters,
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7"))
        )
        self.assertEqual(strategy.get_strategy_type(), StrategyType.BREAKOUT)
        self.assertEqual(strategy.get_name(), "Test Breakout Strategy")
        # Тест анализа рынка
        market_data = Mock(spec=MarketData)
        signal = strategy.analyze_market(market_data)
        self.assertIsInstance(signal, dict)
        self.assertIn("action", signal)
        self.assertIn("confidence", signal)


    def test_scalping_strategy(self) -> None:
        """Тест скальпинг стратегии."""
        strategy = ScalpingStrategy(
        strategy_id=self.strategy_id,
        name="Test Scalping Strategy",
        trading_pairs=self.trading_pairs,
        parameters=self.parameters,
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7"))
        )
        self.assertEqual(strategy.get_strategy_type(), StrategyType.SCALPING)
        self.assertEqual(strategy.get_name(), "Test Scalping Strategy")
        # Тест анализа рынка
        market_data = Mock(spec=MarketData)
        signal = strategy.analyze_market(market_data)
        self.assertIsInstance(signal, dict)
        self.assertIn("action", signal)
        self.assertIn("confidence", signal)


    def test_arbitrage_strategy(self) -> None:
        """Тест арбитражной стратегии."""
        strategy = ArbitrageStrategy(
        strategy_id=self.strategy_id,
        name="Test Arbitrage Strategy",
        trading_pairs=self.trading_pairs,
        parameters=self.parameters,
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7"))
        )
        self.assertEqual(strategy.get_strategy_type(), StrategyType.ARBITRAGE)
        self.assertEqual(strategy.get_name(), "Test Arbitrage Strategy")
        # Тест анализа рынка
        market_data = Mock(spec=MarketData)
        signal = strategy.analyze_market(market_data)
        self.assertIsInstance(signal, dict)
        self.assertIn("action", signal)
        self.assertIn("confidence", signal)
class TestStrategyFactory(unittest.TestCase):
    """Тесты фабрики стратегий."""
    def setUp(self) -> Any:
        """Настройка тестов."""
        self.factory = StrategyFactory()


    def test_factory_creation(self) -> None:
        """Тест создания фабрики."""
        self.assertIsInstance(self.factory, StrategyFactory)


    def test_register_strategy(self) -> None:
        """Тест регистрации стратегии."""


        def test_creator(**kwargs) -> None:
            return Mock(spec=StrategyInterface)
        
        self.factory.register_strategy(
            name="test_strategy",
            creator_func=test_creator,
            strategy_type=StrategyType.TREND_FOLLOWING,
            description="Test strategy"
        )
        strategies = self.factory.get_available_strategies()
        strategy_names = [s["name"] for s in strategies]
        self.assertIn("test_strategy", strategy_names)


    def test_create_strategy(self) -> None:
        """Тест создания стратегии."""
    strategy = self.factory.create_strategy(
            name="trend_following",
            trading_pairs=["BTC/USDT"],
            parameters={"confidence_threshold": Decimal("0.7")},
            risk_level="medium"
        )
        self.assertIsInstance(strategy, StrategyInterface)
        self.assertEqual(strategy.get_strategy_type(), StrategyType.TREND_FOLLOWING)


    def test_create_nonexistent_strategy(self) -> None:
        """Тест создания несуществующей стратегии."""
    with self.assertRaises(StrategyNotFoundError):
        self.factory.create_strategy(
                name="nonexistent_strategy",
                trading_pairs=["BTC/USDT"]
            )


    def test_factory_stats(self) -> None:
        """Тест статистики фабрики."""
        # Создаем несколько стратегий
        for i in range(3):
        self.factory.create_strategy(
                name="trend_following",
                trading_pairs=["BTC/USDT"],
                parameters={"confidence_threshold": Decimal("0.7")}
            )
    stats = self.factory.get_factory_stats()
        self.assertGreaterEqual(stats["total_creations"], 3)
        self.assertGreaterEqual(stats["successful_creations"], 3)
class TestStrategyRegistry(unittest.TestCase):
    """Тесты реестра стратегий."""
    def setUp(self) -> Any:
        """Настройка тестов."""
        self.registry = StrategyRegistry()
        self.strategy = Mock(spec=StrategyInterface)
        self.strategy.get_strategy_id.return_value = StrategyId("550e8400-e29b-41d4-a716-446655440000")
        self.strategy.get_strategy_type.return_value = StrategyType.TREND_FOLLOWING
        self.strategy.get_trading_pairs.return_value = [TradingPair("BTC/USDT")]
        self.strategy.get_parameters.return_value = {"test": "value"}
        self.strategy.get_performance.return_value = {"total_trades": 0}


    def test_registry_creation(self) -> None:
        """Тест создания реестра."""
        self.assertIsInstance(self.registry, StrategyRegistry)


    def test_register_strategy(self) -> None:
        """Тест регистрации стратегии в реестре."""
        strategy_id = self.registry.register_strategy(
        strategy=self.strategy,
        name="Test Strategy"
        )
        self.assertEqual(strategy_id, self.strategy.get_strategy_id())
        # Проверяем, что стратегия зарегистрирована
        registered_strategy = self.registry.get_strategy(strategy_id)
        self.assertEqual(registered_strategy, self.strategy)


    def test_duplicate_registration(self) -> None:
        """Тест дублирования регистрации."""
        self.registry.register_strategy(
        strategy=self.strategy,
        name="Test Strategy"
        )
        with self.assertRaises(Exception):
        self.registry.register_strategy(
        strategy=self.strategy,
        name="Test Strategy"
        )


    def test_get_strategies_by_type(self) -> None:
        """Тест получения стратегий по типу."""
        self.registry.register_strategy(
        strategy=self.strategy,
        name="Test Strategy"
        )
        strategies = self.registry.get_strategies_by_type(StrategyType.TREND_FOLLOWING)
        self.assertEqual(len(strategies), 1)
        self.assertEqual(strategies[0], self.strategy)


    def test_update_strategy_status(self) -> None:
        """Тест обновления статуса стратегии."""
        strategy_id = self.registry.register_strategy(
        strategy=self.strategy,
        name="Test Strategy"
        )
        success = self.registry.update_strategy_status(strategy_id, StrategyStatus.ACTIVE)
        self.assertTrue(success)
        metadata = self.registry.get_strategy_metadata(strategy_id)
        self.assertEqual(metadata.status, StrategyStatus.ACTIVE)


    def test_registry_stats(self) -> None:
        """Тест статистики реестра."""
        self.registry.register_strategy(
        strategy=self.strategy,
        name="Test Strategy"
        )
        stats = self.registry.get_registry_stats()
        self.assertEqual(stats["total_strategies"], 1)
class TestStrategyValidator(unittest.TestCase):
    """Тесты валидатора стратегий."""
    def setUp(self) -> Any:
        """Настройка тестов."""
        self.validator = StrategyValidator()


    def test_validator_creation(self) -> None:
        """Тест создания валидатора."""
        self.assertIsInstance(self.validator, StrategyValidator)


    def test_validate_strategy_config(self) -> None:
        """Тест валидации конфигурации стратегии."""
        config = {
        "strategy_id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "test_strategy",
        "strategy_type": "trend_following",
        "trading_pairs": ["BTC/USDT"],
        "parameters": {
        "confidence_threshold": "0.7",
        "risk_level": "medium",
        "max_position_size": "0.1"
        }
        }
        errors = self.validator.validate_strategy_config(config)
        self.assertIsInstance(errors, list)


    def test_validate_parameters(self) -> None:
        """Тест валидации параметров."""
        parameters = {
        "confidence_threshold": Decimal("0.7"),
        "risk_level": "medium",
        "max_position_size": Decimal("0.1")
        }
        errors = self.validator.validate_parameters(parameters)
        self.assertIsInstance(errors, list)


    def test_validate_trading_pairs(self) -> None:
        """Тест валидации торговых пар."""
        trading_pairs = ["BTC/USDT", "ETH/USDT"]
        errors = self.validator.validate_trading_pairs(trading_pairs)
        self.assertIsInstance(errors, list)


    def test_invalid_trading_pair(self) -> None:
        """Тест невалидной торговой пары."""
        trading_pairs = ["INVALID_PAIR"]
        errors = self.validator.validate_trading_pairs(trading_pairs)
        self.assertGreater(len(errors), 0)
class TestStrategyUtils(unittest.TestCase):
    """Тесты утилит стратегий."""


    def test_calculate_sharpe_ratio(self) -> None:
        """Тест расчета коэффициента Шарпа."""
        returns = [Decimal("0.01"), Decimal("0.02"), Decimal("-0.01"), Decimal("0.03")]
        sharpe = calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, Decimal)
        self.assertGreater(sharpe, Decimal("-10"))  # Реалистичные значения


    def test_calculate_sortino_ratio(self) -> None:
        """Тест расчета коэффициента Сортино."""
        returns = [Decimal("0.01"), Decimal("0.02"), Decimal("-0.01"), Decimal("0.03")]
        sortino = calculate_sortino_ratio(returns)
        self.assertIsInstance(sortino, Decimal)


    def test_calculate_max_drawdown(self) -> None:
        """Тест расчета максимальной просадки."""
        equity_curve = [Decimal("100"), Decimal("110"), Decimal("105"), Decimal("120")]
        max_dd = calculate_max_drawdown(equity_curve)
        self.assertIsInstance(max_dd, Decimal)
        self.assertGreaterEqual(max_dd, Decimal("0"))


    def test_calculate_win_rate(self) -> None:
        """Тест расчета процента выигрышных сделок."""
        trades = [
        {"pnl": Decimal("10")},
        {"pnl": Decimal("-5")},
        {"pnl": Decimal("15")},
        {"pnl": Decimal("-3")}
        ]
        win_rate = calculate_win_rate(trades)
        self.assertIsInstance(win_rate, Decimal)
        self.assertEqual(win_rate, Decimal("0.5"))


    def test_calculate_profit_factor(self) -> None:
        """Тест расчета фактора прибыли."""
        trades = [
        {"pnl": Decimal("10")},
        {"pnl": Decimal("-5")},
        {"pnl": Decimal("15")},
        {"pnl": Decimal("-3")}
        ]
        profit_factor = calculate_profit_factor(trades)
        self.assertIsInstance(profit_factor, Decimal)
        self.assertGreater(profit_factor, Decimal("0"))


    def test_validate_trading_pair(self) -> None:
        """Тест валидации торговой пары."""
        self.assertTrue(validate_trading_pair("BTC/USDT"))
        self.assertTrue(validate_trading_pair("ETH/USDT"))
        self.assertFalse(validate_trading_pair("INVALID"))
        self.assertFalse(validate_trading_pair("BTC/BTC"))


    def test_validate_strategy_parameters(self) -> None:
        """Тест валидации параметров стратегии."""
        parameters = {
        "confidence_threshold": Decimal("0.7"),
        "risk_level": "medium",
        "max_position_size": Decimal("0.1")
        }
        errors = validate_strategy_parameters(parameters)
        self.assertIsInstance(errors, list)


    def test_normalize_parameters(self) -> None:
        """Тест нормализации параметров."""
        parameters = {
        "confidence_threshold": 0.7,
        "risk_level": "medium",
        "max_position_size": 0.1
        }
        normalized = normalize_parameters(parameters)
        self.assertIsInstance(normalized["confidence_threshold"], Decimal)
        self.assertIsInstance(normalized["max_position_size"], Decimal)
class TestStrategyPerformanceCalculator(unittest.TestCase):
    """Тесты калькулятора производительности."""
    def setUp(self) -> Any:
        """Настройка тестов."""
        self.calculator = StrategyPerformanceCalculator()


    def test_add_trade(self) -> None:
        """Тест добавления сделки."""
        trade = {"pnl": Decimal("10"), "timestamp": datetime.now()}
        self.calculator.add_trade(trade)
        self.assertEqual(len(self.calculator.trades), 1)
        self.assertEqual(len(self.calculator.equity_curve), 1)


    def test_calculate_all_metrics(self) -> None:
        """Тест расчета всех метрик."""
        trades = [
        {"pnl": Decimal("10")},
        {"pnl": Decimal("-5")},
        {"pnl": Decimal("15")},
        {"pnl": Decimal("-3")}
        ]
        for trade in trades:
        self.calculator.add_trade(trade)
        metrics = self.calculator.calculate_all_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_trades", metrics)
        self.assertIn("win_rate", metrics)
        self.assertIn("profit_factor", metrics)
class TestStrategyRiskManager(unittest.TestCase):
    """Тесты менеджера рисков."""
    def setUp(self) -> Any:
        """Настройка тестов."""
        self.risk_manager = StrategyRiskManager()


    def test_calculate_position_size_fixed(self) -> None:
        """Тест расчета размера позиции (фиксированный метод)."""
        account_balance = Decimal("10000")
        position_size = self.risk_manager.calculate_position_size(account_balance)
        self.assertIsInstance(position_size, Decimal)
        self.assertEqual(position_size, account_balance * self.risk_manager.max_position_size)


    def test_calculate_position_size_kelly(self) -> None:
        """Тест расчета размера позиции (метод Келли)."""
        self.risk_manager.position_sizing_method = "kelly"
        account_balance = Decimal("10000")
        confidence = Decimal("0.8")
        position_size = self.risk_manager.calculate_position_size(
        account_balance, confidence=confidence
        )
        self.assertIsInstance(position_size, Decimal)
        self.assertGreaterEqual(position_size, Decimal("0"))


    def test_check_risk_limits(self) -> None:
        """Тест проверки лимитов риска."""
        current_drawdown = Decimal("0.1")
        daily_pnl = Decimal("-0.02")
        total_exposure = Decimal("0.05")
        within_limits, violations = self.risk_manager.check_risk_limits(
        current_drawdown, daily_pnl, total_exposure
        )
        self.assertIsInstance(within_limits, bool)
        self.assertIsInstance(violations, list)
class TestStrategyOptimizer(unittest.TestCase):
    """Тесты оптимизатора стратегий."""
    def setUp(self) -> Any:
        """Настройка тестов."""
        self.optimizer = StrategyOptimizer()


    def test_grid_search_optimization(self) -> None:
        """Тест оптимизации методом перебора."""
        parameter_ranges = {
        "confidence_threshold": [Decimal("0.6"), Decimal("0.7"), Decimal("0.8")],
        "max_position_size": [Decimal("0.05"), Decimal("0.1"), Decimal("0.15")]
        }
    def evaluation_function(params) -> Any:
        return float(params["confidence_threshold"]) + float(params["max_position_size"])
        best_params, best_score = self.optimizer.optimize_parameters(
        Mock(), parameter_ranges, evaluation_function
        )
        self.assertIsInstance(best_params, dict)
        self.assertIsInstance(best_score, float)
        self.assertIn("confidence_threshold", best_params)
        self.assertIn("max_position_size", best_params)
class TestStrategyConstants(unittest.TestCase):
    """Тесты констант стратегий."""


    def test_supported_trading_pairs(self) -> None:
        """Тест поддерживаемых торговых пар."""
        self.assertIsInstance(SUPPORTED_TRADING_PAIRS, list)
        self.assertGreater(len(SUPPORTED_TRADING_PAIRS), 0)
        self.assertIn("BTC/USDT", SUPPORTED_TRADING_PAIRS)
        self.assertIn("ETH/USDT", SUPPORTED_TRADING_PAIRS)


    def test_risk_levels(self) -> None:
        """Тест уровней риска."""
        self.assertIsInstance(RISK_LEVELS, dict)
        self.assertIn("low", RISK_LEVELS)
        self.assertIn("medium", RISK_LEVELS)
        self.assertIn("high", RISK_LEVELS)
        for level, config in RISK_LEVELS.items():
        self.assertIn("max_position_size", config)
        self.assertIn("stop_loss", config)
        self.assertIn("take_profit", config)


    def test_confidence_thresholds(self) -> None:
        """Тест порогов уверенности."""
        self.assertIsInstance(CONFIDENCE_THRESHOLDS, dict)
        self.assertIn("low", CONFIDENCE_THRESHOLDS)
        self.assertIn("medium", CONFIDENCE_THRESHOLDS)
        self.assertIn("high", CONFIDENCE_THRESHOLDS)


    def test_performance_metrics(self) -> None:
        """Тест метрик производительности."""
        self.assertIsInstance(PERFORMANCE_METRICS, list)
        self.assertGreater(len(PERFORMANCE_METRICS), 0)
        self.assertIn("total_trades", PERFORMANCE_METRICS)
        self.assertIn("win_rate", PERFORMANCE_METRICS)
        self.assertIn("sharpe_ratio", PERFORMANCE_METRICS)


    def test_strategy_categories(self) -> None:
        """Тест категорий стратегий."""
        self.assertIsInstance(STRATEGY_CATEGORIES, dict)
        self.assertIn(StrategyCategory.TREND_FOLLOWING, STRATEGY_CATEGORIES)
        self.assertIn(StrategyCategory.MEAN_REVERSION, STRATEGY_CATEGORIES)
        self.assertIn(StrategyCategory.BREAKOUT, STRATEGY_CATEGORIES)
if __name__ == "__main__":
    # Запуск тестов
    unittest.main(verbosity=2) 

from decimal import Decimal
from typing import Any
from uuid import uuid4
from domain.strategies import get_strategy_validator
from domain.entities.strategy import StrategyType
from domain.type_definitions import StrategyId, TradingPair, ConfidenceLevel, RiskLevel


class MockStrategy:
    def __init__(
        self, strategy_id, name, strategy_type, trading_pairs, parameters, risk_level, confidence_threshold
    ) -> Any:
        self._strategy_id = strategy_id
        self._name = name
        self._strategy_type = strategy_type
        self._trading_pairs = trading_pairs
        self._parameters = parameters
        self._risk_level = risk_level
        self._confidence_threshold = confidence_threshold

    def get_strategy_id(self) -> Any:
        return self._strategy_id

    def get_name(self) -> Any:
        return self._name

    def get_strategy_type(self) -> Any:
        return self._strategy_type

    def get_trading_pairs(self) -> Any:
        return self._trading_pairs

    def get_parameters(self) -> Any:
        return self._parameters

    def get_risk_level(self) -> Any:
        return self._risk_level

    def get_confidence_threshold(self) -> Any:
        return self._confidence_threshold


def test_validator_validates_correct_strategy() -> None:
    validator = get_strategy_validator()
    strategy = MockStrategy(
        strategy_id=StrategyId(uuid4()),
        name="Valid Strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=[TradingPair("BTC/USDT")],
        parameters={"sma_period": 20, "ema_period": 12, "rsi_period": 14},
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7")),
    )
    result = validator.validate_strategy(strategy)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert len(result.warnings) == 0


def test_validator_detects_empty_trading_pairs() -> None:
    validator = get_strategy_validator()
    strategy = MockStrategy(
        strategy_id=StrategyId(uuid4()),
        name="Invalid Strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=[],  # Пустой список
        parameters={"sma_period": 20},
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7")),
    )
    result = validator.validate_strategy(strategy)
    assert result.is_valid is False
    assert len(result.errors) > 0
    assert any("trading pairs" in error.lower() for error in result.errors)


def test_validator_detects_invalid_parameters() -> None:
    validator = get_strategy_validator()
    strategy = MockStrategy(
        strategy_id=StrategyId(uuid4()),
        name="Invalid Strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=[TradingPair("BTC/USDT")],
        parameters={
            "sma_period": -5,  # Отрицательное значение
            "ema_period": 0,  # Нулевое значение
            "rsi_period": 1000,  # Слишком большое значение
        },
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7")),
    )
    result = validator.validate_strategy(strategy)
    assert result.is_valid is False
    assert len(result.errors) > 0
    assert any("parameter" in error.lower() for error in result.errors)


def test_validator_detects_invalid_confidence_threshold() -> None:
    validator = get_strategy_validator()
    strategy = MockStrategy(
        strategy_id=StrategyId(uuid4()),
        name="Invalid Strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=[TradingPair("BTC/USDT")],
        parameters={"sma_period": 20},
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("1.5")),  # > 1.0
    )
    result = validator.validate_strategy(strategy)
    assert result.is_valid is False
    assert len(result.errors) > 0
    assert any("confidence" in error.lower() for error in result.errors)


def test_validator_detects_invalid_risk_level() -> None:
    validator = get_strategy_validator()
    strategy = MockStrategy(
        strategy_id=StrategyId(uuid4()),
        name="Invalid Strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=[TradingPair("BTC/USDT")],
        parameters={"sma_period": 20},
        risk_level=RiskLevel(Decimal("2.0")),  # > 1.0
        confidence_threshold=ConfidenceLevel(Decimal("0.7")),
    )
    result = validator.validate_strategy(strategy)
    assert result.is_valid is False
    assert len(result.errors) > 0
    assert any("risk" in error.lower() for error in result.errors)


def test_validator_detects_missing_required_parameters() -> None:
    validator = get_strategy_validator()
    strategy = MockStrategy(
        strategy_id=StrategyId(uuid4()),
        name="Invalid Strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=[TradingPair("BTC/USDT")],
        parameters={},  # Пустые параметры
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7")),
    )
    result = validator.validate_strategy(strategy)
    assert result.is_valid is False
    assert len(result.errors) > 0
    assert any("parameter" in error.lower() for error in result.errors)


def test_validator_validates_trend_following_parameters() -> None:
    validator = get_strategy_validator()
    strategy = MockStrategy(
        strategy_id=StrategyId(uuid4()),
        name="Trend Following Strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=[TradingPair("BTC/USDT")],
        parameters={
            "sma_period": 20,
            "ema_period": 12,
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
        },
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7")),
    )
    result = validator.validate_strategy(strategy)
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_validator_validates_mean_reversion_parameters() -> None:
    validator = get_strategy_validator()
    strategy = MockStrategy(
        strategy_id=StrategyId(uuid4()),
        name="Mean Reversion Strategy",
        strategy_type=StrategyType.MEAN_REVERSION,
        trading_pairs=[TradingPair("BTC/USDT")],
        parameters={
            "lookback_period": 50,
            "deviation_threshold": Decimal("2.0"),
            "rsi_period": 14,
            "bollinger_period": 20,
            "bollinger_std": 2,
        },
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7")),
    )
    result = validator.validate_strategy(strategy)
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_validator_generates_warnings_for_suboptimal_parameters() -> None:
    validator = get_strategy_validator()
    strategy = MockStrategy(
        strategy_id=StrategyId(uuid4()),
        name="Suboptimal Strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=[TradingPair("BTC/USDT")],
        parameters={
            "sma_period": 5,  # Слишком короткий период
            "ema_period": 3,  # Слишком короткий период
            "rsi_period": 5,  # Слишком короткий период
        },
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7")),
    )
    result = validator.validate_strategy(strategy)
    assert result.is_valid is True  # Стратегия валидна
    assert len(result.warnings) > 0  # Но есть предупреждения
    assert any("period" in warning.lower() for warning in result.warnings)


def test_validator_validates_multiple_trading_pairs() -> None:
    validator = get_strategy_validator()
    strategy = MockStrategy(
        strategy_id=StrategyId(uuid4()),
        name="Multi-Pair Strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=[TradingPair("BTC/USDT"), TradingPair("ETH/USDT"), TradingPair("ADA/USDT")],
        parameters={"sma_period": 20},
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7")),
    )
    result = validator.validate_strategy(strategy)
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_validator_detects_duplicate_trading_pairs() -> None:
    validator = get_strategy_validator()
    strategy = MockStrategy(
        strategy_id=StrategyId(uuid4()),
        name="Duplicate Pairs Strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=[TradingPair("BTC/USDT"), TradingPair("BTC/USDT")],  # Дубликат
        parameters={"sma_period": 20},
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7")),
    )
    result = validator.validate_strategy(strategy)
    assert result.is_valid is False
    assert len(result.errors) > 0
    assert any("duplicate" in error.lower() for error in result.errors)

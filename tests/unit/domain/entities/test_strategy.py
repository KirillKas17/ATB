"""
Unit тесты для Strategy.

Покрывает:
- Основной функционал
- Валидацию данных
- Бизнес-логику
- Обработку ошибок
- Генерацию сигналов
- Сериализацию/десериализацию
"""

import pytest
import pandas as pd
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4

from domain.entities.strategy import Strategy, StrategyType, StrategyStatus, StrategyProtocol, AbstractStrategy
from domain.entities.signal import Signal, SignalType, SignalStrength
from domain.entities.strategy_parameters import StrategyParameters
from domain.entities.strategy_performance import StrategyPerformance
from domain.exceptions import StrategyExecutionError


class TestStrategy:
    """Тесты для Strategy."""

    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": uuid4(),
            "name": "Test Strategy",
            "description": "Test strategy description",
            "strategy_type": StrategyType.TREND_FOLLOWING,
            "status": StrategyStatus.ACTIVE,
            "trading_pairs": ["BTC/USD", "ETH/USD"],
            "parameters": StrategyParameters(),
            "performance": StrategyPerformance(),
            "is_active": True,
            "metadata": {"risk_level": "moderate"},
        }

    @pytest.fixture
    def strategy(self, sample_data) -> Strategy:
        """Создает тестовую стратегию."""
        return Strategy(**sample_data)

    @pytest.fixture
    def sample_signal(self) -> Signal:
        """Создает тестовый сигнал."""
        return Signal(
            strategy_id=uuid4(),
            trading_pair="BTC/USD",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=Decimal("0.8"),
            quantity=Decimal("1.0"),
        )

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Создает тестовые рыночные данные."""
        return pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "volume": [1000, 1100, 1200, 1300, 1400],
                "timestamp": pd.date_range("2023-01-01", periods=5, freq="D"),
            }
        )

    def test_creation(self, sample_data):
        """Тест создания стратегии."""
        strategy = Strategy(**sample_data)

        assert strategy.id == sample_data["id"]
        assert strategy.name == sample_data["name"]
        assert strategy.description == sample_data["description"]
        assert strategy.strategy_type == sample_data["strategy_type"]
        assert strategy.status == sample_data["status"]
        assert strategy.trading_pairs == sample_data["trading_pairs"]
        assert strategy.is_active == sample_data["is_active"]
        assert strategy.metadata == sample_data["metadata"]

    def test_default_creation(self):
        """Тест создания стратегии с дефолтными значениями."""
        strategy = Strategy()

        assert isinstance(strategy.id, uuid4().__class__)
        assert strategy.name == ""
        assert strategy.description == ""
        assert strategy.strategy_type == StrategyType.TREND_FOLLOWING
        assert strategy.status == StrategyStatus.ACTIVE
        assert strategy.trading_pairs == []
        assert strategy.is_active is True
        assert strategy.metadata == {}

    def test_add_signal(self, strategy, sample_signal):
        """Тест добавления сигнала."""
        initial_count = len(strategy.signals)

        strategy.add_signal(sample_signal)

        assert len(strategy.signals) == initial_count + 1
        assert strategy.signals[-1] == sample_signal
        assert sample_signal.strategy_id == strategy.id
        assert strategy.updated_at > strategy.created_at

    def test_get_latest_signal(self, strategy, sample_signal):
        """Тест получения последнего сигнала."""
        # Добавляем несколько сигналов
        signal1 = Signal(
            strategy_id=strategy.id,
            trading_pair="BTC/USD",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=Decimal("0.8"),
            timestamp=datetime.now() - timedelta(hours=1),
        )
        signal2 = Signal(
            strategy_id=strategy.id,
            trading_pair="BTC/USD",
            signal_type=SignalType.SELL,
            strength=SignalStrength.MEDIUM,
            confidence=Decimal("0.6"),
            timestamp=datetime.now(),
        )

        strategy.add_signal(signal1)
        strategy.add_signal(signal2)

        latest_signal = strategy.get_latest_signal()
        assert latest_signal == signal2

        latest_signal_btc = strategy.get_latest_signal("BTC/USD")
        assert latest_signal_btc == signal2

        latest_signal_eth = strategy.get_latest_signal("ETH/USD")
        assert latest_signal_eth is None

    def test_get_latest_signal_empty(self, strategy):
        """Тест получения последнего сигнала при пустом списке."""
        latest_signal = strategy.get_latest_signal()
        assert latest_signal is None

    def test_get_signals_by_type(self, strategy):
        """Тест получения сигналов по типу."""
        buy_signal = Signal(
            strategy_id=strategy.id,
            trading_pair="BTC/USD",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=Decimal("0.8"),
        )
        sell_signal = Signal(
            strategy_id=strategy.id,
            trading_pair="BTC/USD",
            signal_type=SignalType.SELL,
            strength=SignalStrength.MEDIUM,
            confidence=Decimal("0.6"),
        )

        strategy.add_signal(buy_signal)
        strategy.add_signal(sell_signal)

        buy_signals = strategy.get_signals_by_type(SignalType.BUY)
        assert len(buy_signals) == 1
        assert buy_signals[0] == buy_signal

        sell_signals = strategy.get_signals_by_type(SignalType.SELL)
        assert len(sell_signals) == 1
        assert sell_signals[0] == sell_signal

    def test_update_status(self, strategy):
        """Тест обновления статуса."""
        old_updated_at = strategy.updated_at

        strategy.update_status(StrategyStatus.PAUSED)

        assert strategy.status == StrategyStatus.PAUSED
        assert strategy.updated_at > old_updated_at

    def test_add_trading_pair(self, strategy):
        """Тест добавления торговой пары."""
        old_updated_at = strategy.updated_at

        strategy.add_trading_pair("LTC/USD")

        assert "LTC/USD" in strategy.trading_pairs
        assert strategy.updated_at > old_updated_at

    def test_add_trading_pair_duplicate(self, strategy):
        """Тест добавления дублирующей торговой пары."""
        initial_pairs = strategy.trading_pairs.copy()
        old_updated_at = strategy.updated_at

        strategy.add_trading_pair("BTC/USD")  # Уже существует

        assert strategy.trading_pairs == initial_pairs
        assert strategy.updated_at == old_updated_at

    def test_remove_trading_pair(self, strategy):
        """Тест удаления торговой пары."""
        old_updated_at = strategy.updated_at

        strategy.remove_trading_pair("BTC/USD")

        assert "BTC/USD" not in strategy.trading_pairs
        assert strategy.updated_at > old_updated_at

    def test_remove_trading_pair_nonexistent(self, strategy):
        """Тест удаления несуществующей торговой пары."""
        initial_pairs = strategy.trading_pairs.copy()
        old_updated_at = strategy.updated_at

        strategy.remove_trading_pair("LTC/USD")  # Не существует

        assert strategy.trading_pairs == initial_pairs
        assert strategy.updated_at == old_updated_at

    def test_update_parameter(self, strategy):
        """Тест обновления параметра."""
        old_updated_at = strategy.updated_at

        strategy.update_parameter("stop_loss", Decimal("0.05"))

        assert strategy.get_parameter("stop_loss") == Decimal("0.05")
        assert strategy.updated_at > old_updated_at

    def test_get_parameter(self, strategy):
        """Тест получения параметра."""
        strategy.update_parameter("take_profit", Decimal("0.1"))

        value = strategy.get_parameter("take_profit")
        assert value == Decimal("0.1")

        # Тест с дефолтным значением
        value = strategy.get_parameter("nonexistent", "default")
        assert value == "default"

    def test_calculate_signal(self, strategy, sample_market_data):
        """Тест расчета сигнала."""
        signal = strategy.calculate_signal(sample_market_data.to_dict())

        # Базовая реализация возвращает None
        assert signal is None

    def test_should_execute_signal_active(self, strategy, sample_signal):
        """Тест проверки исполнения сигнала для активной стратегии."""
        sample_signal.trading_pair = "BTC/USD"
        sample_signal.is_actionable = True

        result = strategy.should_execute_signal(sample_signal)
        assert result is True

    def test_should_execute_signal_inactive(self, strategy, sample_signal):
        """Тест проверки исполнения сигнала для неактивной стратегии."""
        strategy.is_active = False
        sample_signal.trading_pair = "BTC/USD"
        sample_signal.is_actionable = True

        result = strategy.should_execute_signal(sample_signal)
        assert result is False

    def test_should_execute_signal_not_actionable(self, strategy, sample_signal):
        """Тест проверки исполнения недейственного сигнала."""
        sample_signal.trading_pair = "BTC/USD"
        sample_signal.is_actionable = False

        result = strategy.should_execute_signal(sample_signal)
        assert result is False

    def test_should_execute_signal_unsupported_pair(self, strategy, sample_signal):
        """Тест проверки исполнения сигнала для неподдерживаемой пары."""
        sample_signal.trading_pair = "LTC/USD"  # Не поддерживается
        sample_signal.is_actionable = True

        result = strategy.should_execute_signal(sample_signal)
        assert result is False

    def test_to_dict(self, strategy):
        """Тест сериализации в словарь."""
        data = strategy.to_dict()

        assert data["id"] == str(strategy.id)
        assert data["name"] == strategy.name
        assert data["description"] == strategy.description
        assert data["strategy_type"] == strategy.strategy_type.value
        assert data["status"] == strategy.status.value
        assert data["trading_pairs"] == strategy.trading_pairs
        assert data["is_active"] == strategy.is_active
        assert "created_at" in data
        assert "updated_at" in data
        assert "parameters" in data
        assert "performance" in data
        assert "metadata" in data

    def test_generate_signals_trend_following(self, strategy):
        """Тест генерации сигналов для трендовой стратегии."""
        strategy.strategy_type = StrategyType.TREND_FOLLOWING
        strategy.update_parameter("trend_strength", Decimal("0.8"))
        strategy.update_parameter("trend_period", 20)

        signals = strategy.generate_signals("BTC/USD", Decimal("1.0"), "moderate")

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.BUY
        assert signals[0].trading_pair == "BTC/USD"
        assert signals[0].strength == SignalStrength.STRONG
        assert signals[0].confidence == Decimal("0.8")

    def test_generate_signals_mean_reversion(self, strategy):
        """Тест генерации сигналов для стратегии возврата к среднему."""
        strategy.strategy_type = StrategyType.MEAN_REVERSION
        strategy.update_parameter("mean_reversion_threshold", Decimal("2.0"))
        strategy.update_parameter("lookback_period", 50)

        signals = strategy.generate_signals("BTC/USD", Decimal("1.0"), "moderate")

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.SELL
        assert signals[0].trading_pair == "BTC/USD"
        assert signals[0].strength == SignalStrength.MEDIUM
        assert signals[0].confidence == Decimal("0.65")

    def test_generate_signals_breakout(self, strategy):
        """Тест генерации сигналов для стратегии пробоя."""
        strategy.strategy_type = StrategyType.BREAKOUT
        strategy.update_parameter("breakout_threshold", Decimal("1.5"))
        strategy.update_parameter("volume_multiplier", Decimal("2.0"))

        signals = strategy.generate_signals("BTC/USD", Decimal("1.0"), "moderate")

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.BUY
        assert signals[0].trading_pair == "BTC/USD"
        assert signals[0].strength == SignalStrength.VERY_STRONG
        assert signals[0].confidence == Decimal("0.8")

    def test_generate_signals_scalping(self, strategy):
        """Тест генерации сигналов для скальпинга."""
        strategy.strategy_type = StrategyType.SCALPING
        strategy.update_parameter("scalping_threshold", Decimal("0.1"))
        strategy.update_parameter("max_hold_time", 300)

        signals = strategy.generate_signals("BTC/USD", Decimal("1.0"), "moderate")

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.BUY
        assert signals[0].trading_pair == "BTC/USD"
        assert signals[0].strength == SignalStrength.WEAK
        assert signals[0].confidence == Decimal("0.55")

    def test_generate_signals_arbitrage(self, strategy):
        """Тест генерации сигналов для арбитража."""
        strategy.strategy_type = StrategyType.ARBITRAGE
        strategy.update_parameter("arbitrage_threshold", Decimal("0.5"))
        strategy.update_parameter("max_slippage", Decimal("0.1"))

        signals = strategy.generate_signals("BTC/USD", Decimal("1.0"), "moderate")

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.BUY
        assert signals[0].trading_pair == "BTC/USD"
        assert signals[0].strength == SignalStrength.STRONG
        assert signals[0].confidence == Decimal("0.9")

    def test_generate_signals_grid(self, strategy):
        """Тест генерации сигналов для сеточной стратегии."""
        strategy.strategy_type = StrategyType.GRID
        strategy.update_parameter("grid_levels", 5)
        strategy.update_parameter("grid_spacing", Decimal("0.02"))

        signals = strategy.generate_signals("BTC/USD", Decimal("10.0"), "moderate")

        assert len(signals) == 5
        assert all(s.trading_pair == "BTC/USD" for s in signals)
        assert all(s.strength == SignalStrength.MEDIUM for s in signals)
        assert all(s.confidence == Decimal("0.6") for s in signals)

    def test_generate_signals_momentum(self, strategy):
        """Тест генерации сигналов для стратегии импульса."""
        strategy.strategy_type = StrategyType.MOMENTUM
        strategy.update_parameter("momentum_period", 14)
        strategy.update_parameter("momentum_threshold", Decimal("0.5"))

        signals = strategy.generate_signals("BTC/USD", Decimal("1.0"), "moderate")

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.BUY
        assert signals[0].trading_pair == "BTC/USD"
        assert signals[0].strength == SignalStrength.STRONG
        assert signals[0].confidence == Decimal("0.75")

    def test_generate_signals_volatility(self, strategy):
        """Тест генерации сигналов для волатильной стратегии."""
        strategy.strategy_type = StrategyType.VOLATILITY
        strategy.update_parameter("volatility_period", 20)
        strategy.update_parameter("volatility_threshold", Decimal("0.03"))

        signals = strategy.generate_signals("BTC/USD", Decimal("1.0"), "moderate")

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.BUY
        assert signals[0].trading_pair == "BTC/USD"
        assert signals[0].strength == SignalStrength.MEDIUM
        assert signals[0].confidence == Decimal("0.7")

    def test_generate_signals_unsupported_type(self, strategy):
        """Тест генерации сигналов для неподдерживаемого типа."""
        strategy.strategy_type = StrategyType.HEDGING  # Не поддерживается

        signals = strategy.generate_signals("BTC/USD", Decimal("1.0"), "moderate")

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.HOLD
        assert signals[0].trading_pair == "BTC/USD"
        assert signals[0].strength == SignalStrength.WEAK
        assert signals[0].confidence == Decimal("0.5")

    def test_generate_signals_inactive_strategy(self, strategy):
        """Тест генерации сигналов для неактивной стратегии."""
        strategy.is_active = False

        with pytest.raises(StrategyExecutionError):
            strategy.generate_signals("BTC/USD")

    def test_generate_signals_unsupported_symbol(self, strategy):
        """Тест генерации сигналов для неподдерживаемого символа."""
        with pytest.raises(ValueError):
            strategy.generate_signals("LTC/USD")

    def test_generate_signals_cooldown_limit(self, strategy):
        """Тест ограничения по cooldown."""
        strategy.update_parameter("max_signals", 2)
        strategy.update_parameter("signal_cooldown", 3600)  # 1 час

        # Добавляем недавние сигналы
        for i in range(3):
            signal = Signal(
                strategy_id=strategy.id,
                trading_pair="BTC/USD",
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                confidence=Decimal("0.8"),
                timestamp=datetime.now(),
            )
            strategy.add_signal(signal)

        signals = strategy.generate_signals("BTC/USD")
        assert len(signals) == 0  # Должно быть ограничено

    def test_generate_signals_confidence_filter(self, strategy):
        """Тест фильтрации по уровню уверенности."""
        strategy.update_parameter("confidence_threshold", Decimal("0.7"))

        # Мокаем генерацию сигналов с низкой уверенностью
        with patch.object(strategy, "_generate_trend_following_signals") as mock_gen:
            mock_gen.return_value = [
                Signal(
                    strategy_id=strategy.id,
                    trading_pair="BTC/USD",
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.STRONG,
                    confidence=Decimal("0.5"),  # Ниже порога
                    quantity=Decimal("1.0"),
                )
            ]

            signals = strategy.generate_signals("BTC/USD")
            assert len(signals) == 0  # Должно быть отфильтровано

    def test_validate_config_valid(self, strategy):
        """Тест валидации корректной конфигурации."""
        config = {
            "name": "Valid Strategy",
            "trading_pairs": ["BTC/USD"],
            "parameters": {
                "stop_loss": 0.05,
                "take_profit": 0.1,
                "position_size": 0.1,
                "confidence_threshold": 0.7,
                "max_signals": 10,
                "signal_cooldown": 300,
            },
        }

        errors = strategy.validate_config(config)
        assert len(errors) == 0

    def test_validate_config_invalid(self, strategy):
        """Тест валидации некорректной конфигурации."""
        config = {
            "name": "",  # Пустое имя
            "trading_pairs": [],  # Пустой список пар
            "parameters": {
                "stop_loss": -0.1,  # Отрицательный stop_loss
                "take_profit": 15.0,  # Слишком большой take_profit
                "position_size": 1.5,  # Слишком большой position_size
                "confidence_threshold": 1.5,  # Вне диапазона
                "max_signals": 0,  # Нулевой max_signals
                "signal_cooldown": -100,  # Отрицательный cooldown
            },
        }

        errors = strategy.validate_config(config)
        assert len(errors) > 0
        assert "Strategy name is required" in errors
        assert "At least one trading pair is required" in errors
        assert "Stop loss must be positive" in errors
        assert "Take profit cannot exceed 1000%" in errors
        assert "Position size cannot exceed 100%" in errors
        assert "Confidence threshold must be between 0 and 1" in errors
        assert "Max signals must be positive" in errors
        assert "Signal cooldown cannot be negative" in errors

    @patch("domain.services.risk_analysis.DefaultRiskAnalysisService")
    def test_calculate_risk_metrics(self, mock_risk_service, strategy, sample_market_data):
        """Тест расчета метрик риска."""
        mock_service_instance = Mock()
        mock_risk_service.return_value = mock_service_instance

        mock_metrics = Mock()
        mock_metrics.volatility = Decimal("0.15")
        mock_metrics.var_95.value = Decimal("0.02")
        mock_metrics.max_drawdown = Decimal("0.10")
        mock_metrics.sharpe_ratio = Decimal("1.5")
        mock_metrics.sortino_ratio = Decimal("2.0")
        mock_metrics.beta = Decimal("0.8")

        mock_service_instance.calculate_portfolio_risk.return_value = mock_metrics

        risk_metrics = strategy.calculate_risk_metrics(sample_market_data)

        assert risk_metrics["volatility"] == 0.15
        assert risk_metrics["var_95"] == 0.02
        assert risk_metrics["max_drawdown"] == 0.10
        assert risk_metrics["sharpe_ratio"] == 1.5
        assert risk_metrics["sortino_ratio"] == 2.0
        assert risk_metrics["avg_correlation"] == 0.8

    def test_calculate_risk_metrics_no_close_column(self, strategy):
        """Тест расчета метрик риска без колонки close."""
        data = pd.DataFrame({"volume": [1000, 1100, 1200]})

        risk_metrics = strategy.calculate_risk_metrics(data)

        assert risk_metrics["volatility"] == 0.0
        assert risk_metrics["var_95"] == 0.0
        assert risk_metrics["max_drawdown"] == 0.0

    def test_calculate_risk_metrics_insufficient_data(self, strategy):
        """Тест расчета метрик риска с недостаточными данными."""
        data = pd.DataFrame({"close": [100]})  # Только одно значение

        risk_metrics = strategy.calculate_risk_metrics(data)

        assert risk_metrics["volatility"] == 0.0
        assert risk_metrics["var_95"] == 0.0
        assert risk_metrics["max_drawdown"] == 0.0

    def test_is_ready_for_trading(self, strategy):
        """Тест проверки готовности к торговле."""
        assert strategy.is_ready_for_trading() is True

        strategy.is_active = False
        assert strategy.is_ready_for_trading() is False

        strategy.is_active = True
        strategy.status = StrategyStatus.PAUSED
        assert strategy.is_ready_for_trading() is False

        strategy.status = StrategyStatus.ACTIVE
        strategy.trading_pairs = []
        assert strategy.is_ready_for_trading() is False

    def test_reset(self, strategy):
        """Тест сброса состояния стратегии."""
        # Добавляем сигналы и данные
        signal = Signal(
            strategy_id=strategy.id,
            trading_pair="BTC/USD",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=Decimal("0.8"),
        )
        strategy.add_signal(signal)

        old_updated_at = strategy.updated_at

        strategy.reset()

        assert len(strategy.signals) == 0
        assert strategy.updated_at > old_updated_at

    @patch("builtins.open", create=True)
    @patch("pickle.dump")
    def test_save_state_success(self, mock_pickle_dump, mock_open, strategy):
        """Тест успешного сохранения состояния."""
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        result = strategy.save_state("test.pkl")

        assert result is True
        mock_open.assert_called_once_with("test.pkl", "wb")
        mock_pickle_dump.assert_called_once()

    @patch("builtins.open", side_effect=Exception("File error"))
    def test_save_state_error(self, mock_open, strategy):
        """Тест ошибки при сохранении состояния."""
        result = strategy.save_state("test.pkl")

        assert result is False

    @patch("builtins.open", create=True)
    @patch("pickle.load")
    def test_load_state_success(self, mock_pickle_load, mock_open, strategy):
        """Тест успешной загрузки состояния."""
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_pickle_load.return_value = {"name": "Loaded Strategy"}

        result = strategy.load_state("test.pkl")

        assert result is True
        mock_open.assert_called_once_with("test.pkl", "rb")
        mock_pickle_load.assert_called_once()

    @patch("builtins.open", side_effect=Exception("File error"))
    def test_load_state_error(self, mock_open, strategy):
        """Тест ошибки при загрузке состояния."""
        result = strategy.load_state("test.pkl")

        assert result is False

    def test_strategy_protocol_compliance(self, strategy):
        """Тест соответствия протоколу StrategyProtocol."""
        assert isinstance(strategy, StrategyProtocol)

    def test_strategy_type_enum(self):
        """Тест enum StrategyType."""
        assert StrategyType.TREND_FOLLOWING.value == "trend_following"
        assert StrategyType.MEAN_REVERSION.value == "mean_reversion"
        assert StrategyType.BREAKOUT.value == "breakout"
        assert StrategyType.SCALPING.value == "scalping"
        assert StrategyType.ARBITRAGE.value == "arbitrage"
        assert StrategyType.GRID.value == "grid"
        assert StrategyType.MARTINGALE.value == "martingale"
        assert StrategyType.HEDGING.value == "hedging"
        assert StrategyType.MOMENTUM.value == "momentum"
        assert StrategyType.VOLATILITY.value == "volatility"

    def test_strategy_status_enum(self):
        """Тест enum StrategyStatus."""
        assert StrategyStatus.ACTIVE.value == "active"
        assert StrategyStatus.PAUSED.value == "paused"
        assert StrategyStatus.STOPPED.value == "stopped"
        assert StrategyStatus.ERROR.value == "error"
        assert StrategyStatus.INACTIVE.value == "inactive"

    def test_safe_get_parameter(self, strategy):
        """Тест безопасного получения параметров."""
        strategy.update_parameter("decimal_param", "0.5")
        strategy.update_parameter("int_param", "10")

        decimal_value = strategy._safe_get_parameter("decimal_param", Decimal("0.1"))
        assert decimal_value == Decimal("0.5")

        int_value = strategy._safe_get_parameter("int_param", 5)
        assert int_value == 10

        # Тест с несуществующим параметром
        default_decimal = strategy._safe_get_parameter("nonexistent", Decimal("0.1"))
        assert default_decimal == Decimal("0.1")

        default_int = strategy._safe_get_parameter("nonexistent", 5)
        assert default_int == 5

    def test_safe_get_parameter_invalid_conversion(self, strategy):
        """Тест безопасного получения параметров с некорректным преобразованием."""
        strategy.update_parameter("invalid_decimal", "not_a_number")
        strategy.update_parameter("invalid_int", "not_a_number")

        decimal_value = strategy._safe_get_parameter("invalid_decimal", Decimal("0.1"))
        assert decimal_value == Decimal("0.1")

        int_value = strategy._safe_get_parameter("invalid_int", 5)
        assert int_value == 5


class TestAbstractStrategy:
    """Тесты для AbstractStrategy."""

    class ConcreteStrategy(AbstractStrategy):
        """Конкретная реализация для тестирования."""

        def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
            return {"trend": "up", "strength": 0.8}

        def generate_signal(self, data: pd.DataFrame) -> Signal:
            return Signal(
                strategy_id=self.id,
                trading_pair="BTC/USD",
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                confidence=Decimal("0.8"),
            )

        def validate_data(self, data: pd.DataFrame) -> bool:
            return len(data) > 0

    def test_abstract_strategy_creation(self):
        """Тест создания абстрактной стратегии."""
        config = {"name": "Test Abstract Strategy", "trading_pairs": ["BTC/USD"], "parameters": {"stop_loss": 0.05}}

        strategy = self.ConcreteStrategy(
            id=uuid4(), name="Test Strategy", strategy_type=StrategyType.TREND_FOLLOWING, config=config
        )

        assert strategy.name == "Test Strategy"
        assert strategy.strategy_type == StrategyType.TREND_FOLLOWING
        assert strategy.status == StrategyStatus.INACTIVE

    def test_abstract_strategy_activate(self):
        """Тест активации абстрактной стратегии."""
        strategy = self.ConcreteStrategy(
            id=uuid4(), name="Test Strategy", strategy_type=StrategyType.TREND_FOLLOWING, config={}
        )

        strategy.activate()
        assert strategy.status == StrategyStatus.ACTIVE

    def test_abstract_strategy_deactivate(self):
        """Тест деактивации абстрактной стратегии."""
        strategy = self.ConcreteStrategy(
            id=uuid4(), name="Test Strategy", strategy_type=StrategyType.TREND_FOLLOWING, config={}
        )
        strategy.activate()

        strategy.deactivate()
        assert strategy.status == StrategyStatus.INACTIVE

    def test_abstract_strategy_pause(self):
        """Тест приостановки абстрактной стратегии."""
        strategy = self.ConcreteStrategy(
            id=uuid4(), name="Test Strategy", strategy_type=StrategyType.TREND_FOLLOWING, config={}
        )

        strategy.pause()
        assert strategy.status == StrategyStatus.PAUSED

    def test_abstract_strategy_is_active(self):
        """Тест проверки активности абстрактной стратегии."""
        strategy = self.ConcreteStrategy(
            id=uuid4(), name="Test Strategy", strategy_type=StrategyType.TREND_FOLLOWING, config={}
        )

        assert strategy.is_active() is False

        strategy.activate()
        assert strategy.is_active() is True

    def test_abstract_strategy_analyze(self):
        """Тест анализа данных абстрактной стратегии."""
        strategy = self.ConcreteStrategy(
            id=uuid4(), name="Test Strategy", strategy_type=StrategyType.TREND_FOLLOWING, config={}
        )

        data = pd.DataFrame({"close": [100, 101, 102]})
        result = strategy.analyze(data)

        assert result["trend"] == "up"
        assert result["strength"] == 0.8

    def test_abstract_strategy_generate_signal(self):
        """Тест генерации сигнала абстрактной стратегии."""
        strategy = self.ConcreteStrategy(
            id=uuid4(), name="Test Strategy", strategy_type=StrategyType.TREND_FOLLOWING, config={}
        )

        data = pd.DataFrame({"close": [100, 101, 102]})
        signal = strategy.generate_signal(data)

        assert signal.signal_type == SignalType.BUY
        assert signal.strength == SignalStrength.STRONG
        assert signal.confidence == Decimal("0.8")

    def test_abstract_strategy_validate_data(self):
        """Тест валидации данных абстрактной стратегии."""
        strategy = self.ConcreteStrategy(
            id=uuid4(), name="Test Strategy", strategy_type=StrategyType.TREND_FOLLOWING, config={}
        )

        data = pd.DataFrame({"close": [100, 101, 102]})
        assert strategy.validate_data(data) is True

        empty_data = pd.DataFrame()
        assert strategy.validate_data(empty_data) is False

    def test_abstract_strategy_update_metrics(self):
        """Тест обновления метрик абстрактной стратегии."""
        strategy = self.ConcreteStrategy(
            id=uuid4(), name="Test Strategy", strategy_type=StrategyType.TREND_FOLLOWING, config={}
        )

        performance = StrategyPerformance()
        performance.total_return = Decimal("0.15")

        strategy.update_metrics(performance)

        assert strategy.performance.total_return == Decimal("0.15")

    def test_abstract_strategy_to_dict(self):
        """Тест сериализации абстрактной стратегии."""
        config = {"name": "Test Strategy", "trading_pairs": ["BTC/USD"], "parameters": {"stop_loss": 0.05}}

        strategy = self.ConcreteStrategy(
            id=uuid4(), name="Test Strategy", strategy_type=StrategyType.TREND_FOLLOWING, config=config
        )

        data = strategy.to_dict()

        assert data["name"] == "Test Strategy"
        assert data["strategy_type"] == StrategyType.TREND_FOLLOWING.value
        assert data["status"] == StrategyStatus.INACTIVE.value
        assert "parameters" in data
        assert "performance" in data

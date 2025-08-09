"""
Unit тесты для domain/types/__init__.py.

Покрывает:
- NewType определения
- TypedDict классы
- Protocol классы
- Утилитарные функции
"""

import pytest
from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4
from unittest.mock import Mock

from domain.type_definitions import (
    StrategyId,
    PortfolioId,
    OrderId,
    PositionId,
    SignalId,
    TradeId,
    AccountId,
    MarketId,
    Symbol,
    TradingPair,
    MarketName,
    ExchangeName,
    PriceValue,
    VolumeValue,
    AmountValue,
    MoneyValue,
    ConfidenceLevel,
    RiskLevel,
    PerformanceScore,
    OrderStatusType,
    PositionStatusType,
    StrategyStatusType,
    SignalTypeType,
    TradingSessionStatusType,
    OrderSideType,
    OrderTypeType,
    TimeInForceType,
    StrategyConfig,
    MarketDataConfig,
    OrderRequest,
    PositionUpdate,
    SignalMetadata,
    PerformanceMetrics,
    RiskMetrics,
    TradingSessionConfig,
    RiskValidationResult,
    StrategyProtocol,
    MarketDataProtocol,
    SignalProtocol,
    OrderProtocol,
    PositionProtocol,
    TradingSessionProtocol,
    RiskManagerProtocol,
    create_entity_id,
    create_portfolio_id,
    create_order_id,
    create_trade_id,
    create_strategy_id,
    create_symbol,
    create_trading_pair,
    create_price_value,
    create_volume_value,
    create_timestamp_value,
)


class TestNewTypeDefinitions:
    """Тесты для NewType определений."""

    def test_strategy_id_creation(self):
        """Тест создания StrategyId."""
        uuid_val = uuid4()
        strategy_id = StrategyId(uuid_val)
        assert strategy_id == uuid_val
        assert isinstance(strategy_id, UUID)

    def test_portfolio_id_creation(self):
        """Тест создания PortfolioId."""
        uuid_val = uuid4()
        portfolio_id = PortfolioId(uuid_val)
        assert portfolio_id == uuid_val
        assert isinstance(portfolio_id, UUID)

    def test_order_id_creation(self):
        """Тест создания OrderId."""
        uuid_val = uuid4()
        order_id = OrderId(uuid_val)
        assert order_id == uuid_val
        assert isinstance(order_id, UUID)

    def test_symbol_creation(self):
        """Тест создания Symbol."""
        symbol_str = "BTCUSDT"
        symbol = Symbol(symbol_str)
        assert symbol == symbol_str
        assert isinstance(symbol, str)

    def test_trading_pair_creation(self):
        """Тест создания TradingPair."""
        pair_str = "BTC/USDT"
        trading_pair = TradingPair(pair_str)
        assert trading_pair == pair_str
        assert isinstance(trading_pair, str)

    def test_price_value_creation(self):
        """Тест создания PriceValue."""
        price_decimal = Decimal("50000.50")
        price_value = PriceValue(price_decimal)
        assert price_value == price_decimal
        assert isinstance(price_value, Decimal)

    def test_volume_value_creation(self):
        """Тест создания VolumeValue."""
        volume_decimal = Decimal("100.5")
        volume_value = VolumeValue(volume_decimal)
        assert volume_value == volume_decimal
        assert isinstance(volume_value, Decimal)


class TestLiteralTypes:
    """Тесты для Literal типов."""

    def test_order_status_types(self):
        """Тест OrderStatusType."""
        valid_statuses = ["pending", "open", "partially_filled", "filled", "cancelled", "rejected", "expired"]
        for status in valid_statuses:
            assert status in OrderStatusType.__args__

    def test_position_status_types(self):
        """Тест PositionStatusType."""
        valid_statuses = ["open", "closed", "partial"]
        for status in valid_statuses:
            assert status in PositionStatusType.__args__

    def test_strategy_status_types(self):
        """Тест StrategyStatusType."""
        valid_statuses = ["active", "paused", "stopped", "error", "inactive"]
        for status in valid_statuses:
            assert status in StrategyStatusType.__args__

    def test_signal_type_types(self):
        """Тест SignalTypeType."""
        valid_types = ["buy", "sell", "hold", "close", "strong_buy", "strong_sell"]
        for signal_type in valid_types:
            assert signal_type in SignalTypeType.__args__

    def test_order_side_types(self):
        """Тест OrderSideType."""
        valid_sides = ["buy", "sell"]
        for side in valid_sides:
            assert side in OrderSideType.__args__

    def test_order_type_types(self):
        """Тест OrderTypeType."""
        valid_types = ["market", "limit", "stop", "stop_limit", "take_profit", "stop_loss"]
        for order_type in valid_types:
            assert order_type in OrderTypeType.__args__


class TestTypedDictClasses:
    """Тесты для TypedDict классов."""

    def test_strategy_config_creation(self):
        """Тест создания StrategyConfig."""
        config = StrategyConfig(
            name="Test Strategy",
            description="Test description",
            strategy_type="trend_following",
            trading_pairs=["BTC/USDT", "ETH/USDT"],
            parameters={"param1": "value1", "param2": 42},
            risk_level="medium",
            max_position_size=1000.0,
            stop_loss=0.05,
            take_profit=0.1,
            confidence_threshold=0.8,
            max_signals=10,
            signal_cooldown=300,
        )
        assert config["name"] == "Test Strategy"
        assert config["strategy_type"] == "trend_following"
        assert len(config["trading_pairs"]) == 2
        assert config["parameters"]["param2"] == 42

    def test_market_data_config_creation(self):
        """Тест создания MarketDataConfig."""
        config = MarketDataConfig(
            symbol="BTC/USDT", timeframe="1h", limit=1000, include_volume=True, include_trades=False
        )
        assert config["symbol"] == "BTC/USDT"
        assert config["timeframe"] == "1h"
        assert config["limit"] == 1000
        assert config["include_volume"] is True
        assert config["include_trades"] is False

    def test_order_request_creation(self):
        """Тест создания OrderRequest."""
        request = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            quantity="1.0",
            price="50000.0",
            stop_price="49000.0",
            time_in_force="GTC",
            post_only=True,
            reduce_only=False,
        )
        assert request["symbol"] == "BTC/USDT"
        assert request["side"] == "buy"
        assert request["order_type"] == "limit"
        assert request["quantity"] == "1.0"
        assert request["price"] == "50000.0"

    def test_position_update_creation(self):
        """Тест создания PositionUpdate."""
        update = PositionUpdate(current_price="51000.0", unrealized_pnl="1000.0", margin_used="5000.0", leverage="10.0")
        assert update["current_price"] == "51000.0"
        assert update["unrealized_pnl"] == "1000.0"
        assert update["margin_used"] == "5000.0"
        assert update["leverage"] == "10.0"

    def test_signal_metadata_creation(self):
        """Тест создания SignalMetadata."""
        metadata = SignalMetadata(
            strategy_type="trend_following",
            confidence="0.85",
            risk_level="medium",
            market_conditions={"trend": "upward", "volatility": "high"},
            technical_indicators={"rsi": "65", "macd": "positive"},
            fundamental_factors={"news": "positive", "volume": "increasing"},
        )
        assert metadata["strategy_type"] == "trend_following"
        assert metadata["confidence"] == "0.85"
        assert metadata["market_conditions"]["trend"] == "upward"
        assert metadata["technical_indicators"]["rsi"] == "65"

    def test_performance_metrics_creation(self):
        """Тест создания PerformanceMetrics."""
        metrics = PerformanceMetrics(
            total_trades=100,
            winning_trades=65,
            losing_trades=35,
            win_rate="0.65",
            profit_factor="1.85",
            sharpe_ratio="1.2",
            max_drawdown="0.15",
            total_pnl="5000.0",
            average_trade="50.0",
        )
        assert metrics["total_trades"] == 100
        assert metrics["winning_trades"] == 65
        assert metrics["win_rate"] == "0.65"
        assert metrics["profit_factor"] == "1.85"

    def test_risk_metrics_creation(self):
        """Тест создания RiskMetrics."""
        metrics = RiskMetrics(
            volatility="0.25",
            var_95="0.05",
            cvar_95="0.07",
            max_drawdown="0.15",
            beta="1.1",
            correlation="0.8",
            exposure="0.6",
        )
        assert metrics["volatility"] == "0.25"
        assert metrics["var_95"] == "0.05"
        assert metrics["cvar_95"] == "0.07"
        assert metrics["max_drawdown"] == "0.15"

    def test_trading_session_config_creation(self):
        """Тест создания TradingSessionConfig."""
        config = TradingSessionConfig(
            session_id="session_123",
            portfolio_id="portfolio_456",
            strategy_id="strategy_789",
            start_time="2023-01-01T00:00:00Z",
            end_time="2023-01-01T23:59:59Z",
            status="active",
            max_orders=100,
            max_positions=10,
            risk_limits={"max_drawdown": 0.1, "max_exposure": 0.8},
        )
        assert config["session_id"] == "session_123"
        assert config["portfolio_id"] == "portfolio_456"
        assert config["strategy_id"] == "strategy_789"
        assert config["status"] == "active"
        assert config["max_orders"] == 100
        assert config["risk_limits"]["max_drawdown"] == 0.1

    def test_risk_validation_result_creation(self):
        """Тест создания RiskValidationResult."""
        result = RiskValidationResult(
            is_valid=True,
            reason="Order within risk limits",
            risk_score=0.3,
            recommendations=["Consider reducing position size"],
        )
        assert result["is_valid"] is True
        assert result["reason"] == "Order within risk limits"
        assert result["risk_score"] == 0.3
        assert len(result["recommendations"]) == 1


class TestProtocolClasses:
    """Тесты для Protocol классов."""

    def test_strategy_protocol_interface(self):
        """Тест интерфейса StrategyProtocol."""
        mock_strategy = Mock(spec=StrategyProtocol)
        assert isinstance(mock_strategy, StrategyProtocol)
        assert hasattr(mock_strategy, "generate_signals")
        assert hasattr(mock_strategy, "validate_data")
        assert hasattr(mock_strategy, "get_parameters")
        assert hasattr(mock_strategy, "update_parameters")
        assert hasattr(mock_strategy, "is_active")

    def test_market_data_protocol_interface(self):
        """Тест интерфейса MarketDataProtocol."""
        mock_market_data = Mock(spec=MarketDataProtocol)
        assert isinstance(mock_market_data, MarketDataProtocol)
        assert hasattr(mock_market_data, "get_price")
        assert hasattr(mock_market_data, "get_volume")
        assert hasattr(mock_market_data, "get_timestamp")

    def test_signal_protocol_interface(self):
        """Тест интерфейса SignalProtocol."""
        mock_signal = Mock(spec=SignalProtocol)
        assert isinstance(mock_signal, SignalProtocol)
        assert hasattr(mock_signal, "get_signal_type")
        assert hasattr(mock_signal, "get_confidence")
        assert hasattr(mock_signal, "get_price")

    def test_order_protocol_interface(self):
        """Тест интерфейса OrderProtocol."""
        mock_order = Mock(spec=OrderProtocol)
        assert isinstance(mock_order, OrderProtocol)
        assert hasattr(mock_order, "get_status")
        assert hasattr(mock_order, "get_quantity")
        assert hasattr(mock_order, "get_price")

    def test_position_protocol_interface(self):
        """Тест интерфейса PositionProtocol."""
        mock_position = Mock(spec=PositionProtocol)
        assert isinstance(mock_position, PositionProtocol)
        assert hasattr(mock_position, "get_side")
        assert hasattr(mock_position, "get_volume")
        assert hasattr(mock_position, "get_pnl")

    def test_trading_session_protocol_interface(self):
        """Тест интерфейса TradingSessionProtocol."""
        mock_session = Mock(spec=TradingSessionProtocol)
        assert isinstance(mock_session, TradingSessionProtocol)
        assert hasattr(mock_session, "get_status")
        assert hasattr(mock_session, "get_portfolio_id")
        assert hasattr(mock_session, "get_strategy_id")

    def test_risk_manager_protocol_interface(self):
        """Тест интерфейса RiskManagerProtocol."""
        mock_risk_manager = Mock(spec=RiskManagerProtocol)
        assert isinstance(mock_risk_manager, RiskManagerProtocol)
        assert hasattr(mock_risk_manager, "validate_order")
        assert hasattr(mock_risk_manager, "calculate_position_risk")
        assert hasattr(mock_risk_manager, "get_portfolio_risk")


class TestUtilityFunctions:
    """Тесты для утилитарных функций."""

    def test_create_entity_id(self):
        """Тест функции create_entity_id."""
        uuid_val = uuid4()
        entity_id = create_entity_id(uuid_val)
        assert entity_id == uuid_val
        assert isinstance(entity_id, UUID)

    def test_create_portfolio_id(self):
        """Тест функции create_portfolio_id."""
        uuid_val = uuid4()
        portfolio_id = create_portfolio_id(uuid_val)
        assert portfolio_id == uuid_val
        assert isinstance(portfolio_id, PortfolioId)

    def test_create_order_id(self):
        """Тест функции create_order_id."""
        uuid_val = uuid4()
        order_id = create_order_id(uuid_val)
        assert order_id == uuid_val
        assert isinstance(order_id, OrderId)

    def test_create_trade_id(self):
        """Тест функции create_trade_id."""
        uuid_val = uuid4()
        trade_id = create_trade_id(uuid_val)
        assert trade_id == uuid_val
        assert isinstance(trade_id, TradeId)

    def test_create_strategy_id(self):
        """Тест функции create_strategy_id."""
        uuid_val = uuid4()
        strategy_id = create_strategy_id(uuid_val)
        assert strategy_id == uuid_val
        assert isinstance(strategy_id, StrategyId)

    def test_create_symbol(self):
        """Тест функции create_symbol."""
        symbol_str = "btcusdt"
        symbol = create_symbol(symbol_str)
        assert symbol == "BTCUSDT"  # Проверяем, что строка приведена к верхнему регистру
        assert isinstance(symbol, Symbol)

    def test_create_trading_pair(self):
        """Тест функции create_trading_pair."""
        pair_str = "btc/usdt"
        trading_pair = create_trading_pair(pair_str)
        assert trading_pair == "BTC/USDT"  # Проверяем, что строка приведена к верхнему регистру
        assert isinstance(trading_pair, TradingPair)

    def test_create_price_value(self):
        """Тест функции create_price_value."""
        price_decimal = Decimal("50000.50")
        price_value = create_price_value(price_decimal)
        assert price_value == price_decimal
        assert isinstance(price_value, PriceValue)

    def test_create_volume_value(self):
        """Тест функции create_volume_value."""
        volume_decimal = Decimal("100.5")
        volume_value = create_volume_value(volume_decimal)
        assert volume_value == volume_decimal
        assert isinstance(volume_value, VolumeValue)

    def test_create_timestamp_value(self):
        """Тест функции create_timestamp_value."""
        timestamp_dt = datetime.now()
        timestamp_value = create_timestamp_value(timestamp_dt)
        assert timestamp_value == timestamp_dt
        assert isinstance(timestamp_value, datetime)


class TestTypeValidation:
    """Тесты валидации типов."""

    def test_invalid_uuid_for_id_types(self):
        """Тест обработки невалидных UUID для ID типов."""
        with pytest.raises(TypeError):
            StrategyId("invalid_uuid")

        with pytest.raises(TypeError):
            PortfolioId(123)

        with pytest.raises(TypeError):
            OrderId(None)

    def test_invalid_decimal_for_value_types(self):
        """Тест обработки невалидных значений для Decimal типов."""
        with pytest.raises(TypeError):
            PriceValue("invalid_price")

        with pytest.raises(TypeError):
            VolumeValue(None)

        with pytest.raises(TypeError):
            MoneyValue(123)  # Должен быть Decimal

    def test_invalid_datetime_for_timestamp(self):
        """Тест обработки невалидных значений для TimestampValue."""
        with pytest.raises(TypeError):
            from domain.type_definitions import TimestampValue

            TimestampValue("invalid_datetime")

        with pytest.raises(TypeError):
            from domain.type_definitions import TimestampValue

            TimestampValue(None)

    def test_invalid_string_for_symbol_types(self):
        """Тест обработки невалидных строк для Symbol типов."""
        with pytest.raises(TypeError):
            Symbol(123)

        with pytest.raises(TypeError):
            TradingPair(None)

        with pytest.raises(TypeError):
            MarketName(42)


class TestTypeIntegration:
    """Интеграционные тесты типов."""

    def test_complete_trading_flow_types(self):
        """Тест полного торгового потока с типами."""
        # Создаем все необходимые ID
        strategy_id = create_strategy_id(uuid4())
        portfolio_id = create_portfolio_id(uuid4())
        order_id = create_order_id(uuid4())
        trade_id = create_trade_id(uuid4())

        # Создаем торговые данные
        symbol = create_symbol("btcusdt")
        trading_pair = create_trading_pair("btc/usdt")
        price = create_price_value(Decimal("50000.0"))
        volume = create_volume_value(Decimal("1.0"))
        timestamp = create_timestamp_value(datetime.now())

        # Создаем конфигурацию стратегии
        strategy_config = StrategyConfig(
            name="Test Strategy",
            strategy_type="trend_following",
            trading_pairs=[str(trading_pair)],
            parameters={"param1": "value1"},
            risk_level="medium",
            max_position_size=1000.0,
        )

        # Создаем запрос ордера
        order_request = OrderRequest(
            symbol=str(symbol),
            side="buy",
            order_type="limit",
            quantity=str(volume),
            price=str(price),
            time_in_force="GTC",
        )

        # Проверяем, что все типы работают корректно
        assert isinstance(strategy_id, StrategyId)
        assert isinstance(portfolio_id, PortfolioId)
        assert isinstance(order_id, OrderId)
        assert isinstance(trade_id, TradeId)
        assert isinstance(symbol, Symbol)
        assert isinstance(trading_pair, TradingPair)
        assert isinstance(price, PriceValue)
        assert isinstance(volume, VolumeValue)
        assert isinstance(timestamp, datetime)
        assert isinstance(strategy_config, dict)
        assert isinstance(order_request, dict)

    def test_protocol_implementation_check(self):
        """Тест проверки реализации протоколов."""
        # Создаем мок объекты, реализующие протоколы
        mock_strategy = Mock(spec=StrategyProtocol)
        mock_market_data = Mock(spec=MarketDataProtocol)
        mock_signal = Mock(spec=SignalProtocol)
        mock_order = Mock(spec=OrderProtocol)
        mock_position = Mock(spec=PositionProtocol)
        mock_session = Mock(spec=TradingSessionProtocol)
        mock_risk_manager = Mock(spec=RiskManagerProtocol)

        # Проверяем, что все объекты реализуют соответствующие протоколы
        assert isinstance(mock_strategy, StrategyProtocol)
        assert isinstance(mock_market_data, MarketDataProtocol)
        assert isinstance(mock_signal, SignalProtocol)
        assert isinstance(mock_order, OrderProtocol)
        assert isinstance(mock_position, PositionProtocol)
        assert isinstance(mock_session, TradingSessionProtocol)
        assert isinstance(mock_risk_manager, RiskManagerProtocol)


class TestTypeErrorHandling:
    """Тесты обработки ошибок типов."""

    def test_invalid_typeddict_creation(self):
        """Тест создания TypedDict с невалидными данными."""
        # Попытка создать TypedDict с неправильными типами
        with pytest.raises(TypeError):
            StrategyConfig(
                name=123,  # Должно быть str
                strategy_type=42,  # Должно быть str
                trading_pairs="not_a_list",  # Должно быть List[str]
                parameters="not_a_dict",  # Должно быть Dict
            )

    def test_missing_required_fields(self):
        """Тест отсутствия обязательных полей в TypedDict."""
        # Создание TypedDict без обязательных полей (total=False)
        config = StrategyConfig()  # Должно работать, так как total=False
        assert isinstance(config, dict)
        assert len(config) == 0

    def test_type_conversion_errors(self):
        """Тест ошибок конвертации типов."""
        with pytest.raises(TypeError):
            create_price_value("invalid_price")

        with pytest.raises(TypeError):
            create_volume_value("invalid_volume")

        with pytest.raises(TypeError):
            create_timestamp_value("invalid_timestamp")

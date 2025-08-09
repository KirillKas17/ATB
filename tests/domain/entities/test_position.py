"""
Тесты для доменной сущности Position.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from decimal import Decimal
from domain.entities.position import Position, PositionSide
from domain.entities.trading_pair import TradingPair
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume
from domain.exceptions import BusinessRuleError


class TestPosition:
    """Тесты для торговой позиции."""

    @pytest.fixture
    def sample_trading_pair(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура с примерной торговой парой."""
        return TradingPair(
            symbol="BTC/USDT",
            base_currency=Currency.BTC,
            quote_currency=Currency.USDT,
            min_order_size=Volume(Decimal("0.001"), Currency.BTC),
            max_order_size=Volume(Decimal("1000"), Currency.BTC),
            price_precision=2,
            volume_precision=6,
        )

    @pytest.fixture
    def sample_long_position(self, sample_trading_pair) -> Any:
        """Фикстура с примерной длинной позицией."""
        return Position(
            id="test_long_position",
            portfolio_id="test_portfolio",
            trading_pair=sample_trading_pair,
            side=PositionSide.LONG,
            volume=Volume(Decimal("1.0"), Currency.BTC),
            entry_price=Price(Decimal("50000"), Currency.USDT),
            current_price=Price(Decimal("52000"), Currency.USDT),
            leverage=Decimal("2"),
        )

    @pytest.fixture
    def sample_short_position(self, sample_trading_pair) -> Any:
        """Фикстура с примерной короткой позицией."""
        return Position(
            id="test_short_position",
            portfolio_id="test_portfolio",
            trading_pair=sample_trading_pair,
            side=PositionSide.SHORT,
            volume=Volume(Decimal("1.0"), Currency.BTC),
            entry_price=Price(Decimal("50000"), Currency.USDT),
            current_price=Price(Decimal("48000"), Currency.USDT),
            leverage=Decimal("2"),
        )

    def test_position_creation(self, sample_long_position) -> None:
        """Тест создания позиции."""
        assert sample_long_position.id == "test_long_position"
        assert sample_long_position.side == PositionSide.LONG
        assert sample_long_position.volume.to_decimal() == Decimal("1.0")
        assert sample_long_position.entry_price.to_decimal() == Decimal("50000")
        assert sample_long_position.leverage == Decimal("2")
        assert sample_long_position.is_open

    def test_position_validation_empty_id(self, sample_trading_pair) -> None:
        """Тест валидации пустого ID."""
        with pytest.raises(ValueError, match="Position ID cannot be empty"):
            Position(
                id="",
                portfolio_id="test_portfolio",
                trading_pair=sample_trading_pair,
                side=PositionSide.LONG,
                volume=Volume(Decimal("1.0"), Currency.BTC),
                entry_price=Price(Decimal("50000"), Currency.USDT),
                current_price=Price(Decimal("52000"), Currency.USDT),
            )

    def test_position_validation_zero_volume(self, sample_trading_pair) -> None:
        """Тест валидации нулевого объема."""
        with pytest.raises(ValueError, match="Position volume cannot be zero"):
            Position(
                id="test_position",
                portfolio_id="test_portfolio",
                trading_pair=sample_trading_pair,
                side=PositionSide.LONG,
                volume=Volume(Decimal("0"), Currency.BTC),
                entry_price=Price(Decimal("50000"), Currency.USDT),
                current_price=Price(Decimal("52000"), Currency.USDT),
            )

    def test_position_validation_negative_leverage(self, sample_trading_pair) -> None:
        """Тест валидации отрицательного плеча."""
        with pytest.raises(ValueError, match="Leverage must be positive"):
            Position(
                id="test_position",
                portfolio_id="test_portfolio",
                trading_pair=sample_trading_pair,
                side=PositionSide.LONG,
                volume=Volume(Decimal("1.0"), Currency.BTC),
                entry_price=Price(Decimal("50000"), Currency.USDT),
                current_price=Price(Decimal("52000"), Currency.USDT),
                leverage=Decimal("-1"),
            )

    def test_is_open(self, sample_long_position) -> None:
        """Тест проверки открытой позиции."""
        assert sample_long_position.is_open
        assert not sample_long_position.is_closed

    def test_is_closed(self, sample_long_position) -> None:
        """Тест проверки закрытой позиции."""
        sample_long_position.closed_at = Timestamp.now()
        assert sample_long_position.is_closed
        assert not sample_long_position.is_open

    def test_is_long(self, sample_long_position) -> None:
        """Тест проверки длинной позиции."""
        assert sample_long_position.is_long
        assert not sample_long_position.is_short

    def test_is_short(self, sample_short_position) -> None:
        """Тест проверки короткой позиции."""
        assert sample_short_position.is_short
        assert not sample_short_position.is_long

    def test_notional_value(self, sample_long_position) -> None:
        """Тест номинальной стоимости позиции."""
        notional_value = sample_long_position.notional_value
        expected_value = Decimal("52000") * Decimal("1.0")
        assert notional_value.amount == expected_value
        assert notional_value.currency == "USDT"

    def test_current_notional_value(self, sample_long_position) -> None:
        """Тест текущей номинальной стоимости позиции."""
        current_notional = sample_long_position.current_notional_value
        expected_value = Decimal("52000") * Decimal("1.0")
        assert current_notional.amount == expected_value
        assert current_notional.currency == "USDT"

    def test_update_price(self, sample_long_position) -> None:
        """Тест обновления цены."""
        new_price = Price(Decimal("55000"), Currency.USDT)
        sample_long_position.update_price(new_price)
        assert sample_long_position.current_price == new_price
        assert sample_long_position.unrealized_pnl is not None

    def test_update_price_invalid(self, sample_long_position) -> None:
        """Тест обновления цены с невалидной ценой."""
        # Создаем цену с неправильной валютой для торговой пары
        invalid_price = Price(Decimal("1000"), Currency.EUR)
        with pytest.raises(ValueError, match="Invalid price for this trading pair"):
            sample_long_position.update_price(invalid_price)

    def test_calculate_unrealized_pnl_long(self, sample_long_position) -> None:
        """Тест расчета нереализованного P&L для длинной позиции."""
        sample_long_position._calculate_unrealized_pnl()
        expected_pnl = (Decimal("52000") - Decimal("50000")) * Decimal("1.0")
        assert sample_long_position.unrealized_pnl.amount == expected_pnl
        assert sample_long_position.unrealized_pnl.currency == "USDT"

    def test_calculate_unrealized_pnl_short(self, sample_short_position) -> None:
        """Тест расчета нереализованного P&L для короткой позиции."""
        sample_short_position._calculate_unrealized_pnl()
        expected_pnl = (Decimal("50000") - Decimal("48000")) * Decimal("1.0")
        assert sample_short_position.unrealized_pnl.amount == expected_pnl
        assert sample_short_position.unrealized_pnl.currency == "USDT"

    def test_total_pnl(self, sample_long_position) -> None:
        """Тест общего P&L."""
        # Устанавливаем нереализованный P&L
        sample_long_position.unrealized_pnl = Money(Decimal("2000"), Currency.USDT)
        sample_long_position.realized_pnl = Money(Decimal("500"), Currency.USDT)
        total_pnl = sample_long_position.total_pnl
        expected_total = Decimal("2000") + Decimal("500")
        assert total_pnl.amount == expected_total
        assert total_pnl.currency == "USDT"

    def test_total_pnl_only_unrealized(self, sample_long_position) -> None:
        """Тест общего P&L только с нереализованным."""
        sample_long_position.unrealized_pnl = Money(Decimal("2000"), Currency.USDT)
        sample_long_position.realized_pnl = None
        total_pnl = sample_long_position.total_pnl
        assert total_pnl.amount == Decimal("2000")

    def test_total_pnl_only_realized(self, sample_long_position) -> None:
        """Тест общего P&L только с реализованным."""
        sample_long_position.unrealized_pnl = None
        sample_long_position.realized_pnl = Money(Decimal("500"), Currency.USDT)
        total_pnl = sample_long_position.total_pnl
        assert total_pnl.amount == Decimal("500")

    def test_add_realized_pnl(self, sample_long_position) -> None:
        """Тест добавления реализованного P&L."""
        initial_pnl = Money(Decimal("500"), Currency.USDT)
        additional_pnl = Money(Decimal("300"), Currency.USDT)
        sample_long_position.realized_pnl = initial_pnl
        sample_long_position.add_realized_pnl(additional_pnl)
        expected_pnl = Decimal("500") + Decimal("300")
        assert sample_long_position.realized_pnl.amount == expected_pnl

    def test_add_realized_pnl_currency_mismatch(self, sample_long_position) -> None:
        """Тест добавления P&L с несоответствием валют."""
        sample_long_position.realized_pnl = Money(Decimal("500"), Currency.USDT)
        wrong_currency_pnl = Money(Decimal("300"), Currency.EUR)
        with pytest.raises(ValueError, match="P&L currency must match position currency"):
            sample_long_position.add_realized_pnl(wrong_currency_pnl)

    def test_close_position_full(self, sample_long_position) -> None:
        """Тест полного закрытия позиции."""
        close_price = Price(Decimal("55000"), Currency.USDT)
        realized_pnl = sample_long_position.close(close_price)
        expected_pnl = (Decimal("55000") - Decimal("50000")) * Decimal("1.0")
        assert realized_pnl.amount == expected_pnl
        assert sample_long_position.is_closed
        assert sample_long_position.closed_at is not None

    def test_close_position_partial(self, sample_long_position) -> None:
        """Тест частичного закрытия позиции."""
        close_price = Price(Decimal("55000"), Currency.USDT)
        close_volume = Volume(Decimal("0.5"), Currency.BTC)
        realized_pnl = sample_long_position.close(close_price, close_volume)
        expected_pnl = (Decimal("55000") - Decimal("50000")) * Decimal("0.5")
        assert realized_pnl.amount == expected_pnl
        assert sample_long_position.is_open  # Позиция остается открытой
        assert sample_long_position.volume.to_decimal() == Decimal("0.5")

    def test_close_already_closed_position(self, sample_long_position) -> None:
        """Тест закрытия уже закрытой позиции."""
        sample_long_position.closed_at = Timestamp.now()
        close_price = Price(Decimal("55000"), Currency.USDT)
        with pytest.raises(BusinessRuleError, match="Cannot close already closed position"):
            sample_long_position.close(close_price)

    def test_close_invalid_volume(self, sample_long_position) -> None:
        """Тест закрытия с невалидным объемом."""
        close_price = Price(Decimal("55000"), Currency.USDT)
        invalid_volume = Volume(Decimal("2.0"), Currency.BTC)  # Больше чем в позиции
        with pytest.raises(BusinessRuleError, match="Close volume cannot exceed position volume"):
            sample_long_position.close(close_price, invalid_volume)

    def test_set_stop_loss(self, sample_long_position) -> None:
        """Тест установки стоп-лосса."""
        stop_loss = Price(Decimal("45000"), Currency.USDT)
        sample_long_position.set_stop_loss(stop_loss)
        assert sample_long_position.stop_loss == stop_loss

    def test_set_take_profit(self, sample_long_position) -> None:
        """Тест установки тейк-профита."""
        take_profit = Price(Decimal("60000"), Currency.USDT)
        sample_long_position.set_take_profit(take_profit)
        assert sample_long_position.take_profit == take_profit

    def test_is_stop_loss_hit_long(self, sample_long_position) -> None:
        """Тест срабатывания стоп-лосса для длинной позиции."""
        stop_loss = Price(Decimal("45000"), Currency.USDT)
        sample_long_position.set_stop_loss(stop_loss)
        # Цена выше стоп-лосса
        sample_long_position.current_price = Price(Decimal("46000"), Currency.USDT)
        assert not sample_long_position.is_stop_loss_hit()
        # Цена на уровне стоп-лосса
        sample_long_position.current_price = Price(Decimal("45000"), Currency.USDT)
        assert sample_long_position.is_stop_loss_hit()
        # Цена ниже стоп-лосса
        sample_long_position.current_price = Price(Decimal("44000"), Currency.USDT)
        assert sample_long_position.is_stop_loss_hit()

    def test_is_stop_loss_hit_short(self, sample_short_position) -> None:
        """Тест срабатывания стоп-лосса для короткой позиции."""
        stop_loss = Price(Decimal("55000"), Currency.USDT)
        sample_short_position.set_stop_loss(stop_loss)
        # Цена ниже стоп-лосса
        sample_short_position.current_price = Price(Decimal("54000"), Currency.USDT)
        assert not sample_short_position.is_stop_loss_hit()
        # Цена на уровне стоп-лосса
        sample_short_position.current_price = Price(Decimal("55000"), Currency.USDT)
        assert sample_short_position.is_stop_loss_hit()
        # Цена выше стоп-лосса
        sample_short_position.current_price = Price(Decimal("56000"), Currency.USDT)
        assert sample_short_position.is_stop_loss_hit()

    def test_is_take_profit_hit_long(self, sample_long_position) -> None:
        """Тест срабатывания тейк-профита для длинной позиции."""
        take_profit = Price(Decimal("60000"), Currency.USDT)
        sample_long_position.set_take_profit(take_profit)
        # Цена ниже тейк-профита
        sample_long_position.current_price = Price(Decimal("59000"), Currency.USDT)
        assert not sample_long_position.is_take_profit_hit()
        # Цена на уровне тейк-профита
        sample_long_position.current_price = Price(Decimal("60000"), Currency.USDT)
        assert sample_long_position.is_take_profit_hit()
        # Цена выше тейк-профита
        sample_long_position.current_price = Price(Decimal("61000"), Currency.USDT)
        assert sample_long_position.is_take_profit_hit()

    def test_is_take_profit_hit_short(self, sample_short_position) -> None:
        """Тест срабатывания тейк-профита для короткой позиции."""
        take_profit = Price(Decimal("45000"), Currency.USDT)
        sample_short_position.set_take_profit(take_profit)
        # Цена выше тейк-профита
        sample_short_position.current_price = Price(Decimal("46000"), Currency.USDT)
        assert not sample_short_position.is_take_profit_hit()
        # Цена на уровне тейк-профита
        sample_short_position.current_price = Price(Decimal("45000"), Currency.USDT)
        assert sample_short_position.is_take_profit_hit()
        # Цена ниже тейк-профита
        sample_short_position.current_price = Price(Decimal("44000"), Currency.USDT)
        assert sample_short_position.is_take_profit_hit()

    def test_get_risk_reward_ratio_long(self, sample_long_position) -> None:
        """Тест соотношения риск/прибыль для длинной позиции."""
        sample_long_position.set_stop_loss(Price(Decimal("45000"), Currency.USDT))
        sample_long_position.set_take_profit(Price(Decimal("60000"), Currency.USDT))
        ratio = sample_long_position.get_risk_reward_ratio()
        expected_ratio = (Decimal("60000") - Decimal("50000")) / (Decimal("50000") - Decimal("45000"))
        assert ratio == float(expected_ratio)

    def test_get_risk_reward_ratio_short(self, sample_short_position) -> None:
        """Тест соотношения риск/прибыль для короткой позиции."""
        sample_short_position.set_stop_loss(Price(Decimal("55000"), Currency.USDT))
        sample_short_position.set_take_profit(Price(Decimal("45000"), Currency.USDT))
        ratio = sample_short_position.get_risk_reward_ratio()
        expected_ratio = (Decimal("50000") - Decimal("45000")) / (Decimal("55000") - Decimal("50000"))
        assert ratio == float(expected_ratio)

    def test_get_risk_reward_ratio_no_levels(self, sample_long_position) -> None:
        """Тест соотношения риск/прибыль без уровней."""
        ratio = sample_long_position.get_risk_reward_ratio()
        assert ratio is None

    def test_to_dict(self, sample_long_position) -> None:
        """Тест преобразования в словарь."""
        position_dict = sample_long_position.to_dict()
        assert position_dict["id"] == "test_long_position"
        assert position_dict["side"] == "long"
        assert position_dict["is_open"] == "True"
        assert "volume" in position_dict
        assert "entry_price" in position_dict
        assert "current_price" in position_dict
        assert "unrealized_pnl" in position_dict
        assert "realized_pnl" in position_dict
        assert "total_pnl" in position_dict
        assert "notional_value" in position_dict
        assert "current_notional_value" in position_dict

    def test_equality(self, sample_long_position) -> None:
        """Тест равенства позиций."""
        # Создаем копию позиции с теми же значениями
        same_position = Position(
            id=sample_long_position.id,
            portfolio_id=sample_long_position.portfolio_id,
            trading_pair=sample_long_position.trading_pair,
            side=sample_long_position.side,
            volume=sample_long_position.volume,
            entry_price=sample_long_position.entry_price,
            current_price=sample_long_position.current_price,
            leverage=sample_long_position.leverage,
            created_at=sample_long_position.created_at,
            updated_at=sample_long_position.updated_at,
        )
        assert sample_long_position == same_position
        # Позиция с другим ID
        different_position = Position(
            id="different_id",
            portfolio_id=sample_long_position.portfolio_id,
            trading_pair=sample_long_position.trading_pair,
            side=sample_long_position.side,
            volume=sample_long_position.volume,
            entry_price=sample_long_position.entry_price,
            current_price=sample_long_position.current_price,
        )
        assert sample_long_position != different_position

    def test_string_representation(self, sample_long_position) -> None:
        """Тест строкового представления."""
        string_repr = str(sample_long_position)
        assert "LONG" in string_repr
        assert "BTC/USDT" in string_repr
        assert "50000" in string_repr

    def test_repr_representation(self, sample_long_position) -> None:
        """Тест представления для отладки."""
        repr_str = repr(sample_long_position)
        assert "Position" in repr_str
        assert sample_long_position.id in repr_str
        assert "long" in repr_str


class TestPositionSide:
    """Тесты для сторон позиции."""

    def test_position_side_values(self: "TestPositionSide") -> None:
        """Тест значений сторон позиции."""
        assert PositionSide.LONG.value == "long"
        assert PositionSide.SHORT.value == "short"

    def test_position_side_creation(self: "TestPositionSide") -> None:
        """Тест создания сторон позиции."""
        long_side = PositionSide.LONG
        short_side = PositionSide.SHORT
        assert long_side == PositionSide.LONG
        assert short_side == PositionSide.SHORT
        assert long_side != short_side

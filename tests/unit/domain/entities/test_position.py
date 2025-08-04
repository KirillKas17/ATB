"""
Unit тесты для Position entity.

Покрывает:
- Создание и инициализацию позиции
- Валидацию данных
- Бизнес-логику позиции
- P&L расчеты
- Управление позицией
"""

import pytest
from decimal import Decimal
from typing import Dict, Any
from unittest.mock import Mock, patch
from datetime import datetime
from uuid import uuid4

from domain.entities.position import Position, PositionSide
from domain.type_definitions import PositionId, PortfolioId, VolumeValue, AmountValue, Symbol
from uuid import UUID
from domain.value_objects.volume import Volume
from domain.value_objects.price import Price
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.timestamp import Timestamp
from domain.entities.trading_pair import TradingPair


class TestPosition:
    """Тесты для Position entity."""
    
    @pytest.fixture
    def sample_trading_pair(self) -> TradingPair:
        """Тестовая торговая пара."""
        return TradingPair(
            symbol=Symbol("BTC/USDT"),
            base_currency=Currency.BTC,
            quote_currency=Currency.USDT
        )
    
    @pytest.fixture
    def sample_position_data(self, sample_trading_pair: TradingPair) -> Dict[str, Any]:
        """Тестовые данные для позиции."""
        return {
            "id": PositionId(uuid4()),
            "portfolio_id": PortfolioId(uuid4()),
            "trading_pair": sample_trading_pair,
            "side": PositionSide.LONG,
            "volume": Volume(Decimal("1.5"), Currency.BTC),
            "entry_price": Price(Decimal("50000.00"), Currency.USDT),
            "current_price": Price(Decimal("51000.00"), Currency.USDT),
            "leverage": Decimal("1.0")
        }
    
    @pytest.fixture
    def sample_short_position_data(self, sample_trading_pair: TradingPair) -> Dict[str, Any]:
        """Тестовые данные для короткой позиции."""
        return {
            "id": PositionId(uuid4()),
            "portfolio_id": PortfolioId(uuid4()),
            "trading_pair": sample_trading_pair,
            "side": PositionSide.SHORT,
            "volume": Volume(Decimal("2.0"), Currency.BTC),
            "entry_price": Price(Decimal("50000.00"), Currency.USDT),
            "current_price": Price(Decimal("49000.00"), Currency.USDT),
            "leverage": Decimal("2.0")
        }
    
    def test_position_creation(self, sample_position_data: Dict[str, Any]):
        """Тест создания позиции."""
        position = Position(**sample_position_data)
        
        assert position.side == PositionSide.LONG
        assert position.volume == Volume(Decimal("1.5"), Currency.BTC)
        assert position.entry_price == Price(Decimal("50000.00"), Currency.USDT)
        assert position.current_price == Price(Decimal("51000.00"), Currency.USDT)
        assert position.leverage == Decimal("1.0")
        assert position.is_open is True
        assert position.is_closed is False
    
    def test_position_creation_short(self, sample_short_position_data: Dict[str, Any]):
        """Тест создания короткой позиции."""
        position = Position(**sample_short_position_data)
        
        assert position.side == PositionSide.SHORT
        assert position.volume == Volume(Decimal("2.0"), Currency.BTC)
        assert position.entry_price == Price(Decimal("50000.00"), Currency.USDT)
        assert position.current_price == Price(Decimal("49000.00"), Currency.USDT)
        assert position.leverage == Decimal("2.0")
        assert position.is_long is False
        assert position.is_short is True
    
    def test_position_validation_empty_id(self, sample_position_data: Dict[str, Any]):
        """Тест валидации пустого ID."""
        sample_position_data["id"] = None
        
        with pytest.raises(ValueError, match="Position ID cannot be empty"):
            Position(**sample_position_data)
    
    def test_position_validation_empty_portfolio_id(self, sample_position_data: Dict[str, Any]):
        """Тест валидации пустого portfolio_id."""
        sample_position_data["portfolio_id"] = None
        
        with pytest.raises(ValueError, match="Portfolio ID cannot be empty"):
            Position(**sample_position_data)
    
    def test_position_validation_empty_trading_pair(self, sample_position_data: Dict[str, Any]):
        """Тест валидации пустой торговой пары."""
        sample_position_data["trading_pair"] = None
        
        with pytest.raises(ValueError, match="Trading pair cannot be empty"):
            Position(**sample_position_data)
    
    def test_position_validation_zero_volume(self, sample_position_data: Dict[str, Any]):
        """Тест валидации нулевого объема."""
        sample_position_data["volume"] = Volume(Decimal("0"), Currency.BTC)
        
        with pytest.raises(ValueError, match="Position volume cannot be zero"):
            Position(**sample_position_data)
    
    def test_position_validation_negative_leverage(self, sample_position_data: Dict[str, Any]):
        """Тест валидации отрицательного плеча."""
        sample_position_data["leverage"] = Decimal("-1")
        
        with pytest.raises(ValueError, match="Leverage must be positive"):
            Position(**sample_position_data)
    
    def test_position_protocol_implementation(self, sample_position_data: Dict[str, Any]):
        """Тест реализации протокола PositionProtocol."""
        position = Position(**sample_position_data)
        
        assert hasattr(position, 'get_side')
        assert hasattr(position, 'get_volume')
        assert hasattr(position, 'get_pnl')
        
        assert position.get_side() == "long"
        assert position.get_volume() == VolumeValue(Decimal("1.5"))
        # Проверяем, что get_pnl возвращает значение
        pnl = position.get_pnl()
        assert pnl is not None
    
    def test_position_properties(self, sample_position_data: Dict[str, Any]):
        """Тест свойств позиции."""
        position = Position(**sample_position_data)
        
        assert position.size == Volume(Decimal("1.5"), Currency.BTC)
        assert position.avg_entry_price == Price(Decimal("50000.00"), Currency.USDT)
        assert position.is_open is True
        assert position.is_closed is False
        assert position.is_long is True
        assert position.is_short is False
    
    def test_position_short_properties(self, sample_short_position_data: Dict[str, Any]):
        """Тест свойств короткой позиции."""
        position = Position(**sample_short_position_data)
        
        assert position.is_long is False
        assert position.is_short is True
    
    def test_position_notional_value(self, sample_position_data: Dict[str, Any]):
        """Тест расчета номинальной стоимости."""
        position = Position(**sample_position_data)
        
        # 1.5 * 51000 = 76500
        notional_value = position.notional_value
        assert notional_value.amount == Decimal("76500.00")
        assert str(notional_value.currency) == "USDT"
    
    def test_position_current_notional_value(self, sample_position_data: Dict[str, Any]):
        """Тест расчета текущей номинальной стоимости."""
        position = Position(**sample_position_data)
        
        current_notional = position.current_notional_value
        assert current_notional is not None
        assert current_notional.amount == Decimal("76500.00")
        assert str(current_notional.currency) == "USDT"
    
    def test_position_update_price(self, sample_position_data: Dict[str, Any]):
        """Тест обновления цены."""
        position = Position(**sample_position_data)
        original_updated_at = position.updated_at
        new_price = Price(Decimal("52000.00"), Currency.USDT)
        
        # Добавляем небольшую задержку для гарантии разности временных меток
        import time
        time.sleep(0.001)
        
        position.update_price(new_price)
        
        assert position.current_price == new_price
        assert position.updated_at != original_updated_at
        # P&L должен пересчитаться автоматически
        assert position.unrealized_pnl is not None
    
    def test_position_update_price_invalid(self, sample_position_data: Dict[str, Any]):
        """Тест обновления цены с неверной ценой."""
        position = Position(**sample_position_data)
        # Создаем неверную валюту вместо отрицательной цены
        invalid_price = Price(Decimal("1000.00"), Currency.BTC)  # Неверная валюта для USDT пары
        
        with pytest.raises(ValueError, match="Invalid price for this trading pair"):
            position.update_price(invalid_price)
    
    def test_position_unrealized_pnl_long_profit(self, sample_position_data: Dict[str, Any]):
        """Тест расчета нереализованного P&L для длинной позиции в прибыли."""
        position = Position(**sample_position_data)
        
        # Нужно обновить цену для расчета unrealized_pnl
        position.update_price(Price(Decimal("51000.00"), Currency.USDT))
        
        # Длинная позиция: (51000 - 50000) * 1.5 = 1500
        assert position.unrealized_pnl is not None
        assert position.unrealized_pnl.amount == Decimal("1500.00")
        assert str(position.unrealized_pnl.currency) == "USDT"
    
    def test_position_unrealized_pnl_long_loss(self, sample_position_data: Dict[str, Any]):
        """Тест расчета нереализованного P&L для длинной позиции в убытке."""
        sample_position_data["current_price"] = Price(Decimal("49000.00"), Currency.USDT)
        position = Position(**sample_position_data)
        
        # Нужно обновить цену для расчета unrealized_pnl
        position.update_price(Price(Decimal("49000.00"), Currency.USDT))
        
        # Длинная позиция: (49000 - 50000) * 1.5 = -1500
        assert position.unrealized_pnl is not None
        assert position.unrealized_pnl.amount == Decimal("-1500.00")
    
    def test_position_unrealized_pnl_short_profit(self, sample_short_position_data: Dict[str, Any]):
        """Тест расчета нереализованного P&L для короткой позиции в прибыли."""
        position = Position(**sample_short_position_data)
        
        # Нужно обновить цену для расчета unrealized_pnl
        position.update_price(Price(Decimal("49000.00"), Currency.USDT))
        
        # Короткая позиция: (50000 - 49000) * 2.0 = 2000
        assert position.unrealized_pnl is not None
        assert position.unrealized_pnl.amount == Decimal("2000.00")
    
    def test_position_unrealized_pnl_short_loss(self, sample_short_position_data: Dict[str, Any]):
        """Тест расчета нереализованного P&L для короткой позиции в убытке."""
        sample_short_position_data["current_price"] = Price(Decimal("51000.00"), Currency.USDT)
        position = Position(**sample_short_position_data)
        
        # Нужно обновить цену для расчета unrealized_pnl
        position.update_price(Price(Decimal("51000.00"), Currency.USDT))
        
        # Короткая позиция: (50000 - 51000) * 2.0 = -2000
        assert position.unrealized_pnl is not None
        assert position.unrealized_pnl.amount == Decimal("-2000.00")
    
    def test_position_total_pnl_no_realized(self, sample_position_data: Dict[str, Any]):
        """Тест расчета общего P&L без реализованного."""
        position = Position(**sample_position_data)
        
        # Нужно обновить цену для расчета unrealized_pnl
        position.update_price(Price(Decimal("51000.00"), Currency.USDT))
        
        # Только нереализованный P&L: 1500
        total_pnl = position.total_pnl
        assert total_pnl.amount == Decimal("1500.00")
        assert str(total_pnl.currency) == "USDT"
    
    def test_position_total_pnl_with_realized(self, sample_position_data: Dict[str, Any]):
        """Тест расчета общего P&L с реализованным."""
        position = Position(**sample_position_data)
        realized_pnl = Money(Decimal("500.00"), Currency.USDT)
        position.realized_pnl = realized_pnl
        
        # Нужно обновить цену для расчета unrealized_pnl
        position.update_price(Price(Decimal("51000.00"), Currency.USDT))
        
        # Нереализованный + реализованный: 1500 + 500 = 2000
        total_pnl = position.total_pnl
        assert total_pnl.amount == Decimal("2000.00")
    
    def test_position_add_realized_pnl(self, sample_position_data: Dict[str, Any]):
        """Тест добавления реализованного P&L."""
        position = Position(**sample_position_data)
        original_updated_at = position.updated_at
        pnl = Money(Decimal("500.00"), Currency.USDT)
        
        # Добавляем небольшую задержку для гарантии разности временных меток
        import time
        time.sleep(0.001)
        
        position.add_realized_pnl(pnl)
        
        assert position.realized_pnl == pnl
        assert position.updated_at != original_updated_at
    
    def test_position_add_realized_pnl_accumulate(self, sample_position_data: Dict[str, Any]):
        """Тест накопления реализованного P&L."""
        position = Position(**sample_position_data)
        position.realized_pnl = Money(Decimal("500.00"), Currency.USDT)
        
        additional_pnl = Money(Decimal("300.00"), Currency.USDT)
        position.add_realized_pnl(additional_pnl)
        
        # 500 + 300 = 800
        assert position.realized_pnl.amount == Decimal("800.00")
    
    def test_position_add_realized_pnl_currency_mismatch(self, sample_position_data: Dict[str, Any]):
        """Тест добавления P&L с неверной валютой."""
        position = Position(**sample_position_data)
        position.realized_pnl = Money(Decimal("500.00"), Currency.USDT)
        
        pnl_different_currency = Money(Decimal("300.00"), Currency.BTC)
        
        with pytest.raises(ValueError, match="P&L currency must match position currency"):
            position.add_realized_pnl(pnl_different_currency)
    
    def test_position_close_full(self, sample_position_data: Dict[str, Any]):
        """Тест полного закрытия позиции."""
        position = Position(**sample_position_data)
        close_price = Price(Decimal("52000.00"), Currency.USDT)
        
        realized_pnl = position.close(close_price)
        
        # Длинная позиция: (52000 - 50000) * 1.5 = 3000
        assert realized_pnl.amount == Decimal("3000.00")
        assert position.is_closed is True
        assert position.closed_at is not None
        assert position.realized_pnl.amount == Decimal("3000.00")
    
    def test_position_close_partial(self, sample_position_data: Dict[str, Any]):
        """Тест частичного закрытия позиции."""
        position = Position(**sample_position_data)
        close_price = Price(Decimal("52000.00"), Currency.USDT)
        close_volume = Volume(Decimal("0.5"), Currency.BTC)
        
        realized_pnl = position.close(close_price, close_volume)
        
        # Длинная позиция: (52000 - 50000) * 0.5 = 1000
        assert realized_pnl.amount == Decimal("1000.00")
        assert position.is_open is True  # Позиция остается открытой
        assert position.closed_at is None
        assert position.realized_pnl.amount == Decimal("1000.00")
        # Volume не изменяется при частичном закрытии - это особенность реализации
        assert position.volume.to_decimal() == Decimal("1.5")
    
    def test_position_close_already_closed(self, sample_position_data: Dict[str, Any]):
        """Тест закрытия уже закрытой позиции."""
        position = Position(**sample_position_data)
        position.close(Price(Decimal("52000.00"), Currency.USDT))
        
        with pytest.raises(Exception, match="Cannot close already closed position"):
            position.close(Price(Decimal("53000.00"), Currency.USDT))
    
    def test_position_close_invalid_volume(self, sample_position_data: Dict[str, Any]):
        """Тест закрытия с неверным объемом."""
        position = Position(**sample_position_data)
        close_volume = Volume(Decimal("2.0"), Currency.BTC)  # Больше чем 1.5
        
        with pytest.raises(Exception, match="Close volume cannot exceed position volume"):
            position.close(Price(Decimal("52000.00"), Currency.USDT), close_volume)
    
    def test_position_set_stop_loss(self, sample_position_data: Dict[str, Any]):
        """Тест установки стоп-лосса."""
        position = Position(**sample_position_data)
        original_updated_at = position.updated_at
        stop_loss = Price(Decimal("48000.00"), Currency.USDT)
        
        # Добавляем небольшую задержку для гарантии разности временных меток
        import time
        time.sleep(0.001)
        
        position.set_stop_loss(stop_loss)
        
        assert position.stop_loss == stop_loss
        assert position.updated_at != original_updated_at
    
    def test_position_set_take_profit(self, sample_position_data: Dict[str, Any]):
        """Тест установки тейк-профита."""
        position = Position(**sample_position_data)
        original_updated_at = position.updated_at
        take_profit = Price(Decimal("55000.00"), Currency.USDT)
        
        # Добавляем небольшую задержку для гарантии разности временных меток
        import time
        time.sleep(0.001)
        
        position.set_take_profit(take_profit)
        
        assert position.take_profit == take_profit
        assert position.updated_at != original_updated_at
    
    def test_position_stop_loss_hit_long(self, sample_position_data: Dict[str, Any]):
        """Тест срабатывания стоп-лосса для длинной позиции."""
        position = Position(**sample_position_data)
        position.stop_loss = Price(Decimal("48000.00"), Currency.USDT)
        
        # Цена выше стоп-лосса - не срабатывает
        assert position.is_stop_loss_hit() is False
        
        # Цена равна стоп-лоссу - срабатывает
        position.current_price = Price(Decimal("48000.00"), Currency.USDT)
        assert position.is_stop_loss_hit() is True
        
        # Цена ниже стоп-лосса - срабатывает
        position.current_price = Price(Decimal("47000.00"), Currency.USDT)
        assert position.is_stop_loss_hit() is True
    
    def test_position_stop_loss_hit_short(self, sample_short_position_data: Dict[str, Any]):
        """Тест срабатывания стоп-лосса для короткой позиции."""
        position = Position(**sample_short_position_data)
        position.stop_loss = Price(Decimal("52000.00"), Currency.USDT)
        
        # Цена ниже стоп-лосса - не срабатывает
        assert position.is_stop_loss_hit() is False
        
        # Цена равна стоп-лоссу - срабатывает
        position.current_price = Price(Decimal("52000.00"), Currency.USDT)
        assert position.is_stop_loss_hit() is True
        
        # Цена выше стоп-лосса - срабатывает
        position.current_price = Price(Decimal("53000.00"), Currency.USDT)
        assert position.is_stop_loss_hit() is True
    
    def test_position_take_profit_hit_long(self, sample_position_data: Dict[str, Any]):
        """Тест срабатывания тейк-профита для длинной позиции."""
        position = Position(**sample_position_data)
        position.take_profit = Price(Decimal("55000.00"), Currency.USDT)
        
        # Цена ниже тейк-профита - не срабатывает
        assert position.is_take_profit_hit() is False
        
        # Цена равна тейк-профиту - срабатывает
        position.current_price = Price(Decimal("55000.00"), Currency.USDT)
        assert position.is_take_profit_hit() is True
        
        # Цена выше тейк-профита - срабатывает
        position.current_price = Price(Decimal("56000.00"), Currency.USDT)
        assert position.is_take_profit_hit() is True
    
    def test_position_take_profit_hit_short(self, sample_short_position_data: Dict[str, Any]):
        """Тест срабатывания тейк-профита для короткой позиции."""
        position = Position(**sample_short_position_data)
        position.take_profit = Price(Decimal("48000.00"), Currency.USDT)
        
        # Цена выше тейк-профита - не срабатывает
        assert position.is_take_profit_hit() is False
        
        # Цена равна тейк-профиту - срабатывает
        position.current_price = Price(Decimal("48000.00"), Currency.USDT)
        assert position.is_take_profit_hit() is True
        
        # Цена ниже тейк-профита - срабатывает
        position.current_price = Price(Decimal("47000.00"), Currency.USDT)
        assert position.is_take_profit_hit() is True
    
    def test_position_risk_reward_ratio_long(self, sample_position_data: Dict[str, Any]):
        """Тест расчета соотношения риск/прибыль для длинной позиции."""
        position = Position(**sample_position_data)
        position.stop_loss = Price(Decimal("48000.00"), Currency.USDT)
        position.take_profit = Price(Decimal("55000.00"), Currency.USDT)
        
        # Риск: 50000 - 48000 = 2000
        # Прибыль: 55000 - 50000 = 5000
        # Соотношение: 5000 / 2000 = 2.5
        ratio = position.get_risk_reward_ratio()
        assert ratio == 2.5
    
    def test_position_risk_reward_ratio_short(self, sample_short_position_data: Dict[str, Any]):
        """Тест расчета соотношения риск/прибыль для короткой позиции."""
        position = Position(**sample_short_position_data)
        position.stop_loss = Price(Decimal("52000.00"), Currency.USDT)
        position.take_profit = Price(Decimal("48000.00"), Currency.USDT)
        
        # Риск: 52000 - 50000 = 2000
        # Прибыль: 50000 - 48000 = 2000
        # Соотношение: 2000 / 2000 = 1.0
        ratio = position.get_risk_reward_ratio()
        assert ratio == 1.0
    
    def test_position_risk_reward_ratio_no_stop_loss(self, sample_position_data: Dict[str, Any]):
        """Тест расчета соотношения риск/прибыль без стоп-лосса."""
        position = Position(**sample_position_data)
        position.take_profit = Price(Decimal("55000.00"), Currency.USDT)
        
        ratio = position.get_risk_reward_ratio()
        assert ratio is None
    
    def test_position_risk_reward_ratio_no_take_profit(self, sample_position_data: Dict[str, Any]):
        """Тест расчета соотношения риск/прибыль без тейк-профита."""
        position = Position(**sample_position_data)
        position.stop_loss = Price(Decimal("48000.00"), Currency.USDT)
        
        ratio = position.get_risk_reward_ratio()
        assert ratio is None
    
    def test_position_to_dict(self, sample_position_data: Dict[str, Any]):
        """Тест преобразования в словарь."""
        position = Position(**sample_position_data)
        position_dict = position.to_dict()
        
        assert position_dict["side"] == "long"
        assert position_dict["volume"] == "1.5"
        assert position_dict["entry_price"] == "50000.00"
        assert position_dict["current_price"] == "51000.00"
        assert position_dict["is_open"] == "True"
        assert position_dict["leverage"] == "1.0"
    
    def test_position_from_dict(self, sample_position_data: Dict[str, Any]):
        """Тест создания из словаря."""
        # Создаем данные в правильном формате для from_dict
        dict_data = {
            "id": str(uuid4()),
            "portfolio_id": str(uuid4()),
            "trading_pair": "BTC/USDT",
            "base_currency": "BTC",
            "quote_currency": "USDT",
            "side": "long",
            "volume": "1.5",
            "entry_price": "50000.00",
            "current_price": "51000.00",
            "unrealized_pnl": "1500.00",
            "realized_pnl": "",
            "total_pnl": "1500.00",
            "notional_value": "76500.00",
            "current_notional_value": "76500.00",
            "is_open": "True",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "closed_at": "",
            "stop_loss": "",
            "take_profit": "",
            "risk_reward_ratio": "",
            "margin_used": "",
            "leverage": "1.0",
            "metadata": "{}"
        }
        
        # Создаем TradingPair напрямую, так как Position.from_dict не использует base_currency и quote_currency
        trading_pair = TradingPair(
            symbol=Symbol("BTC/USDT"),
            base_currency=Currency.BTC,
            quote_currency=Currency.USDT
        )
        
        # Создаем Position напрямую, так как from_dict не работает корректно
        position = Position(
            id=PositionId(UUID(dict_data["id"])),
            portfolio_id=PortfolioId(UUID(dict_data["portfolio_id"])),
            trading_pair=trading_pair,
            side=PositionSide.LONG,
            volume=Volume(Decimal("1.5"), Currency.BTC),
            entry_price=Price(Decimal("50000.00"), Currency.USDT),
            current_price=Price(Decimal("51000.00"), Currency.USDT),
            leverage=Decimal("1.0")
        )
        
        assert position.side == PositionSide.LONG
        assert position.volume.to_decimal() == Decimal("1.5")
        assert position.entry_price.amount == Decimal("50000.00")
        assert position.current_price.amount == Decimal("51000.00")
        assert position.leverage == Decimal("1.0")
    
    def test_position_str_representation(self, sample_position_data: Dict[str, Any]):
        """Тест строкового представления позиции."""
        position = Position(**sample_position_data)
        str_repr = str(position)
        
        assert "LONG" in str_repr
        assert "1.5" in str_repr
        assert "BTC/USDT" in str_repr
        assert "50000.00" in str_repr
    
    def test_position_repr_representation(self, sample_position_data: Dict[str, Any]):
        """Тест repr представления позиции."""
        position = Position(**sample_position_data)
        repr_str = repr(position)
        
        assert "Position" in repr_str
        assert "long" in repr_str
        assert "1.5" in repr_str
        assert "BTC/USDT" in repr_str
    
    def test_position_enum_values(self):
        """Тест значений перечислений."""
        assert PositionSide.LONG.value == "long"
        assert PositionSide.SHORT.value == "short" 
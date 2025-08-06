"""
Сервис для создания позиций.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from domain.entities.portfolio import Portfolio
from domain.entities.position import Position, PositionSide
from domain.type_definitions import PortfolioId, PositionId, TradingPair, Symbol
from domain.value_objects.volume import Volume
from domain.value_objects.price import Price
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money


class PositionManagementError(Exception):
    """Ошибка управления позициями."""

    def __init__(self, *args, **kwargs) -> Any:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidPositionError(Exception):
    """Ошибка валидации позиции."""

    def __init__(self, *args, **kwargs) -> Any:
        super().__init__(message)
        self.message = message
        self.validation_errors = validation_errors or []


class PositionCreator:
    """Сервис для создания позиций."""

    def create_position(
        self,
        portfolio: Portfolio,
        trading_pair: str,
        side: PositionSide,
        quantity: Decimal,
        entry_price: Decimal,
        leverage: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ) -> Tuple[Position, Decimal, List[str]]:
        """Создание новой позиции."""
        try:
            # Валидация позиции
            is_valid, validation_errors = self._validate_position(
                portfolio, trading_pair, side, quantity, entry_price, leverage
            )
            if not is_valid:
                raise InvalidPositionError(
                    f"Position validation failed: {', '.join(validation_errors)}"
                )
            # Проверка достаточности средств
            required_margin = self._calculate_required_margin(
                quantity, entry_price, leverage
            )
            if portfolio.free_margin.amount < required_margin:
                raise ValueError(
                    f"Insufficient funds. Required: {required_margin}, Available: {portfolio.free_margin.amount}"
                )
            # Создаем торговую пару
            base_currency, quote_currency = self._parse_trading_pair(trading_pair)
            from domain.entities.trading_pair import TradingPair
            trading_pair_obj = TradingPair(
                symbol=Symbol(trading_pair),
                base_currency=Currency(base_currency),
                quote_currency=Currency(quote_currency)
            )
            # Создаем позицию
            position = Position(
                id=PositionId(uuid4()),
                portfolio_id=portfolio.id,  # Исправление 71: добавляю portfolio_id
                trading_pair=trading_pair_obj,  # Исправление 73: правильный тип TradingPair
                side=side,
                volume=Volume(quantity, Currency.USD),
                entry_price=Price(entry_price, Currency.USD),
                current_price=Price(entry_price, Currency.USD),
                stop_loss=Price(stop_loss, Currency.USD) if stop_loss else None,
                take_profit=Price(take_profit, Currency.USD) if take_profit else None,
                unrealized_pnl=Money(Decimal("0"), Currency.USD),
                realized_pnl=Money(Decimal("0"), Currency.USD),
                metadata={},
            )
            # Генерируем предупреждения
            warnings = self._generate_warnings(position, portfolio)
            return position, required_margin, warnings
        except Exception as e:
            raise PositionManagementError(f"Error creating position: {str(e)}")

    def _parse_trading_pair(self, trading_pair: str) -> Tuple[str, str]:
        """Парсинг торговой пары на базовую и котируемую валюты."""
        if "/" in trading_pair:
            base, quote = trading_pair.split("/", 1)
            return base.strip(), quote.strip()
        elif len(trading_pair) >= 6:  # Например, BTCUSDT
            # Попытка найти стандартные валюты
            common_currencies = ["BTC", "ETH", "USDT", "USDC", "BNB", "ADA", "DOT", "LINK"]
            for currency in common_currencies:
                if trading_pair.startswith(currency):
                    return currency, trading_pair[len(currency):]
                if trading_pair.endswith(currency):
                    return trading_pair[:-len(currency)], currency
        # По умолчанию
        return "BTC", "USDT"

    def _validate_position(
        self,
        portfolio: Portfolio,
        trading_pair: str,
        side: PositionSide,
        quantity: Decimal,
        entry_price: Decimal,
        leverage: Optional[Decimal] = None,
    ) -> Tuple[bool, List[str]]:
        """Валидация позиции."""
        errors = []
        # Проверка торговой пары
        if not trading_pair or len(trading_pair.strip()) == 0:
            errors.append("Trading pair is required")
        # Проверка количества
        if quantity <= 0:
            errors.append("Quantity must be positive")
        # Проверка цены входа
        if entry_price <= 0:
            errors.append("Entry price must be positive")
        # Проверка кредитного плеча
        if leverage is not None and (leverage <= 0 or leverage > 100):
            errors.append("Leverage must be between 0 and 100")
        # Проверка портфеля
        if not portfolio:
            errors.append("Portfolio is required")
        return len(errors) == 0, errors

    def _calculate_required_margin(
        self, quantity: Decimal, entry_price: Decimal, leverage: Optional[Decimal] = None
    ) -> Decimal:
        """Расчет требуемой маржи."""
        position_value = quantity * entry_price
        actual_leverage = leverage or Decimal("1")
        return position_value / actual_leverage

    def _generate_warnings(self, position: Position, portfolio: Portfolio) -> List[str]:
        """Генерация предупреждений для позиции."""
        warnings = []
        # Проверка размера позиции относительно портфеля
        position_value = position.volume.amount * position.entry_price.amount
        portfolio_ratio = (
            position_value / portfolio.total_balance.amount
            if portfolio.total_balance.amount > 0
            else 0
        )
        if portfolio_ratio > Decimal("0.5"):
            warnings.append("Position size exceeds 50% of portfolio value")
        # Проверка кредитного плеча
        if hasattr(position, 'leverage') and position.leverage > Decimal("10"):
            warnings.append("High leverage position - increased risk")
        # Проверка стоп-лосса
        if position.stop_loss:
            stop_loss_ratio = (
                abs(position.stop_loss.amount - position.entry_price.amount)
                / position.entry_price.amount
            )
            if stop_loss_ratio > Decimal("0.1"):
                warnings.append("Stop loss is more than 10% from entry price")
        return warnings

    def create_hedge_position(
        self, existing_position: Position, portfolio: Portfolio
    ) -> Tuple[Position, Decimal, List[str]]:
        """Создание хеджирующей позиции."""
        try:
            # Создаем противоположную позицию
            hedge_side = (
                PositionSide.SHORT if existing_position.side == PositionSide.LONG else PositionSide.LONG
            )
            return self.create_position(
                portfolio=portfolio,
                trading_pair=existing_position.trading_pair.symbol,
                side=hedge_side,
                quantity=existing_position.volume.amount,
                entry_price=existing_position.current_price.amount,
            )
        except Exception as e:
            raise PositionManagementError(f"Error creating hedge position: {str(e)}")

    def create_scaled_position(
        self,
        portfolio: Portfolio,
        trading_pair: str,
        side: PositionSide,
        total_quantity: Decimal,
        entry_price: Decimal,
        scale_levels: int,
        leverage: Optional[Decimal] = None,
    ) -> List[Tuple[Position, Decimal, List[str]]]:
        """Создание масштабированной позиции."""
        try:
            positions = []
            quantity_per_level = total_quantity / scale_levels
            
            for i in range(scale_levels):
                # Увеличиваем цену входа для каждого уровня
                level_price = entry_price * (Decimal("1") + Decimal("0.01") * i)
                position, margin, warnings = self.create_position(
                    portfolio=portfolio,
                    trading_pair=trading_pair,
                    side=side,
                    quantity=quantity_per_level,
                    entry_price=level_price,
                    leverage=leverage,
                )
                positions.append((position, margin, warnings))
            
            return positions
        except Exception as e:
            raise PositionManagementError(f"Error creating scaled position: {str(e)}")

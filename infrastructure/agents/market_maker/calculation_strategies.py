"""
Стратегии расчета для Market Maker агента.

Включает:
- Расчет спредов
- Определение размеров ордеров
- Позиционирование ордеров
- Управление рисками
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Protocol, runtime_checkable

from domain.value_objects.price import Price
from domain.value_objects.volume import Volume as Quantity
from domain.value_objects.currency import Currency
from domain.type_definitions import Symbol


@dataclass
class SpreadConfig:
    """Конфигурация спреда."""

    min_spread: Decimal
    max_spread: Decimal
    base_spread: Decimal
    volatility_multiplier: float
    volume_multiplier: float


@dataclass
class OrderSizeConfig:
    """Конфигурация размера ордера."""

    min_size: Quantity
    max_size: Quantity
    base_size: Quantity
    position_multiplier: float
    volatility_multiplier: float


@dataclass
class RiskConfig:
    """Конфигурация рисков."""

    max_position: Quantity
    max_daily_loss: Decimal
    max_order_size: Quantity
    stop_loss_threshold: Decimal


@runtime_checkable
class MarketDataProvider(Protocol):
    """Протокол для провайдера рыночных данных."""

    def get_current_price(self, symbol: Symbol) -> Price:
        """Получить текущую цену."""
        ...

    def get_order_book(self, symbol: Symbol) -> Dict[str, List[Dict[str, Any]]]:
        """Получить ордербук."""
        ...

    def get_volatility(self, symbol: Symbol, window: int = 20) -> float:
        """Получить волатильность."""
        ...

    def get_volume(self, symbol: Symbol, window: int = 20) -> float:
        """Получить объем."""
        ...


@runtime_checkable
class PositionManager(Protocol):
    """Протокол для управления позициями."""

    def get_current_position(self, symbol: Symbol) -> Quantity:
        """Получить текущую позицию."""
        ...

    def get_daily_pnl(self, symbol: Symbol) -> Decimal:
        """Получить дневной P&L."""
        ...

    def can_open_position(self, symbol: Symbol, size: Quantity) -> bool:
        """Проверить возможность открытия позиции."""
        ...


class SpreadCalculator(ABC):
    """Базовый класс для расчета спредов."""

    def __init__(self, config: SpreadConfig):
        self.config = config

    @abstractmethod
    def calculate_spread(
        self,
        symbol: Symbol,
        market_data: MarketDataProvider,
        position_manager: PositionManager,
    ) -> Dict[str, Any]:
        """Рассчитать спред для покупки и продажи."""


class AdaptiveSpreadCalculator(SpreadCalculator):
    """Адаптивный калькулятор спредов."""

    def calculate_spread(
        self,
        symbol: Symbol,
        market_data: MarketDataProvider,
        position_manager: PositionManager,
    ) -> Dict[str, Any]:
        """Рассчитать адаптивный спред."""
        current_price = market_data.get_current_price(symbol)
        volatility = market_data.get_volatility(symbol)
        volume = market_data.get_volume(symbol)
        position = position_manager.get_current_position(symbol)

        # Базовый спред
        base_spread = self.config.base_spread

        # Корректировка по волатильности
        volatility_adjustment = volatility * self.config.volatility_multiplier

        # Корректировка по объему
        volume_adjustment = (1 / (volume + 1e-6)) * self.config.volume_multiplier

        # Корректировка по позиции
        position_adjustment = abs(float(position.amount)) * 0.001

        # Итоговый спред
        total_adjustment = (
            volatility_adjustment + volume_adjustment + position_adjustment
        )
        spread = base_spread + Decimal(str(total_adjustment))

        # Ограничение спреда
        spread = max(self.config.min_spread, min(self.config.max_spread, spread))

        # Расчет цен
        bid_price = current_price.with_amount(current_price.amount - spread / 2)
        ask_price = current_price.with_amount(current_price.amount + spread / 2)

        return {"bid": bid_price, "ask": ask_price, "spread": spread}


class OrderSizeCalculator(ABC):
    """Базовый класс для расчета размеров ордеров."""

    def __init__(self, config: OrderSizeConfig):
        self.config = config

    @abstractmethod
    def calculate_order_size(
        self,
        symbol: Symbol,
        side: str,  # 'buy' или 'sell'
        market_data: MarketDataProvider,
        position_manager: PositionManager,
    ) -> Quantity:
        """Рассчитать размер ордера."""


class DynamicOrderSizeCalculator(OrderSizeCalculator):
    """Динамический калькулятор размеров ордеров."""

    def calculate_order_size(
        self,
        symbol: Symbol,
        side: str,
        market_data: MarketDataProvider,
        position_manager: PositionManager,
    ) -> Quantity:
        """Рассчитать динамический размер ордера."""
        base_size = self.config.base_size
        volatility = market_data.get_volatility(symbol)
        position = position_manager.get_current_position(symbol)

        # Корректировка по волатильности
        volatility_multiplier = 1 + volatility * self.config.volatility_multiplier

        # Корректировка по позиции
        if side == "buy" and position.amount > 0:
            # Уменьшаем размер при длинной позиции
            position_multiplier = (
                1 - abs(float(position.amount)) * self.config.position_multiplier
            )
        elif side == "sell" and position.amount < 0:
            # Уменьшаем размер при короткой позиции
            position_multiplier = (
                1 - abs(float(position.amount)) * self.config.position_multiplier
            )
        else:
            position_multiplier = 1.0

        # Итоговый размер
        size = base_size * Decimal(str(volatility_multiplier * position_multiplier))

        # Ограничение размера
        size = max(self.config.min_size, min(self.config.max_size, size))

        return size


class RiskManager:
    """Менеджер рисков для Market Maker."""

    def __init__(self, config: RiskConfig):
        self.config = config

    def validate_order(
        self,
        symbol: Symbol,
        side: str,
        size: Quantity,
        price: Price,
        position_manager: PositionManager,
    ) -> Dict[str, Any]:
        """Проверить ордер на соответствие риск-лимитам."""
        current_position = position_manager.get_current_position(symbol)
        daily_pnl = position_manager.get_daily_pnl(symbol)

        # Проверка размера ордера
        if size.value > self.config.max_order_size.value:
            return {"valid": False, "reason": "Order size exceeds maximum allowed"}

        # Проверка позиции
        new_position_value = current_position.value + (size.value if side == "buy" else -size.value)
        if abs(new_position_value) > self.config.max_position.value:
            return {"valid": False, "reason": "Position would exceed maximum allowed"}

        # Проверка дневного убытка
        if daily_pnl < -self.config.max_daily_loss:
            return {"valid": False, "reason": "Daily loss limit exceeded"}

        return {"valid": True, "reason": "Order validated successfully"}

    def should_stop_trading(
        self, symbol: Symbol, position_manager: PositionManager
    ) -> bool:
        """Проверить, нужно ли остановить торговлю."""
        daily_pnl = position_manager.get_daily_pnl(symbol)
        return daily_pnl < -self.config.max_daily_loss


class MarketMakerCalculationStrategy:
    """Основная стратегия расчета для Market Maker."""

    def __init__(
        self,
        spread_config: SpreadConfig,
        order_size_config: OrderSizeConfig,
        risk_config: RiskConfig,
    ):
        self.spread_calculator = AdaptiveSpreadCalculator(spread_config)
        self.order_size_calculator = DynamicOrderSizeCalculator(order_size_config)
        self.risk_manager = RiskManager(risk_config)

    def calculate_orders(
        self,
        symbol: Symbol,
        market_data: MarketDataProvider,
        position_manager: PositionManager,
    ) -> Dict[str, Any]:
        """Рассчитать ордера для размещения."""
        # Проверка рисков
        if self.risk_manager.should_stop_trading(symbol, position_manager):
            return {"should_trade": False, "reason": "Risk limits exceeded"}

        # Расчет спредов
        spread_data = self.spread_calculator.calculate_spread(
            symbol, market_data, position_manager
        )

        # Расчет размеров ордеров
        bid_size = self.order_size_calculator.calculate_order_size(
            symbol, "buy", market_data, position_manager
        )
        ask_size = self.order_size_calculator.calculate_order_size(
            symbol, "sell", market_data, position_manager
        )

        # Валидация ордеров
        bid_validation = self.risk_manager.validate_order(
            symbol, "buy", bid_size, spread_data["bid"], position_manager
        )
        ask_validation = self.risk_manager.validate_order(
            symbol, "sell", ask_size, spread_data["ask"], position_manager
        )

        return {
            "should_trade": True,
            "bid_order": {
                "price": spread_data["bid"],
                "size": bid_size,
                "valid": bid_validation["valid"],
                "reason": bid_validation["reason"],
            },
            "ask_order": {
                "price": spread_data["ask"],
                "size": ask_size,
                "valid": ask_validation["valid"],
                "reason": ask_validation["reason"],
            },
            "spread": spread_data["spread"],
        }


# Фабричные функции для создания конфигураций
def create_default_spread_config() -> SpreadConfig:
    """Создать конфигурацию спреда по умолчанию."""
    return SpreadConfig(
        min_spread=Decimal("0.001"),
        max_spread=Decimal("0.05"),
        base_spread=Decimal("0.01"),
        volatility_multiplier=0.1,
        volume_multiplier=0.001,
    )


def create_default_order_size_config() -> OrderSizeConfig:
    """Создать конфигурацию размера ордера по умолчанию."""
    return OrderSizeConfig(
        min_size=Quantity(Decimal("0.001"), Currency.USD),
        max_size=Quantity(Decimal("1.0"), Currency.USD),
        base_size=Quantity(Decimal("0.1"), Currency.USD),
        position_multiplier=0.01,
        volatility_multiplier=0.5,
    )


def create_default_risk_config() -> RiskConfig:
    """Создать конфигурацию рисков по умолчанию."""
    return RiskConfig(
        max_position=Quantity(Decimal("10.0"), Currency.USD),
        max_daily_loss=Decimal("1000.0"),
        max_order_size=Quantity(Decimal("1.0"), Currency.USD),
        stop_loss_threshold=Decimal("0.1"),
    )

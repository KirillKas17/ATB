"""
Value Objects для доменной модели.
Этот модуль содержит все Value Objects, используемые в доменной модели.
Каждый Value Object представляет собой неизменяемый объект, который инкапсулирует
бизнес-логику и валидацию для конкретного типа данных.
"""

__all__ = [
    # Базовые классы
    "BaseValueObject",
    # Основные Value Objects
    "Currency",
    "CurrencyCode",
    "CurrencyType",
    "CurrencyNetwork",
    "Money",
    "Price",
    "Percentage",
    "Volume",
    "TradingPair",
    "Signal",
    "Timestamp",
    # Конфигурации
    "MoneyConfig",
    "PriceConfig",
    "PercentageConfig",
    "VolumeConfig",
    "TradingPairConfig",
    "SignalConfig",
    # Enums для сигналов
    "SignalStrength",
    "SignalDirection",
    "SignalType",
    # Кэши
    "MoneyCache",
    "PriceCache",
    # Фабрика
    "ValueObjectFactory",
]

from .base_value_object import BaseValueObject
from .currency import Currency, CurrencyType, CurrencyNetwork, CurrencyCode
from .money import Money, MoneyConfig
from .price import Price
from .price_config import PriceConfig
from .price_cache import PriceCache
from .volume import Volume
from .percentage import Percentage
from .trading_pair import TradingPair
from .signal_config import SignalStrength, SignalDirection, SignalType
from .signal import Signal
from .timestamp import Timestamp
from .factory import ValueObjectFactory

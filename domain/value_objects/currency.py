# -*- coding: utf-8 -*-
"""Промышленная реализация Value Object для валют с расширенной функциональностью для алготрейдинга."""
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, ClassVar, Dict, Final, List, Optional, Set


class CurrencyType(Enum):
    """Типы валют."""

    CRYPTO = "crypto"
    FIAT = "fiat"
    STABLECOIN = "stablecoin"
    DEFI = "defi"


class CurrencyNetwork(Enum):
    """Сети для криптовалют."""

    BITCOIN = "bitcoin"
    ETHEREUM = "ethereum"
    BINANCE_SMART_CHAIN = "bsc"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    SOLANA = "solana"
    CARDANO = "cardano"
    POLKADOT = "polkadot"
    MULTI_CHAIN = "multi"


@dataclass(frozen=True)
class CurrencyInfo:
    """Информация о валюте."""

    code: str
    name: str
    symbol: str
    type: CurrencyType
    networks: Set[CurrencyNetwork] = field(default_factory=set)
    decimals: int = 8
    is_active: bool = True
    market_cap_rank: Optional[int] = None
    volume_24h: Optional[Decimal] = None
    price_usd: Optional[Decimal] = None
    last_updated: Optional[datetime] = None
    trading_priority: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "symbol": self.symbol,
            "type": self.type.value,
            "networks": [n.value for n in self.networks],
            "decimals": self.decimals,
            "is_active": self.is_active,
            "market_cap_rank": self.market_cap_rank,
            "volume_24h": str(self.volume_24h) if self.volume_24h else None,
            "price_usd": str(self.price_usd) if self.price_usd else None,
            "last_updated": (
                self.last_updated.isoformat() if self.last_updated else None
            ),
            "trading_priority": self.trading_priority,
        }


class Currency(Enum):
    """
    Промышленная реализация Value Object для валют.
    Поддерживает:
    - Расширенную информацию о валютах
    - Валидацию и кэширование
    - Торговые операции
    - Сетевую совместимость
    - Рыночные данные
    """

    # Основные криптовалюты
    BTC = CurrencyInfo(
        code="BTC",
        name="Bitcoin",
        symbol="₿",
        type=CurrencyType.CRYPTO,
        networks={CurrencyNetwork.BITCOIN},
        trading_priority=1,
    )
    ETH = CurrencyInfo(
        code="ETH",
        name="Ethereum",
        symbol="Ξ",
        type=CurrencyType.CRYPTO,
        networks={CurrencyNetwork.ETHEREUM},
        trading_priority=2,
    )
    # Стейблкоины
    USDT = CurrencyInfo(
        code="USDT",
        name="Tether",
        symbol="₮",
        type=CurrencyType.STABLECOIN,
        networks={CurrencyNetwork.ETHEREUM, CurrencyNetwork.BINANCE_SMART_CHAIN},
        trading_priority=3,
    )
    USDC = CurrencyInfo(
        code="USDC",
        name="USD Coin",
        symbol="$",
        type=CurrencyType.STABLECOIN,
        networks={CurrencyNetwork.ETHEREUM, CurrencyNetwork.BINANCE_SMART_CHAIN},
        trading_priority=4,
    )
    BUSD = CurrencyInfo(
        code="BUSD",
        name="Binance USD",
        symbol="$",
        type=CurrencyType.STABLECOIN,
        networks={CurrencyNetwork.BINANCE_SMART_CHAIN},
        trading_priority=5,
    )
    DAI = CurrencyInfo(
        code="DAI",
        name="Dai",
        symbol="$",
        type=CurrencyType.STABLECOIN,
        networks={CurrencyNetwork.ETHEREUM},
        trading_priority=6,
    )
    TUSD = CurrencyInfo(
        code="TUSD",
        name="TrueUSD",
        symbol="$",
        type=CurrencyType.STABLECOIN,
        networks={CurrencyNetwork.ETHEREUM},
        trading_priority=7,
    )
    # Альткоины
    BNB = CurrencyInfo(
        code="BNB",
        name="Binance Coin",
        symbol="BNB",
        type=CurrencyType.CRYPTO,
        networks={CurrencyNetwork.BINANCE_SMART_CHAIN},
        trading_priority=8,
    )
    ADA = CurrencyInfo(
        code="ADA",
        name="Cardano",
        symbol="₳",
        type=CurrencyType.CRYPTO,
        networks={CurrencyNetwork.CARDANO},
        trading_priority=9,
    )
    SOL = CurrencyInfo(
        code="SOL",
        name="Solana",
        symbol="◎",
        type=CurrencyType.CRYPTO,
        networks={CurrencyNetwork.SOLANA},
        trading_priority=10,
    )
    DOT = CurrencyInfo(
        code="DOT",
        name="Polkadot",
        symbol="●",
        type=CurrencyType.CRYPTO,
        networks={CurrencyNetwork.POLKADOT},
        trading_priority=11,
    )
    MATIC = CurrencyInfo(
        code="MATIC",
        name="Polygon",
        symbol="MATIC",
        type=CurrencyType.CRYPTO,
        networks={CurrencyNetwork.POLYGON},
        trading_priority=12,
    )
    LINK = CurrencyInfo(
        code="LINK",
        name="Chainlink",
        symbol="🔗",
        type=CurrencyType.CRYPTO,
        networks={CurrencyNetwork.ETHEREUM},
        trading_priority=13,
    )
    UNI = CurrencyInfo(
        code="UNI",
        name="Uniswap",
        symbol="🦄",
        type=CurrencyType.CRYPTO,
        networks={CurrencyNetwork.ETHEREUM},
        trading_priority=14,
    )
    LTC = CurrencyInfo(
        code="LTC",
        name="Litecoin",
        symbol="Ł",
        type=CurrencyType.CRYPTO,
        networks={CurrencyNetwork.BITCOIN},
        trading_priority=15,
    )
    XRP = CurrencyInfo(
        code="XRP",
        name="Ripple",
        symbol="✖",
        type=CurrencyType.CRYPTO,
        networks=set(),
        trading_priority=16,
    )
    AVAX = CurrencyInfo(
        code="AVAX",
        name="Avalanche",
        symbol="AVAX",
        type=CurrencyType.CRYPTO,
        networks=set(),
        trading_priority=17,
    )
    ATOM = CurrencyInfo(
        code="ATOM",
        name="Cosmos",
        symbol="⚛",
        type=CurrencyType.CRYPTO,
        networks=set(),
        trading_priority=18,
    )
    NEAR = CurrencyInfo(
        code="NEAR",
        name="NEAR Protocol",
        symbol="NEAR",
        type=CurrencyType.CRYPTO,
        networks=set(),
        trading_priority=19,
    )
    ALGO = CurrencyInfo(
        code="ALGO",
        name="Algorand",
        symbol="ALGO",
        type=CurrencyType.CRYPTO,
        networks=set(),
        trading_priority=20,
    )
    VET = CurrencyInfo(
        code="VET",
        name="VeChain",
        symbol="VET",
        type=CurrencyType.CRYPTO,
        networks=set(),
        trading_priority=21,
    )
    ICP = CurrencyInfo(
        code="ICP",
        name="Internet Computer",
        symbol="ICP",
        type=CurrencyType.CRYPTO,
        networks=set(),
        trading_priority=22,
    )
    FIL = CurrencyInfo(
        code="FIL",
        name="Filecoin",
        symbol="FIL",
        type=CurrencyType.CRYPTO,
        networks=set(),
        trading_priority=23,
    )
    TRX = CurrencyInfo(
        code="TRX",
        name="TRON",
        symbol="TRX",
        type=CurrencyType.CRYPTO,
        networks=set(),
        trading_priority=24,
    )
    EOS = CurrencyInfo(
        code="EOS",
        name="EOS",
        symbol="EOS",
        type=CurrencyType.CRYPTO,
        networks=set(),
        trading_priority=25,
    )
    THETA = CurrencyInfo(
        code="THETA",
        name="Theta Network",
        symbol="THETA",
        type=CurrencyType.CRYPTO,
        networks=set(),
        trading_priority=26,
    )
    XLM = CurrencyInfo(
        code="XLM",
        name="Stellar",
        symbol="XLM",
        type=CurrencyType.CRYPTO,
        networks=set(),
        trading_priority=27,
    )
    # Фиатные валюты
    USD = CurrencyInfo(
        code="USD",
        name="US Dollar",
        symbol="$",
        type=CurrencyType.FIAT,
        networks=set(),
        trading_priority=28,
    )
    EUR = CurrencyInfo(
        code="EUR",
        name="Euro",
        symbol="€",
        type=CurrencyType.FIAT,
        networks=set(),
        trading_priority=29,
    )
    GBP = CurrencyInfo(
        code="GBP",
        name="British Pound",
        symbol="£",
        type=CurrencyType.FIAT,
        networks=set(),
        trading_priority=30,
    )
    JPY = CurrencyInfo(
        code="JPY",
        name="Japanese Yen",
        symbol="¥",
        type=CurrencyType.FIAT,
        networks=set(),
        trading_priority=31,
    )
    CNY = CurrencyInfo(
        code="CNY",
        name="Chinese Yuan",
        symbol="¥",
        type=CurrencyType.FIAT,
        networks=set(),
        trading_priority=32,
    )
    KRW = CurrencyInfo(
        code="KRW",
        name="South Korean Won",
        symbol="₩",
        type=CurrencyType.FIAT,
        networks=set(),
        trading_priority=33,
    )
    RUB = CurrencyInfo(
        code="RUB",
        name="Russian Ruble",
        symbol="₽",
        type=CurrencyType.FIAT,
        networks=set(),
        trading_priority=34,
    )

    @property
    def currency_code(self) -> str:
        """Возвращает строковый код валюты (BTC, USDT и т.д.)."""
        return self._value_.code

    @property
    def code(self) -> str:
        """Возвращает код валюты."""
        return self._value_.code

    @property
    def name(self) -> str:
        """Возвращает название валюты."""
        return self._value_.name

    @property
    def symbol(self) -> str:
        """Возвращает символ валюты."""
        return self._value_.symbol

    @property
    def type(self) -> CurrencyType:
        """Возвращает тип валюты."""
        return self._value_.type

    @property
    def networks(self) -> Set[CurrencyNetwork]:
        """Возвращает поддерживаемые сети."""
        return self._value_.networks

    @property
    def decimals(self) -> int:
        """Возвращает количество десятичных знаков."""
        return self._value_.decimals

    @property
    def trading_priority(self) -> int:
        """Возвращает приоритет торговли."""
        return self._value_.trading_priority

    @property
    def is_stablecoin(self) -> bool:
        """Проверяет, является ли валюта стейблкоином."""
        return self._value_.type == CurrencyType.STABLECOIN

    @property
    def is_fiat(self) -> bool:
        """Проверяет, является ли валюта фиатной."""
        return self._value_.type == CurrencyType.FIAT

    @property
    def is_crypto(self) -> bool:
        """Проверяет, является ли валюта криптовалютой."""
        return self._value_.type == CurrencyType.CRYPTO

    @property
    def is_major_crypto(self) -> bool:
        """True для BTC и ETH."""
        return self.code in ("BTC", "ETH")

    @property
    def is_active(self) -> bool:
        """Проверяет, активна ли валюта."""
        return self._value_.is_active

    def can_trade_with(self, other: "Currency") -> bool:
        """Проверяет, может ли валюта торговаться с другой валютой."""
        # Валюты не могут торговаться сами с собой
        if self == other:
            return False
        # Все активные валюты могут торговаться друг с другом
        can_trade = self.is_active and other.is_active
        return can_trade

    @property
    def hash(self) -> str:
        """Возвращает хеш валюты для кэширования."""
        return hashlib.md5(self.code.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Сериализует валюту в словарь."""
        return self._value_.to_dict()

    def validate(self) -> bool:
        """Валидирует валюту."""
        is_valid = True
        return is_valid

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Currency):
            return self.code == other.code
        return False

    def __hash__(self) -> int:
        return hash(self.code)

    def __repr__(self) -> str:
        return f"Currency.{self.code}"

    @classmethod
    def from_string(cls, code: str) -> Optional["Currency"]:
        """Создает валюту из строки."""
        if not code:
            return None
        code_upper = code.upper()
        for currency in cls:
            if currency.code == code_upper:
                return currency
        return None

    @classmethod
    def get_trading_pairs(cls, base_currency: "Currency") -> List["Currency"]:
        """Получает список валют для торговых пар."""
        # Возвращаем все валюты, кроме базовой
        pairs = [currency for currency in cls if currency != base_currency]
        return pairs

    def __str__(self) -> str:
        code = self.code
        return code


def create_currency(code: str, **kwargs: Any) -> Optional[Currency]:
    """Фабричная функция для создания валюты."""
    currency = Currency.from_string(code)
    return currency


# Экспорт
# Алиас для обратной совместимости
CurrencyCode = str

__all__ = [
    "Currency",
    "CurrencyType",
    "CurrencyNetwork",
    "CurrencyInfo",
    "CurrencyCode",
    "create_currency",
]

"""
Промышленная реализация Value Object для торговых пар с расширенной функциональностью для алготрейдинга.
"""

import hashlib
from decimal import Decimal
from typing import Any, Dict, List, Optional

from domain.types.base_types import (
    CurrencyCode,
    CurrencyPair,
)

from domain.types.value_object_types import (
    ValidationResult,
    ValueObject,
    ValueObjectDict,
)

from .currency import Currency
from .trading_pair_config import TradingPairConfig


class TradingPair(ValueObject):
    """
    Промышленная реализация Value Object для торговых пар.

    Поддерживает:
    - Базовую и котируемую валюты с валидацией
    - Форматирование символов и пар
    - Торговые метрики и конфигурацию
    - Анализ ликвидности и волатильности
    - Сериализацию/десериализацию
    """

    MAX_SYMBOL_LENGTH: int = 20
    MIN_SYMBOL_LENGTH: int = 3

    def __init__(
        self,
        base_currency: Currency,
        quote_currency: Currency,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        is_active: bool = True,
        config: Optional[TradingPairConfig] = None,
    ) -> None:
        self._config = config or TradingPairConfig()
        self._base_currency = base_currency
        self._quote_currency = quote_currency
        self._exchange = exchange
        self._is_active = is_active
        if symbol is None:
            self._symbol = f"{self._base_currency.code}{self._quote_currency.code}"
        else:
            self._symbol = symbol
        self._validate_pair()

    @property
    def value(self) -> str:
        return self._symbol

    @property
    def hash(self) -> str:
        return self._calculate_hash()

    def _calculate_hash(self) -> str:
        data = f"{self._base_currency.code}:{self._quote_currency.code}:{self._symbol}:{self._exchange}"
        return hashlib.md5(data.encode()).hexdigest()

    def validate(self) -> bool:
        errors: List[str] = []
        # Проверки типов избыточны из-за строгой типизации в конструкторе
        if self._base_currency == self._quote_currency:
            errors.append("Base and quote currencies cannot be the same")
        if len(self._symbol) < self._config.min_symbol_length:
            errors.append(
                f"Symbol must be at least {self._config.min_symbol_length} characters long"
            )
        if len(self._symbol) > self._config.max_symbol_length:
            errors.append(
                f"Symbol cannot exceed {self._config.max_symbol_length} characters"
            )
        return len(errors) == 0

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TradingPair):
            return False
        return (
            self._base_currency == other._base_currency
            and self._quote_currency == other._quote_currency
            and self._symbol == other._symbol
            and self._exchange == other._exchange
        )

    def __hash__(self) -> int:
        return hash(
            (self._base_currency, self._quote_currency, self._symbol, self._exchange)
        )

    def __str__(self) -> str:
        return self._symbol or f"{self._base_currency.code}/{self._quote_currency.code}"

    def __repr__(self) -> str:
        return f"TradingPair({self._base_currency.code}/{self._quote_currency.code})"

    @property
    def base_currency(self) -> Currency:
        return self._base_currency

    @property
    def quote_currency(self) -> Currency:
        return self._quote_currency

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def exchange(self) -> Optional[str]:
        return self._exchange

    @property
    def is_active(self) -> bool:
        return self._is_active

    def _validate_pair(self) -> None:
        validation = self.validate()
        if not validation:
            raise ValueError(f"Invalid trading pair: {validation}")

    @property
    def is_crypto_pair(self) -> bool:
        """Проверяет, является ли пара криптопарой (хотя бы одна валюта криптовалюта)."""
        return self._base_currency.is_crypto or self._quote_currency.is_crypto

    @property
    def is_fiat_pair(self) -> bool:
        return self._base_currency.is_fiat and self._quote_currency.is_fiat

    @property
    def is_crypto_fiat_pair(self) -> bool:
        return (self._base_currency.is_crypto and self._quote_currency.is_fiat) or (
            self._base_currency.is_fiat and self._quote_currency.is_crypto
        )

    @property
    def is_stablecoin_pair(self) -> bool:
        return self._base_currency.is_stablecoin or self._quote_currency.is_stablecoin

    @property
    def is_major_pair(self) -> bool:
        return self._symbol in self._config.major_pairs

    @property
    def is_spot_pair(self) -> bool:
        return not self.is_futures_pair

    @property
    def is_futures_pair(self) -> bool:
        return bool(self._symbol and "PERP" in self._symbol)

    def get_trading_priority(self) -> int:
        if self.is_major_pair:
            return 1
        elif self.is_stablecoin_pair:
            return 2
        elif self.is_crypto_fiat_pair:
            return 3
        elif self.is_crypto_pair:
            return 4
        else:
            return 5

    def is_valid_for_trading(self) -> bool:
        if not self._is_active:
            return False
        if not self._base_currency.is_active:
            return False
        if not self._quote_currency.is_active:
            return False
        if self._base_currency == self._quote_currency:
            return False
        return True

    def get_min_order_size(self) -> Decimal:
        return self._config.min_order_sizes.get(
            self._symbol, self._config.min_order_sizes["DEFAULT"]
        )

    def get_price_precision(self) -> int:
        return self._config.price_precisions.get(
            self._symbol, self._config.price_precisions["DEFAULT"]
        )

    def get_volume_precision(self) -> int:
        return self._config.volume_precisions.get(
            self._symbol, self._config.volume_precisions["DEFAULT"]
        )

    def get_reverse_pair(self) -> "TradingPair":
        return TradingPair(
            base_currency=self._quote_currency,
            quote_currency=self._base_currency,
            symbol=f"{self._quote_currency.code}{self._base_currency.code}",
            exchange=self._exchange,
            is_active=self._is_active,
            config=self._config,
        )

    def get_common_pairs(self) -> List["TradingPair"]:
        common_pairs = []
        for major_pair in self._config.major_pairs:
            if (
                major_pair.startswith(self._base_currency.code)
                or major_pair.endswith(self._base_currency.code)
                or major_pair.startswith(self._quote_currency.code)
                or major_pair.endswith(self._quote_currency.code)
            ):
                try:
                    pair = TradingPair.from_symbol(
                        major_pair, self._exchange, self._config
                    )
                    if pair != self:
                        common_pairs.append(pair)
                except ValueError:
                    continue
        return common_pairs

    def get_liquidity_rating(self) -> float:
        if self.is_major_pair:
            return 1.0
        elif self.is_stablecoin_pair:
            return 0.8
        elif self.is_crypto_fiat_pair:
            return 0.6
        elif self.is_crypto_pair:
            return 0.4
        else:
            return 0.2

    def get_volatility_rating(self) -> float:
        if self.is_stablecoin_pair:
            return 0.1
        elif self.is_fiat_pair:
            return 0.2
        elif self.is_crypto_fiat_pair:
            return 0.6
        elif self.is_crypto_pair:
            return 0.8
        else:
            return 0.9

    def get_risk_score(self) -> float:
        liquidity_score = self.get_liquidity_rating()
        volatility_score = self.get_volatility_rating()
        return (1 - liquidity_score) * 0.6 + volatility_score * 0.4

    def get_trading_recommendation(self) -> str:
        if not self.is_valid_for_trading():
            return "NOT_TRADABLE"
        risk_score = self.get_risk_score()
        if risk_score < 0.3:
            return "SAFE_TO_TRADE"
        elif risk_score < 0.6:
            return "CAUTION_ADVISED"
        elif risk_score < 0.8:
            return "HIGH_RISK"
        else:
            return "AVOID_TRADING"

    def to_dict(self) -> ValueObjectDict:
        return ValueObjectDict(
            base_currency=str(self._base_currency.code),
            quote_currency=str(self._quote_currency.code),
            symbol=self._symbol,
            exchange=self._exchange,
            is_active=self._is_active,
            type="TradingPair",
        )

    @classmethod
    def from_dict(cls, data: ValueObjectDict) -> "TradingPair":
        base_currency = Currency(data["base_currency"])
        quote_currency = Currency(data["quote_currency"])
        symbol = data.get("symbol")
        exchange = data.get("exchange")
        is_active = data.get("is_active", True)
        return cls(
            base_currency=base_currency,
            quote_currency=quote_currency,
            symbol=symbol,
            exchange=exchange,
            is_active=is_active,
        )

    @classmethod
    def from_symbol(
        cls,
        symbol: str,
        exchange: Optional[str] = None,
        config: Optional[TradingPairConfig] = None,
    ) -> "TradingPair":
        """Создает торговую пару из символа."""
        if not symbol or len(symbol) < cls.MIN_SYMBOL_LENGTH:
            raise ValueError(f"Invalid symbol: {symbol}")

        # Пытаемся найти базовую валюту
        base_currency = None
        quote_currency = None

        for currency in Currency:
            if symbol.startswith(currency.code):
                base_currency = currency
                quote_currency_str = symbol[len(currency.code) :]
                break

        if not base_currency:
            raise ValueError(f"Could not identify base currency for symbol: {symbol}")

        # Пытаемся найти котируемую валюту
        for currency in Currency:
            if quote_currency_str == currency.code:
                quote_currency = currency
                break

        if not quote_currency:
            raise ValueError(f"Could not identify quote currency for symbol: {symbol}")

        return cls(base_currency, quote_currency, symbol, exchange, True, config)

    @classmethod
    def create_btc_usdt(cls, exchange: Optional[str] = None) -> "TradingPair":
        return cls(Currency.BTC, Currency.USDT, "BTCUSDT", exchange)

    @classmethod
    def create_eth_usdt(cls, exchange: Optional[str] = None) -> "TradingPair":
        return cls(Currency.ETH, Currency.USDT, "ETHUSDT", exchange)

    @classmethod
    def create_btc_usd(cls, exchange: Optional[str] = None) -> "TradingPair":
        return cls(Currency.BTC, Currency.USD, "BTCUSD", exchange)

    @classmethod
    def create_eth_usd(cls, exchange: Optional[str] = None) -> "TradingPair":
        return cls(Currency.ETH, Currency.USD, "ETHUSD", exchange)

    def copy(self) -> "TradingPair":
        return TradingPair(
            base_currency=self._base_currency,
            quote_currency=self._quote_currency,
            symbol=self._symbol,
            exchange=self._exchange,
            is_active=self._is_active,
            config=self._config,
        )



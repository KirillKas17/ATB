from typing import TYPE_CHECKING, Dict, Optional, Union

from domain.type_definitions.value_object_types import (
    CurrencyCode,
    NumericType,
    generate_cache_key,
)

from .currency import Currency
from .money_config import MoneyConfig

if TYPE_CHECKING:
    from .money import Money

class MoneyCache:
    """Класс для кэширования Money объектов."""
    _money_cache: Dict[str, 'Money'] = {}
    _config: MoneyConfig = MoneyConfig()
    @classmethod
    def get_cached(cls, amount: NumericType, currency: Union[Currency, str]) -> Optional['Money']:
        if not cls._config.cache_enabled:
            return None
        if isinstance(currency, str):
            currency_obj = Currency.from_string(currency)
            if currency_obj is None:
                return None
            currency = currency_obj
        cache_key = generate_cache_key('Money', amount=str(amount), currency=str(currency.code))
        return cls._money_cache.get(cache_key)
    @classmethod
    def set_cached(cls, money: 'Money') -> None:
        if not cls._config.cache_enabled:
            return
        currency_code = money.currency
        cache_key = generate_cache_key('Money', amount=str(money.amount), currency=currency_code)
        cls._money_cache[cache_key] = money
    @classmethod
    def clear_cache(cls) -> None:
        cls._money_cache.clear()
    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        return {
            'money_objects': len(cls._money_cache),
            'cache_enabled': int(cls._config.cache_enabled)
        } 
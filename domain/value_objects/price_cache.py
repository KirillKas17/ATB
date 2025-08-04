from typing import TYPE_CHECKING, Dict, Optional

from domain.type_definitions.value_object_types import (
    CurrencyCode,
    NumericType,
    generate_cache_key,
)

from .currency import Currency
from .price import Price
from .price_config import PriceConfig

if TYPE_CHECKING:
    pass

class PriceCache:
    """Класс для кэширования Price объектов."""
    _price_cache: Dict[str, 'Price'] = {}
    _config: PriceConfig = PriceConfig()
    @classmethod
    def get_cached(cls, amount: NumericType, base_currency: Currency, quote_currency: Currency) -> Optional['Price']:
        if not cls._config.cache_enabled:
            return None
        cache_key = generate_cache_key('Price', amount=str(amount), base_currency=str(base_currency.code), quote_currency=str(quote_currency.code))
        return cls._price_cache.get(cache_key)
    @classmethod
    def set_cached(cls, price: 'Price') -> None:
        if cls._config.cache_enabled:
            cache_key = generate_cache_key(
                'Price',
                amount=str(price.amount),
                base_currency=str(price.currency.code),
                quote_currency=str(price.quote_currency.code) if price.quote_currency is not None else "?"
            )
            cls._price_cache[cache_key] = price
    @classmethod
    def clear_cache(cls) -> None:
        cls._price_cache.clear()
    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        return {
            'price_objects': len(cls._price_cache),
            'cache_enabled': int(cls._config.cache_enabled)
        } 
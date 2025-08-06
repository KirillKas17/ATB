"""
Symbol entity для торговых пар и символов.
"""

from dataclasses import dataclass
from typing import Optional, Any
from datetime import datetime

from domain.value_objects.currency import Currency


@dataclass
class Symbol:
    """Торговый символ (пара)."""
    
    base_currency: Currency
    quote_currency: Currency
    name: Optional[str] = None
    is_active: bool = True
    created_at: datetime = None
    
    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.name is None:
            self.name = f"{self.base_currency.value.code}/{self.quote_currency.value.code}"
    
    @property
    def symbol_name(self) -> str:
        """Имя символа."""
        return self.name or f"{self.base_currency.value.code}/{self.quote_currency.value.code}"
    
    @property
    def base_code(self) -> str:
        """Код базовой валюты."""
        return self.base_currency.value.code
    
    @property
    def quote_code(self) -> str:
        """Код котируемой валюты."""
        return self.quote_currency.value.code
    
    def __str__(self) -> str:
        return self.symbol_name
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Symbol):
            return False
        return (self.base_currency == other.base_currency and 
                self.quote_currency == other.quote_currency)
    
    def __hash__(self) -> int:
        return hash((self.base_currency, self.quote_currency))
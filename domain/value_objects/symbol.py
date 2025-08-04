"""
Value Object для торгового символа.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import re

from domain.value_objects.base_value_object import BaseValueObject

@dataclass(frozen=True)
class Symbol(BaseValueObject):
    """
    Торговый символ.
    
    Attributes:
        value: Строковое представление символа (например, "BTCUSDT")
        base: Базовая валюта (например, "BTC")
        quote: Котируемая валюта (например, "USDT")
        exchange: Биржа (опционально)
    """
    
    value: str
    base: Optional[str] = None
    quote: Optional[str] = None
    exchange: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Валидация и парсинг символа."""
        if not self.value:
            raise ValueError("Symbol value cannot be empty")
        
        # Проверка формата символа
        if not re.match(r'^[A-Z0-9]{2,}$', self.value.upper()):
            raise ValueError(f"Invalid symbol format: {self.value}")
        
        # Автоматический парсинг base/quote если не заданы
        if not self.base or not self.quote:
            parsed = self._parse_symbol(self.value)
            if parsed:
                object.__setattr__(self, 'base', parsed[0])
                object.__setattr__(self, 'quote', parsed[1])
    
    def _parse_symbol(self, symbol: str) -> Optional[tuple[str, str]]:
        """Парсинг символа на base и quote валюты."""
        symbol = symbol.upper()
        
        # Список популярных quote валют
        quote_currencies = ['USDT', 'USDC', 'USD', 'EUR', 'BTC', 'ETH', 'BNB']
        
        for quote in quote_currencies:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                if len(base) >= 2:
                    return (base, quote)
        
        # Попытка парсинга для других форматов
        if len(symbol) >= 6:
            # Предполагаем, что первые 3 символа - base
            return (symbol[:3], symbol[3:])
        
        return None
    
    @property
    def normalized(self) -> str:
        """Нормализованное представление символа."""
        return self.value.upper()
    
    def to_exchange_format(self, exchange: str) -> str:
        """Конвертация в формат конкретной биржи."""
        normalized = self.normalized
        
        if exchange.lower() == 'binance':
            return normalized
        elif exchange.lower() == 'bybit':
            return normalized
        elif exchange.lower() == 'okx':
            return f"{self.base}-{self.quote}" if self.base and self.quote else normalized
        else:
            return normalized
    
    def is_crypto_pair(self) -> bool:
        """Проверка, является ли символ криптовалютной парой."""
        crypto_symbols = {'BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'SOL', 'MATIC'}
        stable_coins = {'USDT', 'USDC', 'BUSD', 'DAI'}
        
        if self.base and self.quote:
            return (self.base in crypto_symbols or self.quote in crypto_symbols or 
                   self.quote in stable_coins)
        return True  # По умолчанию считаем крипто-парой
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            'value': self.value,
            'base': self.base,
            'quote': self.quote,
            'exchange': self.exchange
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Symbol':
        """Создание из словаря."""
        return cls(
            value=data['value'],
            base=data.get('base'),
            quote=data.get('quote'),
            exchange=data.get('exchange')
        )
    
    @classmethod
    def from_string(cls, symbol_str: str, exchange: Optional[str] = None) -> 'Symbol':
        """Создание из строки."""
        return cls(value=symbol_str, exchange=exchange)
    
    def __str__(self) -> str:
        """Строковое представление."""
        return self.value
    
    def __repr__(self) -> str:
        """Подробное представление."""
        return f"Symbol(value={self.value}, base={self.base}, quote={self.quote})"
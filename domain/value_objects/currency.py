"""
Currency value object.
"""

from dataclasses import dataclass
from enum import Enum


class Currency(Enum):
    """Currency enumeration."""
    USD = "USD"
    EUR = "EUR"
    BTC = "BTC"
    ETH = "ETH"
    
    @property
    def code(self) -> str:
        return self.value

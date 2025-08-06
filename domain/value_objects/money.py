"""
Money value object.
"""

from decimal import Decimal
from dataclasses import dataclass
from .currency import Currency


@dataclass(frozen=True)
class Money:
    """Money value object."""
    
    amount: Decimal
    currency: Currency
    
    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("Money amount cannot be negative")
    
    def __str__(self) -> str:
        return f"{self.amount} {self.currency.code}"

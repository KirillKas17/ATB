"""
Price value object.
"""

from decimal import Decimal
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Price:
    """Price value object."""
    
    amount: Decimal
    currency: str = "USD"
    
    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("Price cannot be negative")
    
    def __str__(self) -> str:
        return f"{self.amount} {self.currency}"

"""
Volume value object.
"""

from decimal import Decimal
from dataclasses import dataclass


@dataclass(frozen=True)
class Volume:
    """Volume value object."""
    
    amount: Decimal
    
    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("Volume cannot be negative")
    
    def __str__(self) -> str:
        return str(self.amount)

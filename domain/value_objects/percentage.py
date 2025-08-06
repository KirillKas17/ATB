"""
Percentage value object.
"""

from decimal import Decimal
from dataclasses import dataclass


@dataclass(frozen=True)
class Percentage:
    """Percentage value object."""
    
    value: Decimal
    
    def __post_init__(self):
        if self.value < 0 or self.value > 100:
            raise ValueError("Percentage must be between 0 and 100")
    
    def __str__(self) -> str:
        return f"{self.value}%"

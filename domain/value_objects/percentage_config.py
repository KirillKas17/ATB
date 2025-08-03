from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from domain.types.value_object_types import PERCENTAGE_PRECISION


@dataclass(frozen=True)
class PercentageConfig:
    """Конфигурация для Percentage value object."""

    precision: int = PERCENTAGE_PRECISION
    rounding_mode: str = "ROUND_HALF_UP"
    allow_negative: bool = True
    validate_limits: bool = True
    max_percentage: Decimal = Decimal("10000")
    min_percentage: Decimal = Decimal("-10000")

    def get_rounding_mode(self) -> Any:
        from decimal import Decimal

        return getattr(Decimal, self.rounding_mode)

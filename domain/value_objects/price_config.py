from dataclasses import dataclass
from decimal import ROUND_HALF_UP
from typing import Any

from domain.type_definitions.value_object_types import PRICE_PRECISION


@dataclass(frozen=True)
class PriceConfig:
    """Конфигурация для Price value object."""

    precision: int = PRICE_PRECISION
    rounding_mode: str = "ROUND_HALF_UP"
    allow_zero: bool = False
    validate_limits: bool = True
    cache_enabled: bool = True

    def get_rounding_mode(self) -> Any:
        return ROUND_HALF_UP

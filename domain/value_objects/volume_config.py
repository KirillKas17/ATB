from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from domain.type_definitions.value_object_types import VOLUME_PRECISION


@dataclass(frozen=True)
class VolumeConfig:
    """Конфигурация для Volume value object."""

    precision: int = VOLUME_PRECISION
    rounding_mode: str = "ROUND_HALF_UP"
    allow_negative: bool = False
    validate_limits: bool = True
    max_volume: Decimal = Decimal("999999999999.99999999")
    min_volume: Decimal = Decimal("0")
    high_liquidity_threshold: Decimal = Decimal("1000000")
    medium_liquidity_threshold: Decimal = Decimal("100000")
    low_liquidity_threshold: Decimal = Decimal("10000")

    def get_rounding_mode(self) -> Any:
        from decimal import Decimal

        return getattr(Decimal, self.rounding_mode)

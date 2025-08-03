from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from domain.types.value_object_types import MONEY_PRECISION


@dataclass(frozen=True)
class MoneyConfig:
    """Конфигурация для Money value object."""

    precision: int = MONEY_PRECISION
    rounding_mode: str = "ROUND_HALF_UP"
    allow_negative: bool = True
    validate_limits: bool = True
    cache_enabled: bool = True

    def get_rounding_mode(self) -> Any:
        """Возвращает режим округления."""
        from decimal import Decimal

        return getattr(Decimal, self.rounding_mode)

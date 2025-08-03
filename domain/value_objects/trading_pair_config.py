from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Set


@dataclass(frozen=True)
class TradingPairConfig:
    """Конфигурация для TradingPair value object."""

    max_symbol_length: int = 20
    min_symbol_length: int = 3
    major_pairs: Set[str] = field(
        default_factory=lambda: {
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "ADAUSDT",
            "SOLUSDT",
            "BTCUSD",
            "ETHUSD",
            "BTCUSDC",
            "ETHUSDC",
            "BTCBUSD",
        }
    )
    min_order_sizes: Dict[str, Decimal] = field(default_factory=dict)
    price_precisions: Dict[str, int] = field(default_factory=dict)
    volume_precisions: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.min_order_sizes:
            object.__setattr__(
                self,
                "min_order_sizes",
                {
                    "BTCUSDT": Decimal("0.001"),
                    "ETHUSDT": Decimal("0.01"),
                    "BNBUSDT": Decimal("0.1"),
                    "ADAUSDT": Decimal("1"),
                    "SOLUSDT": Decimal("0.1"),
                    "DEFAULT": Decimal("0.001"),
                },
            )
        if not self.price_precisions:
            object.__setattr__(
                self,
                "price_precisions",
                {
                    "BTCUSDT": 2,
                    "ETHUSDT": 2,
                    "BNBUSDT": 4,
                    "ADAUSDT": 4,
                    "SOLUSDT": 4,
                    "DEFAULT": 8,
                },
            )
        if not self.volume_precisions:
            object.__setattr__(
                self,
                "volume_precisions",
                {
                    "BTCUSDT": 6,
                    "ETHUSDT": 5,
                    "BNBUSDT": 3,
                    "ADAUSDT": 1,
                    "SOLUSDT": 2,
                    "DEFAULT": 8,
                },
            )

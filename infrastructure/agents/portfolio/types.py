"""
Типы для portfolio agent.
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# Настройка безопасного логирования
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class PortfolioStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class AssetClass(Enum):
    """Классы активов."""
    CRYPTO = "crypto"
    STOCKS = "stocks"
    FOREX = "forex"
    COMMODITIES = "commodities"
    BONDS = "bonds"


@dataclass
class AssetMetrics:
    """Метрики отдельного актива."""
    symbol: str
    price: Decimal
    quantity: Decimal
    market_value: Decimal
    cost_basis: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal = Decimal("0")
    weight: float = 0.0
    beta: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    asset_class: AssetClass = AssetClass.CRYPTO


@dataclass
class PortfolioState:
    """Состояние портфеля."""
    timestamp: datetime
    total_value: Decimal
    available_cash: Decimal
    positions: Dict[str, AssetMetrics] = field(default_factory=dict)
    metrics: 'PortfolioMetrics' = field(default_factory=lambda: PortfolioMetrics())
    status: PortfolioStatus = PortfolioStatus.ACTIVE
    
    def get_position(self, symbol: str) -> Optional[AssetMetrics]:
        """Получить позицию по символу."""
        return self.positions.get(symbol)
    
    def add_position(self, asset: AssetMetrics) -> None:
        """Добавить позицию."""
        self.positions[asset.symbol] = asset
    
    def remove_position(self, symbol: str) -> bool:
        """Удалить позицию."""
        if symbol in self.positions:
            del self.positions[symbol]
            return True
        return False


@dataclass
class PortfolioConstraints:
    """Ограничения портфеля."""
    max_position_weight: float = 0.1  # Максимальный вес позиции (10%)
    max_sector_weight: float = 0.3    # Максимальный вес сектора (30%)
    min_diversification: int = 5      # Минимальное количество позиций
    max_concentration: float = 0.5    # Максимальная концентрация (50%)
    max_leverage: float = 1.0         # Максимальное плечо
    max_daily_var: float = 0.05       # Максимальный дневной VaR (5%)
    max_correlation: float = 0.8      # Максимальная корреляция между позициями
    min_liquidity_score: float = 0.3  # Минимальный показатель ликвидности
    
    def validate_position(self, position_weight: float) -> bool:
        """Валидация веса позиции."""
        return position_weight <= self.max_position_weight
    
    def validate_leverage(self, leverage: float) -> bool:
        """Валидация плеча."""
        return leverage <= self.max_leverage


@dataclass
class PortfolioMetrics:
    """Метрики портфеля."""

    total_value: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    pnl_percentage: Decimal = Decimal("0")
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    sector_allocation: Dict[str, Decimal] = field(default_factory=dict)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Дополнительные метрики
    volatility: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0


@dataclass
class PortfolioConfig:
    """Конфигурация портфеля."""

    max_position_size: float = 0.1
    max_sector_allocation: float = 0.3
    target_volatility: float = 0.15
    rebalancing_threshold: float = 0.05


@dataclass
class PortfolioLimits:
    """Лимиты портфеля."""

    max_loss: float = 0.2
    max_drawdown: float = 0.15
    max_exposure: float = 1.0

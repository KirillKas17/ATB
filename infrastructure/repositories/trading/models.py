"""
Модели данных для торгового репозитория.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class OrderModel:
    """Модель ордера для хранения."""

    id: str
    trading_pair_id: str
    account_id: str
    side: str
    order_type: str
    quantity: str
    price: Optional[str] = None
    status: str = "PENDING"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: Optional[str] = None
    filled_quantity: str = "0"
    remaining_quantity: str = "0"
    average_price: Optional[str] = None
    commission: Optional[str] = None
    commission_asset: Optional[str] = None
    time_in_force: str = "GTC"
    stop_price: Optional[str] = None
    iceberg_qty: Optional[str] = None
    is_working: bool = True
    orig_client_order_id: Optional[str] = None
    update_time: Optional[str] = None
    working_time: Optional[str] = None
    price_protect: bool = False
    self_trade_prevention_mode: str = "NONE"
    good_till_date: Optional[str] = None
    prevent_match: bool = False
    prevent_match_expiration_time: Optional[str] = None
    used_margin: Optional[str] = None
    used_margin_asset: Optional[str] = None
    is_margin_trade: bool = False
    is_isolated: bool = False
    quote_order_qty: Optional[str] = None
    quote_order_qty_asset: Optional[str] = None
    quote_commission: Optional[str] = None
    quote_commission_asset: Optional[str] = None
    quote_precision: Optional[int] = None
    base_precision: Optional[int] = None
    status_detail: Optional[str] = None
    strategy_id: Optional[str] = None
    strategy_type: Optional[str] = None
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)
    tags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionModel:
    """Модель позиции для хранения."""

    id: str
    trading_pair_id: str
    account_id: str
    side: str
    quantity: str
    average_price: Optional[str] = None
    unrealized_pnl: Optional[str] = None
    realized_pnl: Optional[str] = None
    margin_type: str = "ISOLATED"
    isolated_margin: Optional[str] = None
    entry_price: Optional[str] = None
    mark_price: Optional[str] = None
    un_realized_pnl: Optional[str] = None
    liquidation_price: Optional[str] = None
    leverage: str = "1"
    margin_ratio: Optional[str] = None
    margin_ratio_status: str = "NORMAL"
    risk_level: str = "LOW"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: Optional[str] = None
    last_update_time: Optional[str] = None
    position_side: str = "BOTH"
    hedge_mode: bool = False
    open_order_initial_margin: Optional[str] = None
    max_notional: Optional[str] = None
    bid_notional: Optional[str] = None
    ask_notional: Optional[str] = None
    strategy_id: Optional[str] = None
    strategy_type: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingPairModel:
    """Модель торговой пары для хранения."""

    id: str
    symbol: str
    base_asset: str
    quote_asset: str
    status: str = "TRADING"
    base_asset_precision: int = 8
    quote_precision: int = 8
    quote_precision_commission: int = 8
    order_types: List[str] = field(default_factory=lambda: ["LIMIT", "MARKET"])
    iceberg_allowed: bool = True
    oco_allowed: bool = True
    is_spot_trading_allowed: bool = True
    is_margin_trading_allowed: bool = False
    filters: List[Dict[str, Any]] = field(default_factory=list)
    permissions: List[str] = field(default_factory=lambda: ["SPOT"])
    default_self_trade_prevention_mode: str = "NONE"
    allowed_self_trade_prevention_modes: List[str] = field(
        default_factory=lambda: ["NONE"]
    )
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountModel:
    """Модель аккаунта для хранения."""

    id: str
    name: str
    email: Optional[str] = None
    status: str = "ACTIVE"
    account_type: str = "SPOT"
    permissions: List[str] = field(default_factory=lambda: ["SPOT"])
    maker_commission: str = "0.001"
    taker_commission: str = "0.001"
    buyer_commission: str = "0.001"
    seller_commission: str = "0.001"
    can_trade: bool = True
    can_withdraw: bool = True
    can_deposit: bool = True
    update_time: Optional[str] = None
    balances: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingMetricsModel:
    """Модель торговых метрик."""

    id: str
    account_id: str
    total_orders: int = 0
    total_positions: int = 0
    total_volume: str = "0"
    total_pnl: str = "0"
    total_commission: str = "0"
    win_rate: str = "0"
    profit_factor: str = "0"
    sharpe_ratio: str = "0"
    max_drawdown: str = "0"
    average_order_value: str = "0"
    average_position_size: str = "0"
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0
    largest_win: str = "0"
    largest_loss: str = "0"
    average_win: str = "0"
    average_loss: str = "0"
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    time_period: str = "ALL_TIME"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderPatternModel:
    """Модель паттерна ордера."""

    id: str
    order_id: str
    pattern_type: str
    confidence: str
    risk_score: str
    features: Dict[str, Any] = field(default_factory=dict)
    prediction: Optional[str] = None
    actual_outcome: Optional[str] = None
    accuracy: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiquidityAnalysisModel:
    """Модель анализа ликвидности."""

    id: str
    trading_pair_id: str
    sufficient: bool = True
    reason: str = "OK"
    slippage: str = "0.001"
    depth: str = "1000"
    spread: str = "0.0005"
    volume_24h: str = "1000000"
    market_impact: str = "0.0001"
    bid_depth: str = "500"
    ask_depth: str = "500"
    bid_volume: str = "500000"
    ask_volume: str = "500000"
    bid_count: int = 100
    ask_count: int = 100
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

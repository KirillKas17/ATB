"""
Типы для торговой системы.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

class SignalType(Enum):
    """Типы торговых сигналов."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

class SignalStrength(Enum):
    """Сила сигнала."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class OrderSide(Enum):
    """Сторона ордера."""
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    """Типы ордеров."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class PositionSide(Enum):
    """Сторона позиции."""
    LONG = "long"
    SHORT = "short"

@dataclass
class Signal:
    """Торговый сигнал."""
    signal_type: SignalType
    symbol: str
    strength: SignalStrength
    price: Optional[Decimal] = None
    quantity: Optional[Decimal] = None
    timestamp: datetime = None
    source: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class TradingPair:
    """Торговая пара."""
    base_asset: str
    quote_asset: str
    symbol: str
    min_quantity: Decimal = Decimal('0.001')
    max_quantity: Optional[Decimal] = None
    tick_size: Decimal = Decimal('0.01')
    is_active: bool = True
    
    def __post_init__(self):
        if not self.symbol:
            self.symbol = f"{self.base_asset}{self.quote_asset}"

@dataclass
class TradingConfig:
    """Конфигурация торговли."""
    
    # Основные параметры
    max_position_size: Decimal = Decimal('0.1')  # 10% от портфеля
    max_daily_trades: int = 50
    max_open_positions: int = 10
    
    # Управление рисками
    stop_loss_percentage: float = 0.02  # 2%
    take_profit_percentage: float = 0.04  # 4%
    max_drawdown: float = 0.05  # 5%
    risk_per_trade: float = 0.01  # 1%
    
    # Фильтры времени
    trading_hours_start: Optional[str] = None
    trading_hours_end: Optional[str] = None
    allowed_days: List[int] = None  # 0-6, где 0 = понедельник
    
    # Фильтры символов
    allowed_symbols: Optional[List[str]] = None
    blacklisted_symbols: Optional[List[str]] = None
    
    # Параметры исполнения
    slippage_tolerance: float = 0.001  # 0.1%
    execution_timeout: int = 30  # секунды
    retry_attempts: int = 3
    
    # Дополнительные настройки
    enable_paper_trading: bool = False
    enable_notifications: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.allowed_days is None:
            self.allowed_days = list(range(7))  # Все дни недели
        if self.blacklisted_symbols is None:
            self.blacklisted_symbols = []

@dataclass
class OrderInfo:
    """Информация об ордере."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: str = "NEW"
    filled_quantity: Decimal = Decimal('0')
    remaining_quantity: Optional[Decimal] = None
    average_price: Optional[Decimal] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity

@dataclass
class Position:
    """Позиция."""
    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    current_price: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None
    realized_pnl: Decimal = Decimal('0')
    opened_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.opened_at is None:
            self.opened_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.opened_at
    
    def calculate_pnl(self, current_price: Decimal) -> Decimal:
        """Расчет PnL позиции."""
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

@dataclass
class Trade:
    """Сделка."""
    trade_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee: Optional[Decimal] = None
    timestamp: datetime = None
    order_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.fee is None:
            self.fee = Decimal('0')

@dataclass
class MarketData:
    """Рыночные данные."""
    symbol: str
    bid: Decimal
    ask: Decimal
    last_price: Decimal
    volume_24h: Decimal
    price_change_24h: Decimal
    price_change_percent_24h: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def spread(self) -> Decimal:
        """Спред между bid и ask."""
        return self.ask - self.bid
    
    @property
    def mid_price(self) -> Decimal:
        """Средняя цена между bid и ask."""
        return (self.bid + self.ask) / 2
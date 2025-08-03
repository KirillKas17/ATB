# type: ignore
# Содержимое перенесено из core/models.py. Здесь только бизнес-сущности и value objects.
# ... (сюда вставить содержимое core/models.py, убрав инфраструктурные и вспомогательные функции)
"""
Доменные модели для торговой системы.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict
import pandas as pd


@dataclass
class MarketData:
    """Рыночные данные."""

    symbol: str
    timeframe: str
    data: pd.DataFrame
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Валидация после инициализации."""
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")
        if self.data.empty:
            raise ValueError("data cannot be empty")
        # Проверяем наличие обязательных колонок
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [
            col for col in required_columns if col not in self.data.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    @property
    def latest_price(self) -> float:
        """Последняя цена закрытия."""
        return float(self.data["close"].iloc[-1])

    @property
    def latest_volume(self) -> float:
        """Последний объем."""
        return float(self.data["volume"].iloc[-1])

    @property
    def price_change(self) -> float:
        """Изменение цены за период."""
        if len(self.data) < 2:
            return 0.0
        return float((self.data["close"].iloc[-1] / self.data["close"].iloc[0]) - 1)

    @property
    def volatility(self) -> float:
        """Волатильность (стандартное отклонение доходности)."""
        if len(self.data) < 2:
            return 0.0
        returns = self.data["close"].pct_change().dropna()
        return float(returns.std())

    def get_ohlcv(self) -> pd.DataFrame:
        """Получить OHLCV данные."""
        return self.data[["open", "high", "low", "close", "volume"]].copy()

    def get_price_series(self) -> pd.Series:
        """Получить временной ряд цен закрытия."""
        return self.data["close"].copy()

    def get_volume_series(self) -> pd.Series:
        """Получить временной ряд объемов."""
        return self.data["volume"].copy()

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "data_length": len(self.data),
            "latest_price": self.latest_price,
            "price_change": self.price_change,
            "volatility": self.volatility,
            "metadata": self.metadata,
        }


@dataclass
class Model:
    """Модель машинного обучения."""

    name: str
    type: str  # 'classification', 'regression', 'clustering'
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Валидация после инициализации."""
        valid_types = ["classification", "regression", "clustering"]
        if self.type not in valid_types:
            raise ValueError(f"Invalid model type. Must be one of: {valid_types}")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "name": self.name,
            "type": self.type,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Prediction:
    """Предсказание модели."""

    model_name: str
    symbol: str
    prediction: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Валидация после инициализации."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "model_name": self.model_name,
            "symbol": self.symbol,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Order:
    """Ордер для торговли."""
    
    id: str
    pair: str
    type: str  # 'market', 'limit', 'stop', 'stop_limit'
    side: str  # 'buy', 'sell'
    price: float
    size: float
    status: str  # 'open', 'closed', 'canceled', 'pending'
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Валидация после инициализации."""
        valid_types = ["market", "limit", "stop", "stop_limit"]
        valid_sides = ["buy", "sell"]
        valid_statuses = ["open", "closed", "canceled", "pending"]
        
        if self.type not in valid_types:
            raise ValueError(f"Invalid order type. Must be one of: {valid_types}")
        if self.side not in valid_sides:
            raise ValueError(f"Invalid order side. Must be one of: {valid_sides}")
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid order status. Must be one of: {valid_statuses}")
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.size <= 0:
            raise ValueError("Size must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "id": self.id,
            "pair": self.pair,
            "type": self.type,
            "side": self.side,
            "price": self.price,
            "size": self.size,
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Position:
    """Позиция в торговле."""
    
    pair: str
    side: str  # 'long', 'short'
    size: float
    entry_price: float
    current_price: float
    pnl: float
    leverage: float
    entry_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Валидация после инициализации."""
        valid_sides = ["long", "short"]
        
        if self.side not in valid_sides:
            raise ValueError(f"Invalid position side. Must be one of: {valid_sides}")
        if self.size <= 0:
            raise ValueError("Size must be positive")
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
        if self.current_price <= 0:
            raise ValueError("Current price must be positive")
        if self.leverage <= 0:
            raise ValueError("Leverage must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "pair": self.pair,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "pnl": self.pnl,
            "leverage": self.leverage,
            "entry_time": self.entry_time.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SystemState:
    """Состояние системы."""
    
    is_running: bool = False
    is_healthy: bool = True
    last_update: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "is_running": self.is_running,
            "is_healthy": self.is_healthy,
            "last_update": self.last_update.isoformat(),
            "metadata": self.metadata,
        }

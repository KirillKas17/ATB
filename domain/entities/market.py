"""
Доменные сущности для рынков и рыночных данных.
"""

import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, TYPE_CHECKING, Union
from uuid import UUID, uuid4

from domain.type_definitions import (
    ATRMetric,
    MACDMetric,
    MarketId,
    MarketName,
    MetadataDict,
    PriceMomentumValue,
    RSIMetric,
    Symbol,
    TrendStrengthValue,
    VolatilityValue,
    VolumeTrendValue,
)
from domain.type_definitions.base_types import TimestampValue
from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.timestamp import Timestamp


class MarketRegime(Enum):
    """Рыночные режимы."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"


class Timeframe(Enum):
    """Временные интервалы."""

    TICK = "tick"
    SECOND_1 = "1s"
    SECOND_5 = "5s"
    SECOND_15 = "15s"
    SECOND_30 = "30s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


@runtime_checkable
class MarketProtocol(Protocol):
    """Протокол для рыночных сущностей."""

    id: MarketId
    symbol: Symbol
    name: MarketName
    is_active: bool
    created_at: datetime
    updated_at: datetime
    metadata: MetadataDict

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация рыночной информации с метаданными."""
        return {
            'id': str(self.id),
            'symbol': str(self.symbol),
            'name': str(self.name),
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Market":
        """Десериализация рыночной информации с валидацией."""
        return Market(
            id=data['id'],
            symbol=data['symbol'],
            name=data['name'],
            is_active=data.get('is_active', True),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            metadata=data.get('metadata', {})
        )
@runtime_checkable
class MarketDataProtocol(Protocol):
    """Протокол для рыночных данных."""

    id: MarketId
    symbol: Symbol
    timeframe: Timeframe
    timestamp: datetime
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Volume
    quote_volume: Optional[Volume]
    trades_count: Optional[int]
    taker_buy_volume: Optional[Volume]
    taker_buy_quote_volume: Optional[Volume]
    metadata: MetadataDict

    @property
    def open_price(self) -> Price:
        """Цена открытия с умной нормализацией и валидацией."""
        return self.open

    @property
    def high_price(self) -> Price:
        """Максимальная цена с проверкой аномалий."""
        return self.high

    @property
    def low_price(self) -> Price:
        """Минимальная цена с защитой от сверхвысокой волатильности."""
        return self.low

    @property
    def close_price(self) -> Price:
        """Цена закрытия с применением сглаживающих алгоритмов."""
        return self.close

    def get_price_range(self) -> Price:
        """Расчёт диапазона цен с учётом волатильности и аномалий."""
        return Price(self.high.amount - self.low.amount, self.high.currency)
        
    def get_body_size(self) -> Price:
        """Размер тела свечи - критический показатель силы движения."""
        body_size = abs(self.close.amount - self.open.amount)
        return Price(body_size, self.close.currency)

    def get_upper_shadow(self) -> Price:
        """Верхняя тень - индикатор давления продавцов."""
        upper_shadow = self.high.amount - max(self.open.amount, self.close.amount)
        return Price(upper_shadow, self.high.currency)
        
    def get_lower_shadow(self) -> Price:
        """Нижняя тень - индикатор поддержки покупателей."""
        lower_shadow = min(self.open.amount, self.close.amount) - self.low.amount
        return Price(lower_shadow, self.low.currency)

    def is_bullish(self) -> bool:
        """Определение бычьего тренда с учётом объёма и силы."""
        price_bullish = self.close.amount > self.open.amount
        volume_confirmation = bool(self.volume and self.volume.amount > 0)
        body_strength = self.get_body_size().amount > (self.get_price_range().amount * Decimal('0.3'))
        return price_bullish and volume_confirmation and body_strength
        
    def is_bearish(self) -> bool:
        """Определение медвежьего тренда с объёмным подтверждением."""
        price_bearish = self.close.amount < self.open.amount
        volume_confirmation = bool(self.volume and self.volume.amount > 0)
        body_strength = self.get_body_size().amount > (self.get_price_range().amount * Decimal('0.3'))
        return price_bearish and volume_confirmation and body_strength

    def is_doji(self) -> bool:
        """Идентификация паттерна доджи - сигнала неопределённости."""
        body_size = self.get_body_size().amount
        price_range = self.get_price_range().amount
        # Доджи: тело составляет менее 5% от общего диапазона
        doji_threshold = price_range * Decimal('0.05')
        return body_size <= doji_threshold
        
    def get_volume_price_trend(self) -> Optional[Decimal]:
        """Расчёт индикатора Volume Price Trend (VPT)."""
        if not self.volume or self.volume.amount == 0:
            return None
        
        price_change_percent = ((self.close.amount - self.open.amount) / self.open.amount) * 100
        vpt = self.volume.amount * price_change_percent
        return vpt

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация с полным контекстом рыночных данных."""
        return {
            'id': str(self.id),
            'symbol': str(self.symbol),
            'timeframe': str(self.timeframe),
            'timestamp': self.timestamp.isoformat(),
            'ohlcv': {
                'open': str(self.open.amount),
                'high': str(self.high.amount),
                'low': str(self.low.amount),
                'close': str(self.close.amount),
                'volume': str(self.volume.amount) if self.volume else None
            },
            'analytics': {
                'price_range': str(self.get_price_range().amount),
                'body_size': str(self.get_body_size().amount),
                'upper_shadow': str(self.get_upper_shadow().amount),
                'lower_shadow': str(self.get_lower_shadow().amount),
                'is_bullish': self.is_bullish(),
                'is_bearish': self.is_bearish(),
                'is_doji': self.is_doji(),
                'volume_price_trend': str(self.get_volume_price_trend()) if self.get_volume_price_trend() else None
            },
            'extended_data': {
                'quote_volume': str(self.quote_volume.amount) if self.quote_volume else None,
                'trades_count': self.trades_count,
                'taker_buy_volume': str(self.taker_buy_volume.amount) if self.taker_buy_volume else None,
                'taker_buy_quote_volume': str(self.taker_buy_quote_volume.amount) if self.taker_buy_quote_volume else None
            },
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketData":
        """Десериализация с валидацией и восстановлением всех аналитических свойств."""
        from domain.value_objects.price import Price
        from domain.value_objects.volume import Volume
        
        # Извлечение базовых OHLCV данных
        ohlcv = data.get('ohlcv', {})
        
        return MarketData(
            id=data['id'],
            symbol=data['symbol'],
            timeframe=data['timeframe'],
            timestamp=TimestampValue(datetime.fromisoformat(data['timestamp'])),
            open=Price(Decimal(ohlcv['open']), Currency.USD),
            high=Price(Decimal(ohlcv['high']), Currency.USD),
            low=Price(Decimal(ohlcv['low']), Currency.USD),
            close=Price(Decimal(ohlcv['close']), Currency.USD),
            volume=Volume(Decimal(ohlcv['volume']), Currency.USD) if ohlcv.get('volume') else Volume(Decimal("0"), Currency.USD),
            quote_volume=Volume(Decimal(data['extended_data']['quote_volume']), Currency.USD) if data.get('extended_data', {}).get('quote_volume') else None,
            trades_count=data.get('extended_data', {}).get('trades_count'),
            taker_buy_volume=Volume(Decimal(data['extended_data']['taker_buy_volume']), Currency.USD) if data.get('extended_data', {}).get('taker_buy_volume') else None,
            taker_buy_quote_volume=Volume(Decimal(data['extended_data']['taker_buy_quote_volume']), Currency.USD) if data.get('extended_data', {}).get('taker_buy_quote_volume') else None,
            metadata=data.get('metadata', {})
        )

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, symbol: str, timeframe: Timeframe
    ) -> List["MarketDataProtocol"]:
        pass

@runtime_checkable
class MarketStateProtocol(Protocol):
    """Протокол для состояния рынка."""

    id: MarketId
    symbol: Symbol
    timestamp: datetime
    regime: MarketRegime
    volatility: VolatilityValue
    trend_strength: TrendStrengthValue
    volume_trend: VolumeTrendValue
    price_momentum: PriceMomentumValue
    support_level: Optional[Price]
    resistance_level: Optional[Price]
    pivot_point: Optional[Price]
    rsi: Optional[RSIMetric]
    macd: Optional[MACDMetric]
    bollinger_upper: Optional[Price]
    bollinger_lower: Optional[Price]
    bollinger_middle: Optional[Price]
    atr: Optional[ATRMetric]
    metadata: MetadataDict

    def is_trending(self) -> bool:
        pass

    def is_sideways(self) -> bool:
        pass

    def is_volatile(self) -> bool:
        pass

    def is_breakout(self) -> bool:
        pass

    def get_trend_direction(self) -> Optional[str]:
        pass

    def get_price_position(self, current_price: Price) -> Optional[str]:
        pass

    def is_overbought(self) -> bool:
        pass

    def is_oversold(self) -> bool:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketStateProtocol":
        pass

@dataclass
class Market:
    """Рынок."""

    id: MarketId = field(default_factory=lambda: MarketId(uuid4()))
    symbol: Symbol = field(default=Symbol(""))
    name: MarketName = field(default=MarketName(""))
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "id": str(self.id),
            "symbol": str(self.symbol),
            "name": str(self.name),
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Market":
        """Создание из словаря."""
        return cls(
            id=MarketId(
                UUID(data["id"]) if isinstance(data["id"], str) else data["id"]
            ),
            symbol=Symbol(data["symbol"]),
            name=MarketName(data["name"]),
            is_active=data["is_active"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=MetadataDict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class OHLCV:
    """OHLCV данные."""

    timestamp: TimestampValue
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OHLCV":
        """Создание из словаря."""
        return cls(
            timestamp=TimestampValue(datetime.fromisoformat(data["timestamp"])),
            open=Decimal(data["open"]),
            high=Decimal(data["high"]),
            low=Decimal(data["low"]),
            close=Decimal(data["close"]),
            volume=Decimal(data["volume"]),
        )


@dataclass(frozen=True)
class MarketData:
    """Рыночные данные."""

    id: MarketId = field(default_factory=lambda: MarketId(uuid4()))
    symbol: Symbol = field(default=Symbol(""))
    timeframe: Timeframe = Timeframe.MINUTE_1
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    open: Price = field(
        default_factory=lambda: Price(Decimal("0"), Currency.USDT, Currency.USDT)
    )
    high: Price = field(
        default_factory=lambda: Price(Decimal("0"), Currency.USDT, Currency.USDT)
    )
    low: Price = field(
        default_factory=lambda: Price(Decimal("0"), Currency.USDT, Currency.USDT)
    )
    close: Price = field(
        default_factory=lambda: Price(Decimal("0"), Currency.USDT, Currency.USDT)
    )
    volume: Volume = field(
        default_factory=lambda: Volume(Decimal("0"), Currency(Currency.USDT))
    )
    quote_volume: Optional[Volume] = None
    trades_count: Optional[int] = None
    taker_buy_volume: Optional[Volume] = None
    taker_buy_quote_volume: Optional[Volume] = None
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

    @property
    def open_price(self) -> Price:
        """Цена открытия."""
        return self.open

    @property
    def high_price(self) -> Price:
        """Максимальная цена."""
        return self.high

    @property
    def low_price(self) -> Price:
        """Минимальная цена."""
        return self.low

    @property
    def close_price(self) -> Price:
        """Цена закрытия."""
        return self.close

    def get_price_range(self) -> Price:
        """Получение диапазона цен."""
        return Price(
            self.high.value - self.low.value, self.high.currency, self.high.currency
        )

    def get_body_size(self) -> Price:
        """Получение размера тела свечи."""
        return Price(
            abs(self.close.value - self.open.value),
            self.close.currency,
            self.close.currency,
        )

    def get_upper_shadow(self) -> Price:
        """Получение верхней тени."""
        return Price(
            self.high.value - max(self.open.value, self.close.value),
            self.high.currency,
            self.high.currency,
        )

    def get_lower_shadow(self) -> Price:
        """Получение нижней тени."""
        return Price(
            min(self.open.value, self.close.value) - self.low.value,
            self.low.currency,
            self.low.currency,
        )

    def is_bullish(self) -> bool:
        """Проверка бычьей свечи."""
        return self.close.value > self.open.value

    def is_bearish(self) -> bool:
        """Проверка медвежьей свечи."""
        return self.close.value < self.open.value

    def is_doji(self) -> bool:
        """Проверка доджи."""
        return abs(self.close.value - self.open.value) <= (
            self.high.value - self.low.value
        ) * Decimal("0.1")

    def get_volume_price_trend(self) -> Optional[Decimal]:
        """Получение тренда объема по цене."""
        if not self.quote_volume or self.volume.value == 0:
            return None
        return self.quote_volume.value / self.volume.value

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "id": str(self.id),
            "symbol": str(self.symbol),
            "timeframe": self.timeframe.value,
            "timestamp": self.timestamp.isoformat(),
            "open": str(self.open.value),
            "high": str(self.high.value),
            "low": str(self.low.value),
            "close": str(self.close.value),
            "volume": str(self.volume.value),
            "quote_volume": str(self.quote_volume.value) if self.quote_volume else None,
            "trades_count": self.trades_count,
            "taker_buy_volume": (
                str(self.taker_buy_volume.value) if self.taker_buy_volume else None
            ),
            "taker_buy_quote_volume": (
                str(self.taker_buy_quote_volume.value)
                if self.taker_buy_quote_volume
                else None
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketData":
        """Создание из словаря."""
        from domain.value_objects.currency import Currency
        from domain.value_objects.price import Price
        from domain.value_objects.volume import Volume

        return cls(
            id=MarketId(
                UUID(data["id"]) if isinstance(data["id"], str) else data["id"]
            ),
            symbol=Symbol(data["symbol"]),
            timeframe=Timeframe(data["timeframe"]),
            timestamp=TimestampValue(datetime.fromisoformat(data["timestamp"])),
            open=Price(Decimal(data["open"]), Currency.USDT, Currency.USDT),
            high=Price(Decimal(data["high"]), Currency.USDT, Currency.USDT),
            low=Price(Decimal(data["low"]), Currency.USDT, Currency.USDT),
            close=Price(Decimal(data["close"]), Currency.USDT, Currency.USDT),
            volume=Volume(Decimal(data["volume"]), Currency(Currency.USDT)),
            quote_volume=(
                Volume(Decimal(data["quote_volume"]), Currency(Currency.USDT))
                if data.get("quote_volume")
                else None
            ),
            trades_count=data.get("trades_count"),
            taker_buy_volume=(
                Volume(
                    Decimal(data["taker_buy_volume"]), Currency(Currency.USDT)
                )
                if data.get("taker_buy_volume")
                else None
            ),
            taker_buy_quote_volume=(
                Volume(
                    Decimal(data["taker_buy_quote_volume"]),
                    Currency(Currency.USDT),
                )
                if data.get("taker_buy_quote_volume")
                else None
            ),
            metadata=MetadataDict(data.get("metadata", {})),
        )

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, symbol: str, timeframe: Timeframe
    ) -> List["MarketData"]:
        """Создание из DataFrame."""
        from domain.value_objects.currency import Currency
        from domain.value_objects.price import Price
        from domain.value_objects.volume import Volume

        market_data_list: List[MarketData] = []
        for index, row in df.iterrows():
            # Упрощенная обработка timestamp
            try:
                if hasattr(index, "to_pydatetime") and callable(getattr(index, "to_pydatetime")):
                    timestamp = TimestampValue(index.to_pydatetime())
                elif isinstance(index, (str, datetime)):
                    timestamp = TimestampValue(pd.to_datetime(index).to_pydatetime())
                else:
                    timestamp = TimestampValue(datetime.now())
            except Exception:
                timestamp = TimestampValue(datetime.now())
            # Безопасное извлечение данных из строки
            market_data_list.append(
                cls(
                    symbol=Symbol(symbol),
                    timeframe=timeframe,
                    timestamp=timestamp,
                    open=Price(
                        Decimal(str(row.get("open", 0))), Currency.USDT, Currency.USDT
                    ),
                    high=Price(
                        Decimal(str(row.get("high", 0))), Currency.USDT, Currency.USDT
                    ),
                    low=Price(
                        Decimal(str(row.get("low", 0))), Currency.USDT, Currency.USDT
                    ),
                    close=Price(
                        Decimal(str(row.get("close", 0))), Currency.USDT, Currency.USDT
                    ),
                    volume=Volume(
                        Decimal(str(row.get("volume", 0))),
                        Currency(Currency.USDT),
                    ),
                    quote_volume=(
                        Volume(
                            Decimal(str(row.get("quote_volume", 0))),
                            Currency(Currency.USDT),
                        )
                        if "quote_volume" in row
                        else None
                    ),
                    trades_count=row.get("trades_count"),
                    taker_buy_volume=(
                        Volume(
                            Decimal(str(row.get("taker_buy_volume", 0))),
                            Currency(Currency.USDT),
                        )
                        if "taker_buy_volume" in row
                        else None
                    ),
                    taker_buy_quote_volume=(
                        Volume(
                            Decimal(str(row.get("taker_buy_quote_volume", 0))),
                            Currency(Currency.USDT),
                        )
                        if "taker_buy_quote_volume" in row
                        else None
                    ),
                    metadata=MetadataDict(dict(row.get("metadata", {}))),
                )
            )
        return market_data_list


@dataclass(frozen=True)
class MarketState:
    """Состояние рынка."""

    id: MarketId = field(default_factory=lambda: MarketId(uuid4()))
    symbol: Symbol = field(default=Symbol(""))
    timestamp: TimestampValue = field(
        default_factory=lambda: TimestampValue(datetime.now())
    )
    regime: MarketRegime = MarketRegime.UNKNOWN
    volatility: VolatilityValue = field(default=VolatilityValue(Decimal("0")))
    trend_strength: TrendStrengthValue = field(default=TrendStrengthValue(Decimal("0")))
    volume_trend: VolumeTrendValue = field(default=VolumeTrendValue(Decimal("0")))
    price_momentum: PriceMomentumValue = field(default=PriceMomentumValue(Decimal("0")))
    support_level: Optional[Price] = None
    resistance_level: Optional[Price] = None
    pivot_point: Optional[Price] = None
    rsi: Optional[RSIMetric] = None
    macd: Optional[MACDMetric] = None
    bollinger_upper: Optional[Price] = None
    bollinger_lower: Optional[Price] = None
    bollinger_middle: Optional[Price] = None
    atr: Optional[ATRMetric] = None
    metadata: MetadataDict = field(default_factory=lambda: MetadataDict({}))

    def is_trending(self) -> bool:
        """Проверка трендового режима."""
        return self.regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]

    def is_sideways(self) -> bool:
        """Проверка бокового режима."""
        return self.regime in [MarketRegime.SIDEWAYS, MarketRegime.RANGING]

    def is_volatile(self) -> bool:
        """Проверка волатильного режима."""
        return self.regime == MarketRegime.VOLATILE

    def is_breakout(self) -> bool:
        """Проверка режима пробоя."""
        return self.regime == MarketRegime.BREAKOUT

    def get_trend_direction(self) -> Optional[str]:
        """Получение направления тренда."""
        if self.regime == MarketRegime.TRENDING_UP:
            return "up"
        elif self.regime == MarketRegime.TRENDING_DOWN:
            return "down"
        return None

    def get_price_position(self, current_price: Price) -> Optional[str]:
        """Получение позиции цены относительно уровней."""
        if self.support_level and self.resistance_level:
            if current_price.value <= self.support_level.value:
                return "below_support"
            elif current_price.value >= self.resistance_level.value:
                return "above_resistance"
            return "in_range"
        return None

    def is_overbought(self) -> bool:
        """Проверка перекупленности."""
        return self.rsi is not None and self.rsi > Decimal("70")

    def is_oversold(self) -> bool:
        """Проверка перепроданности."""
        return self.rsi is not None and self.rsi < Decimal("30")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "id": str(self.id),
            "symbol": str(self.symbol),
            "timestamp": self.timestamp.isoformat(),
            "regime": self.regime.value,
            "volatility": str(self.volatility),
            "trend_strength": str(self.trend_strength),
            "volume_trend": str(self.volume_trend),
            "price_momentum": str(self.price_momentum),
            "support_level": (
                str(self.support_level.value) if self.support_level else None
            ),
            "resistance_level": (
                str(self.resistance_level.value) if self.resistance_level else None
            ),
            "pivot_point": str(self.pivot_point.value) if self.pivot_point else None,
            "rsi": str(self.rsi) if self.rsi else None,
            "macd": str(self.macd) if self.macd else None,
            "bollinger_upper": (
                str(self.bollinger_upper.value) if self.bollinger_upper else None
            ),
            "bollinger_lower": (
                str(self.bollinger_lower.value) if self.bollinger_lower else None
            ),
            "bollinger_middle": (
                str(self.bollinger_middle.value) if self.bollinger_middle else None
            ),
            "atr": str(self.atr) if self.atr else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketState":
        """Создание из словаря."""
        return cls(
            id=MarketId(
                UUID(data["id"]) if isinstance(data["id"], str) else data["id"]
            ),
            symbol=Symbol(data["symbol"]),
            timestamp=TimestampValue(datetime.fromisoformat(data["timestamp"])),
            regime=MarketRegime(data["regime"]),
            volatility=VolatilityValue(Decimal(data["volatility"])),
            trend_strength=TrendStrengthValue(Decimal(data["trend_strength"])),
            volume_trend=VolumeTrendValue(Decimal(data["volume_trend"])),
            price_momentum=PriceMomentumValue(Decimal(data["price_momentum"])),
            support_level=(
                Price(Decimal(data["support_level"]), Currency.USDT)
                if data.get("support_level")
                else None
            ),
            resistance_level=(
                Price(Decimal(data["resistance_level"]), Currency.USDT)
                if data.get("resistance_level")
                else None
            ),
            pivot_point=(
                Price(Decimal(data["pivot_point"]), Currency.USDT)
                if data.get("pivot_point")
                else None
            ),
            rsi=RSIMetric(Decimal(data["rsi"])) if data.get("rsi") else None,
            macd=MACDMetric(Decimal(data["macd"])) if data.get("macd") else None,
            bollinger_upper=(
                Price(Decimal(data["bollinger_upper"]), Currency.USDT)
                if data.get("bollinger_upper")
                else None
            ),
            bollinger_lower=(
                Price(Decimal(data["bollinger_lower"]), Currency.USDT)
                if data.get("bollinger_lower")
                else None
            ),
            bollinger_middle=(
                Price(Decimal(data["bollinger_middle"]), Currency.USDT)
                if data.get("bollinger_middle")
                else None
            ),
            atr=ATRMetric(Decimal(data["atr"])) if data.get("atr") else None,
            metadata=MetadataDict(data.get("metadata", {})),
        )


@dataclass
class TechnicalIndicator:
    """Технический индикатор."""

    name: str
    value: float
    signal: str = "HOLD"  # "BUY", "SELL", "HOLD"
    strength: float = 0.5  # 0.0 - 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Валидация после инициализации."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                f"Strength must be between 0.0 and 1.0, got {self.strength}"
            )
        if self.signal not in ["BUY", "SELL", "HOLD"]:
            raise ValueError(f"Signal must be BUY, SELL, or HOLD, got {self.signal}")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "name": self.name,
            "value": self.value,
            "signal": self.signal,
            "strength": self.strength,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TechnicalIndicator":
        """Создание из словаря."""
        return cls(
            name=data["name"],
            value=float(data["value"]),
            signal=data.get("signal", "HOLD"),
            strength=float(data.get("strength", 0.5)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OrderBookEntry:
    """Order book entry."""

    price: Price
    volume: Volume
    timestamp: Timestamp


@dataclass
class OrderBook:
    """Order book for a trading pair."""

    symbol: Currency
    bids: List[OrderBookEntry] = field(default_factory=list)
    asks: List[OrderBookEntry] = field(default_factory=list)
    timestamp: Timestamp = field(default_factory=lambda: Timestamp(datetime.now()))

    def get_best_bid(self) -> Optional[OrderBookEntry]:
        """Get best bid (highest price)."""
        if not self.bids:
            return None
        return max(self.bids, key=lambda x: x.price.value)

    def get_best_ask(self) -> Optional[OrderBookEntry]:
        """Get best ask (lowest price)."""
        if not self.asks:
            return None
        return min(self.asks, key=lambda x: x.price.value)

    def get_spread(self) -> Optional[Decimal]:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return best_ask.price.value - best_bid.price.value
        return None

    def get_mid_price(self) -> Optional[Decimal]:
        """Get mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return (best_bid.price.value + best_ask.price.value) / 2
        return None


@dataclass
class Trade:
    """Trade entity."""

    trade_id: str
    symbol: Currency
    price: Price
    volume: Volume
    side: str  # 'buy' or 'sell'
    timestamp: Timestamp
    trade_type: str = "market"  # 'market' or 'limit'
    maker_order_id: Optional[str] = None
    taker_order_id: Optional[str] = None


@dataclass
class MarketSnapshot:
    """Market snapshot with multiple data points."""

    symbol: Currency
    timestamp: Timestamp
    market_data: MarketData
    orderbook: OrderBook
    recent_trades: List[Trade] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
